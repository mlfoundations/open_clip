"""
Constructs a webdataset for the 10M image-text pairs from iNat21, EOL and BIOSCAN-1M.

Uses the images from iNat21, EOL and BIOSCAN-1M.
We use the scientific name directly as the text.


FUTURE VERSIONS
1. Where possible, we use the common name as the caption. If not possible, we use the scientific name as the caption.
2. For each image, if we have a species-level label and a species-level description from EOL, we pair that description with the image. Otherwise use the common name. Otherwise use the scientific name.

Warnings:

/users/PAS1576/samuelstevens/projects/open_clip/.venv/lib/python3.10/site-packages/PIL/TiffImagePlugin.py:866: UserWarning: Truncated File Read
  warnings.warn(str(msg))

/users/PAS1576/samuelstevens/projects/open_clip/.venv/lib/python3.10/site-packages/PIL/Image.py:3157: DecompressionBombWarning: Image size (154508376 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
  warnings.warn(
"""
import csv
import json
import logging
import multiprocessing
import os
import re
import sqlite3
import tarfile
import uuid

from PIL import Image, ImageFile
from tqdm import tqdm

from imageomics import naming, wds, eol

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True

max_workers = 32

eol_root_dir = "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive"
inat_root_dir = "/fs/ess/PAS2136/foundation_model/inat21/raw/train"
bioscan_root_dir = "/fs/scratch/PAS2136/bioscan/cropped_256"
bioscan_metadata_path = (
    "/fs/scratch/PAS2136/bioscan/BIOSCAN_Insect_Dataset_metadata.jsonld"
)

resize_size = (224, 224)
output_dir = os.path.join(
    "/fs/ess/PAS2136/open_clip/data/evobio10m/", f"{resize_size[0]}x{resize_size[1]}"
)
os.makedirs(output_dir, exist_ok=True)

db_path = os.path.join(output_dir, "mapping.sqlite")
db_write_frequency = 1000


schema = """
CREATE TABLE IF NOT EXISTS eol (
    content_id INT NOT NULL,
    page_id INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS inat21 (
    image_id TEXT NOT NULL,
    cls_name TEXT NOT NULL,
    cls_num INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS bioscan (
    part INT NOT NULL,
    image_id TEXT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

PRAGMA journal_mode=WAL;  -- write-ahead log
"""


def get_db():
    db = sqlite3.connect(db_path, timeout=120)
    db.execute("PRAGMA busy_timeout = 120000;")  # 120 second timeout
    db.commit()
    db.executescript(schema)
    db.commit()
    return db


def get_global_id():
    return str(uuid.uuid4())


######################
# Encyclopedia of Life
######################


eol_insert_stmt = """
INSERT INTO eol
    (content_id, page_id, evobio10m_id)
VALUES
    (?, ?, ?);
"""


class EolNameLookup:
    media_cargo_archive_map_csv = (
        "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive_map.csv"
    )
    taxon_tab = "data/eol/dh21/taxon.tab"
    provider_ids_csv = "data/eol/provider_ids.csv"
    pages_csv = "data/eol/trait_bank/pages.csv"
    vernacularnames_csv = "data/eol/vernacularnames.csv"
    hierarchies_path = (
        "/fs/scratch/PAS2136/eol/data/interim/all_hierarchies_in_graph.csv"
    )
    scraped_page_ids_csv = "data/eol/scraped_page_ids.csv"

    def __init__(self):
        self.common = {}
        self.taxonomic = {}
        self.scientific = {}

        # scientific names
        with open(self.provider_ids_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc="provider_ids.csv"):
                if not row["page_id"] or not row["preferred_canonical_for_page"]:
                    continue
                page_id = int(row["page_id"])
                if page_id not in self.scientific:
                    self.scientific[page_id] = row["preferred_canonical_for_page"]

        # scientific names
        with open(self.taxon_tab) as fd:
            reader = csv.DictReader(fd, delimiter="\t")
            for row in tqdm(reader, desc="taxon.tab"):
                if not row["eolID"] or not row["scientificName"]:
                    continue
                page_id = int(row["eolID"])
                if page_id not in self.scientific:
                    self.scientific[page_id] = row["scientificName"]

        with open(self.pages_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc="trait_bank/pages.csv"):
                if not row["page_id"] or not row["canonical"]:
                    continue
                page_id = int(row["page_id"])
                if page_id not in self.scientific:
                    self.scientific[page_id] = row["canonical"]

        vernacular_names = eol.VernacularNameLookup(self.vernacularnames_csv)
        for page_id in vernacular_names:
            name = vernacular_names[page_id]
            if page_id not in self.scientific:
                self.scientific[page_id] = name.canonical

            if page_id not in self.common:
                self.common[page_id] = name.common

        with open(self.hierarchies_path) as fd:
            reader = csv.reader(fd)
            next(reader)  # skip header row
            for page_id, raw_canonical, _, _ in tqdm(
                reader, desc=self.hierarchies_path
            ):
                canonical_chain = raw_canonical.split("->")
                page_id = int(page_id)
                if page_id not in self.taxonomic:
                    self.taxonomic[page_id] = " ".join(canonical_chain)

                if page_id not in self.scientific:
                    self.scientific[page_id] = " ".join(canonical_chain[-2:])

        with open(self.scraped_page_ids_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc="scraped_page_ids.csv"):
                page_id = int(row["page_id"])

                if page_id not in self.scientific:
                    self.scientific[page_id] = row["scientific_name"]

    def get(self, page_id, *, preferred=("common", "taxonomic", "scientific")):
        for kind in preferred:
            if not hasattr(self, kind):
                raise ValueError(
                    f"{self.__class__.__name__} has no lookup called {kind}!"
                )

            lookup = getattr(self, kind)
            if page_id in lookup:
                return lookup[page_id]

        # Couldn't find it
        return None


def copy_eol_from_tar(sink, imgset_path):
    """
    Copies all the files out of an imgset (.tar.gz file), resizes the images, then
    copies them to a new, unique path in images/. Stores the mapping between image id
    and the new id in a sqlite database.

    Note: the sqlite database cannot be on an NFS drive, because multiple processes
    cannot write to a sqlite database on an NFS drive without corrupting the database.
    """
    logger = logging.getLogger(f"p{os.getpid()}")

    # each process get its own db connection.
    db = get_db()

    insert_values = []

    # r|gz indcates reading from a gzipped file, streaming only
    with tarfile.open(imgset_path, "r|gz") as tar:
        for i, member in enumerate(tar):
            eol_img = eol.ImageFilename.from_filename(member.name)
            scientific_name = eol_name_lookup.get(
                eol_img.page_id, preferred=("scientific",)
            )
            if not scientific_name:
                continue

            global_id = get_global_id()

            insert_values.append((eol_img.content_id, eol_img.page_id, global_id))

            file = tar.extractfile(member)
            try:
                img = Image.open(file).resize(resize_size)
            except OSError as err:
                logger.warning(
                    "Error opening file. Skipping. [tar: %s, err: %s]", imgset_path, err
                )
                continue

            sink.write({"__key__": global_id, "jpg": img, "txt": scientific_name})

            if i % db_write_frequency == 0:
                try:
                    db.executemany(eol_insert_stmt, insert_values)
                    db.commit()
                    # If we throw an err on executemany, then we don't reset
                    # insert_values so we can try again next round.
                    insert_values = []
                except sqlite3.OperationalError as err:
                    logger.warning(
                        "Error inserting. [len: %d, err: %s]", len(insert_values), err
                    )

    # flush any leftover values
    db.executemany(eol_insert_stmt, insert_values)
    db.commit()
    db.close()


########
# INAT21
########

inat_insert_stmt = """
INSERT INTO inat21
    (image_id, cls_name, cls_num, evobio10m_id)
VALUES
    (?, ?, ?, ?);
"""


inat_filename_pattern = re.compile(r"(.*?)\.jpg")


def parse_inat_filename(filename):
    match = inat_filename_pattern.match(filename)
    if not match:
        raise ValueError(filename)
    return match.group(1), "jpg"


def copy_inat_from_clsdir(sink, clsdir):
    # each process get its own db connection.
    db = get_db()

    logger = logging.getLogger(f"p{os.getpid()}")

    insert_values = []
    for i, filename in enumerate(os.listdir(os.path.join(inat_root_dir, clsdir))):
        taxon = naming.dataset_class_to_taxon(clsdir)
        image_id, ext = parse_inat_filename(filename)
        global_id = get_global_id()

        filepath = os.path.join(inat_root_dir, clsdir, filename)

        img = Image.open(filepath).resize(resize_size)
        txt = taxon.scientific_name
        insert_values.append(
            (image_id, taxon.taxonomic_name, taxon.dataset_id, global_id)
        )

        sink.write({"__key__": global_id, "jpg": img, "txt": txt})

        if i % db_write_frequency == 0:
            try:
                db.executemany(inat_insert_stmt, insert_values)
                db.commit()
                # If we throw an err on executemany, then we don't reset
                # insert_values so we can try again next round.
                insert_values = []
            except sqlite3.OperationalError as err:
                logger.warning(
                    "Error inserting. [len: %d, err: %s]", len(insert_values), err
                )

    # flush any leftover values
    db.executemany(inat_insert_stmt, insert_values)
    db.commit()
    db.close()


#########
# BIOSCAN
#########

bioscan_insert_stmt = """
INSERT INTO bioscan
    (part, image_id, evobio10m_id)
VALUES
    (?, ?, ?);
"""

bioscan_txt_lookup = {}


def init_bioscan_metadata():
    with open(bioscan_metadata_path) as fd:
        bioscan_metadata = json.load(fd)

    for row in tqdm(bioscan_metadata, desc="Loading Bioscan metadata"):
        taxon = (
            "Animalia",
            "Arthropoda",
            "Insecta",
            row["order"],
            row["family"],
            row["genus"],
            row["species"],
        )

        # scientific name
        txt = " ".join(label for label in taxon[-2:] if label != "not_classified")
        bioscan_txt_lookup[row["image_file"]] = txt


def copy_bioscan_from_part(sink, part):
    # each process get its own db connection.
    db = get_db()

    logger = logging.getLogger(f"p{os.getpid()}")

    logger = logging.getLogger(f"p{os.getpid()}")
    if not bioscan_txt_lookup:
        logger.error("bioscan_txt_lookup is empty!")
    assert bioscan_txt_lookup, "bioscan_txt_lookup is empty"

    insert_values = []
    partdir = os.path.join(bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        global_id = get_global_id()
        insert_values.append((part, filename, global_id))

        if filename not in bioscan_txt_lookup:
            logger.warning(
                "Cannot find taxon. Skipping. [part: %s, filename: %s]", part, filename
            )
            continue

        txt = bioscan_txt_lookup[filename]

        filepath = os.path.join(partdir, filename)
        img = Image.open(filepath).resize(resize_size)

        sink.write({"__key__": global_id, "jpg": img, "txt": txt})

        if i % db_write_frequency == 0:
            try:
                db.executemany(bioscan_insert_stmt, insert_values)
                db.commit()
                # If we throw an err on executemany, then we don't reset
                # insert_values so we can try again next round.
                insert_values = []
            except sqlite3.OperationalError as err:
                logger.warning(
                    "Error inserting. [len: %d, err: %s]", len(insert_values), err
                )

    # flush any leftover values
    db.executemany(bioscan_insert_stmt, insert_values)
    db.commit()
    db.close()


######
# MAIN
######

sentinel = "STOP"


def worker(input):
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    with wds.ShardWriter(shard_pattern, shard_counter, maxsize=3e9) as sink:
        for func, args in iter(input.get, sentinel):
            func(sink, *args)


if __name__ == "__main__":
    eol_name_lookup = EolNameLookup()
    init_bioscan_metadata()

    # Creates a shared integer
    shard_counter = multiprocessing.Value("I", 0, lock=True)

    task_queue = multiprocessing.Queue()

    # Submit all tasks
    # EOL
    for imgset_name in sorted(os.listdir(eol_root_dir)):
        assert imgset_name.endswith(".tar.gz")
        imgset_path = os.path.join(eol_root_dir, imgset_name)
        task_queue.put((copy_eol_from_tar, (imgset_path,)))

    # Bioscan
    # 113 parts in bioscan
    for i in range(1, 114):
        task_queue.put((copy_bioscan_from_part, (i,)))

    # iNat
    for clsdir in os.listdir(inat_root_dir):
        task_queue.put((copy_inat_from_clsdir, (clsdir,)))

    processes = []
    # Start worker processes
    for i in range(max_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        processes.append(p)
        p.start()

    # Stop worker processes
    for i in range(max_workers):
        task_queue.put(sentinel)

    for p in processes:
        p.join()
