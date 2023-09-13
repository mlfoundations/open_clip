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
import concurrent.futures
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

from imageomics import concurrency, naming, wds

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True

max_workers = 32

eol_root_dir = "/fs/scratch/PAS2136/eol/data/interim/media_cargo_archive"
inat_root_dir = "/fs/ess/PAS2136/foundation_model/inat21/raw/train"
bioscan_root_dir = "/fs/scratch/PAS2136/bioscan/cropped_256"
bioscan_metadata_path = (
    "/fs/ess/PAS2136/BIOSCAN/google_drive/BIOSCAN_Insect_Dataset_metadata.jsonld"
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


sink = None


def init_sink(shard_counter: multiprocessing.Value):
    global sink
    sink = wds.ShardWriter(
        os.path.join(output_dir, "shard-%06d.tar"),
        shard_counter,
        verbose=False,
        maxsize=3e9,  # 3 GB
    )


def close_sink():
    global sink
    sink.close()


######################
# Encyclopedia of Life
######################


eol_insert_stmt = """
INSERT INTO eol
    (content_id, page_id, evobio10m_id)
VALUES
    (?, ?, ?);
"""


hierarchies_path = "/fs/scratch/PAS2136/eol/data/interim/all_hierarchies_in_graph.csv"
eol_hierarchies_lookup = {}


def init_eol_hierarchies_lookup():
    with open(hierarchies_path) as fd:
        reader = csv.reader(fd)
        next(reader)  # skip header row
        for page_id, raw_canonical_chain, raw_rank_chain, raw_page_id_chain in tqdm(
            reader
        ):
            canonical_chain = raw_canonical_chain.split("->")
            rank_chain = raw_rank_chain.split("->")
            page_id_chain = [int(i) for i in raw_page_id_chain.split("->")]
            eol_hierarchies_lookup[int(page_id)] = canonical_chain


def get_taxonomic_name(eol_page_id):
    if eol_page_id not in eol_hierarchies_lookup:
        return None

    return " ".join(eol_hierarchies_lookup[eol_page_id])


eol_filename_pattern = re.compile(r"(\d+)_(\d+)_eol.*jpg")


def parse_eol_filename(filename):
    match = eol_filename_pattern.match(filename)
    if not match:
        raise ValueError(filename)
    return int(match.group(1)), int(match.group(2)), "jpg"


def copy_from_imgset(imgset_path):
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
            content_id, page_id, ext = parse_eol_filename(member.name)
            taxonomic_name = get_taxonomic_name(page_id)
            if not taxonomic_name:
                continue

            global_id = get_global_id()

            insert_values.append((content_id, page_id, global_id))

            file = tar.extractfile(member)
            try:
                img = Image.open(file).resize(resize_size)
            except OSError as err:
                logger.warning(
                    "Error opening file. Skipping. [tar: %s, err: %s]", imgset_path, err
                )
                continue

            sink.write({"__key__": global_id, "jpg": img, "txt": taxonomic_name})

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


def parse_inat_clsdir(clsdir):
    taxon = naming.dataset_class_to_taxon(clsdir)

    return taxon


def copy_from_inat_clsdir(clsdir):
    # each process get its own db connection.
    db = get_db()

    logger = logging.getLogger(f"p{os.getpid()}")

    insert_values = []
    for i, filename in enumerate(os.listdir(os.path.join(inat_root_dir, clsdir))):
        taxon = parse_inat_clsdir(clsdir)
        image_id, ext = parse_inat_filename(filename)
        global_id = get_global_id()

        filepath = os.path.join(inat_root_dir, clsdir, filename)

        img = Image.open(filepath).resize(resize_size)
        txt = taxon.scientific_name
        insert_values.append(
            (image_id, taxon.scientific_name, taxon.dataset_id, global_id)
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

        txt = " ".join(label for label in taxon if label != "not_classified")
        bioscan_txt_lookup[row["image_file"]] = txt


def copy_bioscan_from_part(part):
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


if __name__ == "__main__":
    init_eol_hierarchies_lookup()
    init_bioscan_metadata()

    try:
        shard_counter = multiprocessing.Value("I", 0, lock=True)

        pool = concurrency.BoundedExecutor(
            pool_cls=concurrent.futures.ProcessPoolExecutor,
            max_workers=max_workers,
            initializer=init_sink,
            initargs=(shard_counter,),
        )

        # EOL
        for imgset_name in sorted(os.listdir(eol_root_dir)):
            assert imgset_name.endswith(".tar.gz")
            imgset_path = os.path.join(eol_root_dir, imgset_name)
            pool.submit(copy_from_imgset, imgset_path)
        pool.finish(desc="Copying EOL images")

        # Bioscan
        # 113 parts in bioscan
        for i in range(1, 114):
            pool.submit(copy_bioscan_from_part, i)
        pool.finish(desc="Copying BioScan images")

        # iNat
        for clsdir in os.listdir(inat_root_dir):
            pool.submit(copy_from_inat_clsdir, clsdir)
        pool.finish(desc="Copying iNat21 images")
        
        pool.submit(close_sink)
        pool.finish(desc="Writing all .tar files.")
    finally:
        pool.shutdown()
