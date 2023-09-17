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
import argparse
import json
import logging
import multiprocessing
import os
import re
import sqlite3
import tarfile
import uuid

from PIL import Image, ImageFile

from imageomics import eol, naming, wds

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True

max_workers = 64

eol_root_dir = "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive"
inat_root_dir = "/fs/ess/PAS2136/foundation_model/inat21/raw/train"
bioscan_root_dir = "/fs/scratch/PAS2136/bioscan/cropped_256"


seen_in_training_json = "data/rarespecies/seen_in_training.json"
unseen_in_training_json = "data/rarespecies/unseen_in_training.json"

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


########
# SHARED
########


def load_img(file):
    img = Image.open(file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img.resize(resize_size, resample=Image.BICUBIC)


# A dictionary of scientific name (str) to taxonomic name (str)
class TaxonomicNameLookup(dict):
    def add(self, lookup: naming.NameLookup):
        """Adds all taxonomic names from an naming.NameLookup instance."""

        for key in lookup.keys():
            scientific = lookup.scientific(key)
            taxonomic = lookup.taxonomic(key)

            if scientific in self:
                continue

            if scientific is not None and taxonomic is not None:
                self[scientific] = taxonomic


# A dictionary of scientific name (str) to common name (str)
class CommonNameLookup(dict):
    def add(self, lookup: naming.NameLookup):
        """Adds all common names from an naming.NameLookup instance."""

        for key in lookup.keys():
            scientific = lookup.scientific(key)
            common = lookup.common(key)

            if scientific in self:
                continue

            if scientific is not None and common is not None:
                self[scientific] = common


class ImageFilter:
    """Filters images based on a black list."""

    def __init__(self, blacklist):
        self.blacklist = set(blacklist)

    def passes(self, file):
        return file not in self.blacklist


class SpeciesFilter:
    """Filters species based on a"""

    pass


######################
# Encyclopedia of Life
######################


eol_insert_stmt = """
INSERT INTO eol
    (content_id, page_id, evobio10m_id)
VALUES
    (?, ?, ?);
"""


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
            if eol_img.raw in image_blacklist:
                continue

            scientific = eol_name_lookup.scientific(eol_img.page_id)
            if scientific is None:
                continue

            if scientific in species_blacklist:
                continue

            global_id = get_global_id()

            insert_values.append((eol_img.content_id, eol_img.page_id, global_id))

            file = tar.extractfile(member)
            try:
                img = load_img(file).resize(resize_size)
            except OSError as err:
                logger.warning(
                    "Error opening file. Skipping. [tar: %s, err: %s]", imgset_path, err
                )
                continue

            taxonomic = eol_name_lookup.taxonomic(eol_img.page_id)
            tagged = eol_name_lookup.tagged(eol_img.page_id)
            txt_dct = make_txt(
                scientific=scientific, taxonomic=taxonomic, tagged=tagged
            )

            sink.write({"__key__": global_id, "jpg": img, **txt_dct})

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
        filepath = os.path.join(inat_root_dir, clsdir, filename)
        img = load_img(filepath).resize(resize_size)

        image_id, ext = parse_inat_filename(filename)
        global_id = get_global_id()

        scientific = inat21_name_lookup.scientific(clsdir)
        if scientific in species_blacklist:
            continue

        taxonomic = inat21_name_lookup.taxonomic(clsdir)
        tagged = inat21_name_lookup.tagged(clsdir)
        txt_dct = make_txt(scientific=scientific, taxonomic=taxonomic, tagged=tagged)

        sink.write({"__key__": global_id, "jpg": img, **txt_dct})

        index, *_ = clsdir.split("_")
        index = int(index)
        insert_values.append((image_id, taxonomic, index, global_id))

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


def copy_bioscan_from_part(sink, part):
    # each process get its own db connection.
    db = get_db()

    logger = logging.getLogger(f"p{os.getpid()}")

    logger = logging.getLogger(f"p{os.getpid()}")

    insert_values = []
    partdir = os.path.join(bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        global_id = get_global_id()
        insert_values.append((part, filename, global_id))

        tagged = bioscan_name_lookup.tagged(filename)
        if tagged is None:
            logger.warning(
                "Cannot find taxon. Skipping. [part: %s, filename: %s]", part, filename
            )
            continue

        if bioscan_name_lookup.scientific(filename) in species_blacklist:
            continue

        filepath = os.path.join(partdir, filename)
        img = load_img(filepath).resize(resize_size)

        sink.write({"__key__": global_id, "jpg": img, **make_txt(tagged=tagged)})

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


def make_txt(*, scientific=None, taxonomic=None, common=None, tagged=None):
    """
    From tagged, we can construct the scientific and the taxonomic names.
    Common names have to come from scientific names.
    """

    # 1. Make sure scientific name is not empty
    # 1.1. Try using tagged.
    if not scientific and tagged:
        scientific = naming.tagged_to_scientific(tagged)

    # 1.2. Try using taxonomic
    if not scientific and taxonomic:
        scientific = naming.taxonomic_to_scientific(taxonomic)

    # 1.3. Give up.
    if not scientific:
        scientific = ""

    # 2. Make sure taxonomic name is not empty.
    # 2.1. Try using the taxonomic lookup.
    if not taxonomic and scientific and scientific in taxonomic_name_lookup:
        taxonomic = taxonomic_name_lookup[scientific]
        tiers = ("kingdom", "phylum", "class", "order", "genus", "species")

        # If we use the taxonomic_name_lookup, set up tagged
        if not tagged:
            tagged = [(tag, value) for tag, value in zip(taxonomic.split(), tiers)]

    # 2.2. Try using the tagged name
    if not taxonomic and tagged:
        # Don't both doing capitalize/lower because this likely came from naming.Taxon
        taxonomic = " ".join(value for _, value in tagged)

    # 2.3 Give up
    if not taxonomic:
        taxonomic = scientific

    # 3. Make sure tagged is not empty.
    if not tagged:
        values = scientific.split()
        if len(values) == 1:
            tagged = [("genus", scientific)]
        elif len(values) == 2:
            genus, species = values
            tagged = [("genus", genus.capitalize()), ("species", species.lower())]
        else:
            genus, *species = values  # sometimes there is a space in a species name
            species = " ".join(species)
            tagged = [("genus", genus.capitalize()), ("species", species.lower())]

    # 4. Make sure common is not empty
    # 4.1. Try the common name lookup
    if not common and scientific in common_name_lookup:
        common = common_name_lookup[scientific]

    # 4.2. Give up.
    if not common:
        common = scientific

    taxonomic_tag = " ".join(f"{tier} {value}" for tier, value in tagged)

    sci = f"a photo of {scientific}."
    taxon = f"a photo of {taxonomic}."
    taxon_tag = f"a photo of {taxonomic_tag}."

    com = f"a photo of {common}."
    sci_com = f"a photo of {scientific} with common name {common}."
    taxon_com = f"a photo of {taxonomic} with common name {common}."
    taxon_tag_com = f"a photo of {taxonomic_tag} with common name {common}."

    return {
        # Names
        "scientific_name.txt": scientific,
        "taxonomic_name.txt": taxonomic,
        "common_name.txt": common,
        # "A photo of"... captions
        "sci.txt": sci,
        "com.txt": com,
        "taxon.txt": taxon,
        "taxonTag.txt": taxon_tag,
        "sci_com.txt": sci_com,
        "taxon_com.txt": taxon_com,
        "taxonTag_com.txt": taxon_tag_com,
    }


sentinel = "STOP"


def worker(input):
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    with wds.ShardWriter(shard_pattern, shard_counter, maxsize=3e9) as sink:
        for func, args in iter(input.get, sentinel):
            func(sink, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width", type=int, default=224, help="Width of resized images."
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Height of resized images."
    )

    parser.add_argument("--tag", default="dev", help="The suffix for the directory.")

    args = parser.parse_args()

    # Set up some global variables that depend on CLI args.
    resize_size = (args.width, args.height)
    output_dir = f"/fs/ess/PAS2136/open_clip/data/evobio10m-{args.tag}/{args.width}x{args.height}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing images to {output_dir}.")

    db_path = os.path.join(output_dir, "mapping.sqlite")

    taxonomic_name_lookup = TaxonomicNameLookup()
    common_name_lookup = CommonNameLookup()

    # Add Bioscan
    bioscan_name_lookup = naming.BioscanNameLookup()
    taxonomic_name_lookup.add(bioscan_name_lookup)
    common_name_lookup.add(bioscan_name_lookup)

    # Add EOL
    eol_name_lookup = eol.EolNameLookup()
    taxonomic_name_lookup.add(eol_name_lookup)
    common_name_lookup.add(eol_name_lookup)

    # Add iNaturalist
    inaturalist_name_lookup = naming.iNaturalistNameLookup()
    taxonomic_name_lookup.add(inaturalist_name_lookup)
    common_name_lookup.add(inaturalist_name_lookup)

    # Add iNat21
    inat21_name_lookup = naming.iNat21NameLookup()
    taxonomic_name_lookup.add(inat21_name_lookup)
    common_name_lookup.add(inat21_name_lookup)

    image_blacklist = set()
    species_blacklist = set()

    with open(seen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)

    with open(unseen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)
            species_blacklist.add(scientific)

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
