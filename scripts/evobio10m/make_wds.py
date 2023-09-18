"""
Writes the training and validation data to webdataset format.
"""
import argparse
import json
import logging
import multiprocessing
import os
import re
import tarfile

from PIL import Image, ImageFile

from imageomics import eol, evobio10m, naming, wds

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True

max_workers = 64

seen_in_training_json = "data/rarespecies/seen_in_training.json"
unseen_in_training_json = "data/rarespecies/unseen_in_training.json"


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


######################
# Encyclopedia of Life
######################


def copy_eol_from_tar(sink, imgset_path):
    """
    Copies all the files out of an imgset (.tar.gz file), resizes the images, then
    writes it to a .tar file in the webdataset format.
    """
    db = evobio10m.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, content_id, page_id FROM eol;"
    evobio10m_id_lookup = {
        (content_id, page_id): evobio10m_id
        for evobio10m_id, content_id, page_id in db.execute(select_stmt).fetchall()
    }

    logger = logging.getLogger(f"p{os.getpid()}")

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

            if (eol_img.content_id, eol_img.page_id) not in evobio10m_id_lookup:
                logger.warning(
                    "EvoBio10m ID missing. [content: %d, page: %d]",
                    eol_img.content_id,
                    eol_img.page_id,
                )
                continue

            global_id = evobio10m_id_lookup[(eol_img.content_id, eol_img.page_id)]
            if global_id not in splits[args.split]:
                continue

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


########
# INAT21
########


inat21_filename_pattern = re.compile(r"(.*?)\.jpg")


def parse_inat21_filename(filename):
    match = inat21_filename_pattern.match(filename)
    if not match:
        raise ValueError(filename)
    return match.group(1), "jpg"


def copy_inat21_from_clsdir(sink, clsdir):
    # each process get its own db connection.
    db = evobio10m.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, filename, cls_num FROM inat21;"
    evobio10m_id_lookup = {
        (filename, cls_num): evobio10m_id
        for evobio10m_id, filename, cls_num in db.execute(select_stmt).fetchall()
    }

    logger = logging.getLogger(f"p{os.getpid()}")

    clsdir_path = os.path.join(evobio10m.inat21_root_dir, clsdir)
    for i, filename in enumerate(os.listdir(clsdir_path)):
        filepath = os.path.join(clsdir_path, filename)
        img = load_img(filepath).resize(resize_size)

        cls_num, *_ = clsdir.split("_")
        cls_num = int(cls_num)

        if (filename, cls_num) not in evobio10m_id_lookup:
            logger.warning(
                "Evobio10m ID missing. [image: %s, cls: %d]", filename, cls_num
            )
            continue

        global_id = evobio10m_id_lookup[(filename, cls_num)]
        if global_id not in splits[args.split]:
            continue

        scientific = inat21_name_lookup.scientific(clsdir)
        if scientific in species_blacklist:
            continue

        taxonomic = inat21_name_lookup.taxonomic(clsdir)
        tagged = inat21_name_lookup.tagged(clsdir)
        txt_dct = make_txt(scientific=scientific, taxonomic=taxonomic, tagged=tagged)

        sink.write({"__key__": global_id, "jpg": img, **txt_dct})


#########
# BIOSCAN
#########


def copy_bioscan_from_part(sink, part):
    # each process get its own db connection.
    db = evobio10m.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, part, filename FROM bioscan;"
    evobio10m_id_lookup = {
        (part, filename): evobio10m_id
        for evobio10m_id, part, filename in db.execute(select_stmt).fetchall()
    }

    logger = logging.getLogger(f"p{os.getpid()}")

    partdir = os.path.join(evobio10m.bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        if (part, filename) not in evobio10m_id_lookup:
            logger.warning(
                "EvoBio10m ID missing. [part: %d, filename: %s]", part, filename
            )
            continue

        global_id = evobio10m_id_lookup[(part, filename)]
        if global_id not in splits[args.split]:
            continue

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


######
# MAIN
######


def load_splits():
    db = evobio10m.get_db(db_path)
    split_stmt = "SELECT evobio10m_id FROM split WHERE is_train = (?)"
    splits = {
        "train": {row[0] for row in db.execute(split_stmt, (1,)).fetchall()},
        "val": {row[0] for row in db.execute(split_stmt, (0,)).fetchall()},
    }
    db.close()
    return splits


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

    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--tag", default="dev", help="The suffix for the directory.")

    args = parser.parse_args()

    # Set up some global variables that depend on CLI args.
    resize_size = (args.width, args.height)
    output_dir = f"{evobio10m.get_output_dir(args.tag)}/{args.width}x{args.height}/{args.split}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing images to {output_dir}.")

    db_path = f"{evobio10m.get_output_dir(args.tag)}/mapping.sqlite"

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

    # Load train/val splits
    splits = load_splits()

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
    for imgset_name in sorted(os.listdir(evobio10m.eol_root_dir)):
        assert imgset_name.endswith(".tar.gz")
        imgset_path = os.path.join(evobio10m.eol_root_dir, imgset_name)
        task_queue.put((copy_eol_from_tar, (imgset_path,)))

    # Bioscan
    # 113 parts in bioscan
    for i in range(1, 114):
        task_queue.put((copy_bioscan_from_part, (i,)))

    # iNat
    for clsdir in os.listdir(evobio10m.inat21_root_dir):
        task_queue.put((copy_inat21_from_clsdir, (clsdir,)))

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
