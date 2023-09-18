"""
Reads all the files and makes a master mapping to specific files in EvoBio-10M. All 
files in the 10M have a UUID as the key. This script 

1. Creates the initial mapping (without reading the file contents)
2. Chooses 5% (configurable) of each dataset as a validation split.
3. Writes all these details to a sqlite database.
"""
import argparse
import logging
import multiprocessing
import os
import sqlite3
import tarfile
import uuid
import random
import time

from imageomics import eol, evobio10m


db_write_frequency = 50_000

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def get_global_id():
    return str(uuid.uuid4())


def get_logger():
    return logging.getLogger(f"p{os.getpid()}")


def randsleep(max, logger):
    sleep = random.randrange(max)
    logger.info("Sleeping to avoid database lock error. [seconds: %d]", sleep)
    time.sleep(sleep)


######################
# Encyclopedia of Life
######################


eol_insert_stmt = """
INSERT INTO eol
    (content_id, page_id, evobio10m_id)
VALUES
    (?, ?, ?);
"""


def read_eol_from_tar(imgset_path):
    """
    Reads all filenames from an imgset (.tar.gz file), assigns a uuid, then inserts
    it into a sqlite database.
    """
    logger = get_logger()

    db = evobio10m.get_db(db_path)

    insert_values = []

    # r|gz indcates reading from a gzipped file, streaming only
    with tarfile.open(imgset_path, "r|gz") as tar:
        for i, member in enumerate(tar):
            eol_img = eol.ImageFilename.from_filename(member.name)

            global_id = get_global_id()

            insert_values.append((eol_img.content_id, eol_img.page_id, global_id))

            if i % db_write_frequency == 0:
                randsleep(10, logger)
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
    randsleep(100, logger)
    db.executemany(eol_insert_stmt, insert_values)
    db.commit()
    db.close()


#########
# BIOSCAN
#########

bioscan_insert_stmt = """
INSERT INTO bioscan
    (part, filename, evobio10m_id)
VALUES
    (?, ?, ?);
"""


def read_bioscan_from_part(part):
    # each process get its own db connection.
    db = evobio10m.get_db(db_path)

    logger = get_logger()

    insert_values = []
    partdir = os.path.join(evobio10m.bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        global_id = get_global_id()
        insert_values.append((part, filename, global_id))

        if i % db_write_frequency == 0:
            randsleep(10, logger)
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
    randsleep(100, logger)
    db.executemany(bioscan_insert_stmt, insert_values)
    db.commit()
    db.close()


########
# INAT21
########

inat21_insert_stmt = """
INSERT INTO inat21
    (filename, cls_name, cls_num, evobio10m_id)
VALUES
    (?, ?, ?, ?);
"""


def read_inat21_from_clsdir(clsdir):
    # each process get its own db connection.
    db = evobio10m.get_db(db_path)

    logger = get_logger()

    insert_values = []
    for i, filename in enumerate(
        os.listdir(os.path.join(evobio10m.inat21_root_dir, clsdir))
    ):
        global_id = get_global_id()

        index, *taxa = clsdir.split("_")
        index = int(index)
        taxon = "_".join(taxa)
        insert_values.append((filename, taxon, index, global_id))

        if i % db_write_frequency == 0:
            randsleep(10, logger)
            try:
                db.executemany(inat21_insert_stmt, insert_values)
                db.commit()
                # If we throw an err on executemany, then we don't reset
                # insert_values so we can try again next round.
                insert_values = []
            except sqlite3.OperationalError as err:
                logger.warning(
                    "Error inserting. [len: %d, err: %s]", len(insert_values), err
                )

    # flush any leftover values
    randsleep(100, logger)
    db.executemany(inat21_insert_stmt, insert_values)
    db.commit()
    db.close()


def worker(queue):
    logger = get_logger()
    for func, args in iter(queue.get, sentinel):
        logger.info(f"Started {func.__name__}({', '.join(args)})")
        func(*args)
        logger.info(f"Finished {func.__name__}({', '.join(args)})")


sentinel = "STOP"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="dev", help="The suffix for the directory.")
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of processes to use."
    )
    args = parser.parse_args()

    # Set up some global variables that depend on CLI args.
    output_dir = evobio10m.get_output_dir(args.tag)
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "mapping.sqlite")
    print(f"Writing to {db_path}.")

    # 1. Create an initial mapping of all images in a sqlite database.
    task_queue = multiprocessing.Queue()

    # Submit all tasks
    # EOL
    for imgset_name in sorted(os.listdir(evobio10m.eol_root_dir)):
        assert imgset_name.endswith(".tar.gz")
        imgset_path = os.path.join(evobio10m.eol_root_dir, imgset_name)
        task_queue.put((read_eol_from_tar, (imgset_path,)))

    # Bioscan
    # 113 parts in bioscan
    for i in range(1, 114):
        task_queue.put((read_bioscan_from_part, (i,)))

    # iNat
    for clsdir in os.listdir(evobio10m.inat21_root_dir):
        task_queue.put((read_inat21_from_clsdir, (clsdir,)))

    processes = []
    # Start worker processes
    for i in range(args.workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        processes.append(p)
        p.start()

    # Stop worker processes
    for i in range(args.workers):
        task_queue.put(sentinel)

    for p in processes:
        p.join()
