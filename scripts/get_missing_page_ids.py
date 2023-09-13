"""
This script finds page ids in the media_cargo_archive/*.tar that are missing the scientific name.
We look in these files for the scientific name:

* data/eol/dh21/taxon.tab
* data/eol/trait_bank/pages.csv
* data/eol/provider_ids.csv
* data/eol/vernacularnames.csv
* data/eol/missing_page_ids.csv

It then queries the eol.org HTTP API to get the name and writes them to data/eol/missing_page_ids.csv.
"""

import collections
import csv
import logging
import os
import re
import time

import requests
from tqdm import tqdm

from imageomics import eol

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("eol-scraper")

media_cargo_archive_map_csv = (
    "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive_map.csv"
)
taxon_tab = "data/eol/dh21/taxon.tab"
provider_ids_csv = "data/eol/provider_ids.csv"
pages_csv = "data/eol/trait_bank/pages.csv"
vernacularnames_csv = "data/eol/vernacularnames.csv"
scraped_page_ids_csv = "data/eol/scraped_page_ids.csv"

# Page Ids that EOL is missing
missing_page_ids_csv = "data/eol/missing_page_ids.csv"


def get_scientific_name_from_eol(page_id, *, max_retries=1):
    """
    Fetch and process text data
    """

    for retry in range(max_retries):
        try:
            url = f"https://eol.org/api/pages/1.0/{page_id}.json"
            params = {"taxonomy": True, "vetted": 0}

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                return data.get("taxonConcept", {}).get(
                    "scientificName", "not provided"
                )

            # 500 error, retry up to a limit in case it's a transient issue.
            if response.status_code == 500:
                # Only retry if we haven't exhausted our retries
                if retry < max_retries - 1:
                    logger.info(
                        "HTTP %d error for %s, retrying in %d seconds.",
                        response.status_code,
                        url,
                        2**retry,
                    )
                    time.sleep(2**retry)
            else:
                logger.info(
                    "HTTP %d for page_id %d. Quitting.", response.status_code, page_id
                )
                return None  # Exit if status code isn't 200 or 500

        except requests.exceptions.HTTPError:
            if retry < max_retries - 1:
                logger.info("Retrying in %d seconds.", 2**retry)
                time.sleep(2**retry)
        except requests.exceptions.ConnectTimeout:
            if retry < max_retries - 1:
                logger.info("Timeout for %s. Retrying in %d seconds.", url, 2**retry)
                time.sleep(2**retry)

    return None


def clean_html(string):
    re.sub(r"<.*?>(.*?)</.*?>", r"\1", string)


if __name__ == "__main__":
    if not os.path.isfile(missing_page_ids_csv):
        with open(missing_page_ids_csv, "w") as fd:
            fd.write("page_id\n")

    if not os.path.isfile(scraped_page_ids_csv):
        with open(scraped_page_ids_csv, "w") as fd:
            fd.write("page_id,scientific_name\n")

    media_page_id_lists = collections.defaultdict(list)

    with open(media_cargo_archive_map_csv) as fd:
        reader = csv.reader(fd)
        next(reader)  # headers
        for img, _ in tqdm(reader, desc="media_cargo_archive_map.csv"):
            eol_filename = eol.ImageFilename.from_filename(img)
            media_page_id_lists[eol_filename.page_id].append(eol_filename)

    assert sum(len(imgs) for imgs in media_page_id_lists.values()) > 6.7e6

    missing = set(media_page_id_lists)
    page_id_to_scientific = {}

    with open(taxon_tab) as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        for row in tqdm(reader, desc="taxon.tab"):
            if not row["eolID"] or not row["scientificName"]:
                continue
            page_id = int(row["eolID"])
            if page_id not in page_id_to_scientific:
                page_id_to_scientific[page_id] = row["scientificName"]

    missing -= set(page_id_to_scientific)
    print(
        f"After reading taxon.tab, missing {len(missing)} page ids' scientific names."
    )

    # provider_ids.csv
    with open(provider_ids_csv) as fd:
        reader = csv.DictReader(fd)
        for row in tqdm(reader, desc="provider_ids.csv"):
            if not row["page_id"] or not row["preferred_canonical_for_page"]:
                continue
            page_id = int(row["page_id"])
            if page_id not in page_id_to_scientific:
                page_id_to_scientific[page_id] = row["preferred_canonical_for_page"]

    missing -= set(page_id_to_scientific)
    print(
        f"After reading provider_ids.csv, missing {len(missing)} page ids' scientific names."
    )

    # trait_bank/pages.csv
    with open(pages_csv) as fd:
        reader = csv.DictReader(fd)
        for row in tqdm(reader, desc="trait_bank/pages.csv"):
            if not row["page_id"] or not row["canonical"]:
                continue
            page_id = int(row["page_id"])
            if page_id not in page_id_to_scientific:
                page_id_to_scientific[page_id] = row["canonical"]

    missing -= set(page_id_to_scientific)
    print(
        f"After reading trait_bank/pages.csv, missing {len(missing)} page ids' scientific names."
    )

    # vernacularnames.csv
    with open(vernacularnames_csv) as fd:
        reader = csv.DictReader(fd)
        for row in tqdm(reader, desc="vernacularnames.csv"):
            page_id = int(row["page_id"])

            if page_id not in page_id_to_scientific:
                page_id_to_scientific[page_id] = clean_html(row["canonical_form"])

    missing -= set(page_id_to_scientific)
    print(
        f"After reading vernacularnames.csv, missing {len(missing)} page ids' scientific names."
    )

    # scraped_page_ids.csv is the file we write to when scraping EOL's website
    with open(scraped_page_ids_csv) as fd:
        reader = csv.DictReader(fd)
        for row in tqdm(reader, desc="scraped_page_ids.csv"):
            page_id = int(row["page_id"])

            if page_id not in page_id_to_scientific:
                page_id_to_scientific[page_id] = row["scientific_name"]

    missing -= set(page_id_to_scientific)
    print(
        f"After reading scraped_page_ids.csv, missing {len(missing)} page ids' scientific names."
    )

    reqs_per_sec = 1  # just a guess
    print(f"Querying eol.org/api/pages/1.0/ for {len(missing)} page ids.")
    print(
        f"At {reqs_per_sec} requests/second, this will take {len(missing) / reqs_per_sec / 60 / 60:.1f} hours."
    )

    with open(missing_page_ids_csv) as fd:
        actually_missing = set(int(i) for i in fd.read().split()[1:])

    for page_id in tqdm(missing):
        if page_id in actually_missing:
            continue

        scientific_name = get_scientific_name_from_eol(page_id)
        if scientific_name is None:
            # logger.info("Couldn't find scientific name for page_id %d", page_id)
            with open(missing_page_ids_csv, "a") as fd:
                fd.write(f"{page_id}\n")
        else:
            with open(scraped_page_ids_csv, "a") as fd:
                fd.write(f"{page_id},{scientific_name}\n")
