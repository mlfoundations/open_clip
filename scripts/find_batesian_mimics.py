"""
Finds all descriptions that mention the word "batesian" (as in batesian mimicry).
Then write those descriptions with their page ids into an HTML file so that it's easy to read and manually find the other species that is being mimicked.
"""

import collections
import csv
import json
import re

from tqdm import tqdm

species_text_filepath = "data/eol/all_species_texts.jsonl"
media_manifest_filepath = "data/eol/media_manifest.csv"
output_dir_path = "data/eol/descriptions"

# Get image counts for each page_id
img_counts = collections.defaultdict(int)

with open(media_manifest_filepath) as fd:
    reader = csv.reader(fd)
    next(reader)
    for content_id, page_id, *_ in tqdm(reader):
        img_counts[page_id] += 1

# Get HTML descriptions matching "batesian"
descs = {}
pattern = re.compile(r"\b([Bb]atesian)\b")

with open(species_text_filepath) as fd:
    for line in tqdm(fd):
        data = json.loads(line)
        page_id = data["page_id"]
        desc = data["text"]["description"]
        if not pattern.search(desc.lower()):
            continue

        # Don't add it if we already have a longer description
        if page_id not in descs or len(descs[page_id]) < desc:
            descs[page_id] = desc

# Write HTML pages for each page_id
for page_id in sorted(descs):
    with open(f"{output_dir_path}/{page_id}.html", "w") as fd:
        desc = descs[page_id]
        n_img = img_counts[page_id]
        fd.write(f"<h1>Page Id: {page_id}</h1>\n\n")
        fd.write(f"<h2>Images: {n_img}</h2>\n\n")
        fd.write(desc)

