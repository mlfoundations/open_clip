# Dataset Card for Mimicry Classification

**(Note: this is an internal-only dataset card for now, simply describing how I made the dataset.)**

## To Do

* Create a webdataset format for evaluation.
* Provide example images.
* Write a script to evaluate CLIP models.

## Dataset Creation

Given ~700K descriptions in `all_species_texts.jsonl`, I filtered those down to 196 descriptions by searching for the word "batesian".
**Note: I used a regular expression with word breaks: `\bbatesian\b`.**
Then I deduplicated page ids (which corresponds to a species, genus, family, etc. in EOL) and kept the longer descriptions.
That led to 146 unique page ids with the word "batesian" in at least one description.
For each description, I read the description, looking for batesian, to try to find mimic-model species pairs.
Given a species name or a common name, I would look up the corresponding page id using EOL's `vernacularnames.csv`.
I wrote down the mimic and model page ids in `mimics.csv` along with any notes.

This led to 72 mimic-model pairs.
However, not every species has multiple images.
So I filtered these 72 pairs such that every species had at least `{20,30,40,50}` images.
This reduced the datset to `{53,42,32,22}` pairs and only `{1060,1260,1280,1100}` images.
See the table below.

| $n$ | Images | Pairs |
|-----|--------|-------|
| 20 | 1060 | 53 |
| 30 | 1260 | 42 |
| 40 | 1280 | 32 |
| 50 | 1100 | 22 |


### Reproducing

The scripts I used to filter were `scripts/batesian_mimics.py`.
This script needs paths to the `all_species_texts.jsonl` file, the `media_manifest.csv` file and an output directory for the HTML pages (for easy reading).
After running the script, run `python -m http.server` in the output directory and open up `http://localhost:8000` to start browsing.

I simply used CTRL-F in `vernacularnames.csv`, only choosing rows that had the language code `eng`.

## Source Data

All the data is sourced from EOL.

