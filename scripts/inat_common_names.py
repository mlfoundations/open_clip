"""
Gets a mapping of scientific names to common names for iNat21 dataset
"""

import csv
import os

import imageomics.naming

archive_path = "data/inaturalist-taxonomy.dwca"
taxa_path = os.path.join(archive_path, "taxa.csv")
english_path = os.path.join(archive_path, "VernacularNames-english.csv")
inat_root = "/local/scratch/cv_datasets/inat21/raw"


def row_dict_to_taxon(row):
    assert row["taxonRank"] == "species"
    return imageomics.naming.Taxon(
        row["kingdom"],
        row["phylum"],
        row["class"],
        row["order"],
        row["family"],
        row["genus"],
        row["specificEpithet"],
        website_id=int(row["id"]),
    )


def load_dataset_classes():
    directory = os.path.join(inat_root, "val")
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")
    assert len(classes) == 10_000, "iNat21 has 10K classes!"

    return classes


def main():
    website_id_lookup = {}
    # Load all
    with open(taxa_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["taxonRank"] != "species":
                continue
            taxon = row_dict_to_taxon(row)

            assert taxon.website_id not in website_id_lookup
            website_id_lookup[taxon.website_id] = taxon

    # Set common names
    with open(english_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            website_id = int(row["id"])

            if website_id not in website_id_lookup:
                print(f"missing id {website_id} in taxa.csv")
                continue

            common_name = row["vernacularName"]
            website_id_lookup[website_id].common_name = common_name

    # Lookup from scientific name to taxa
    scientific_name_lookup = {}
    for _, taxon in website_id_lookup.items():
        scientific_name_lookup[taxon.scientific_name] = taxon

    mapping = {}
    for c in load_dataset_classes():
        taxon = imageomics.naming.dataset_class_to_taxon(c)
        if taxon.scientific_name not in scientific_name_lookup:
            print(f"missing species {taxon.scientific_name} in taxa.csv")
            continue

        scientific_name_lookup[taxon.scientific_name].dataset_id = taxon.dataset_id
        taxon = scientific_name_lookup[taxon.scientific_name]
        if not taxon.common_name:
            print(f"missing common name for {taxon.scientific_name}")
            continue

        mapping[taxon.dataset_id] = taxon.common_name

    imageomics.naming.write_mapping(mapping)


if __name__ == "__main__":
    main()
