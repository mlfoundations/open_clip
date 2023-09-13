"""
Helpers and utilities for manipulating scientific and common names of species.
"""
import dataclasses
import json

mapping_file = "data/inat/common-names-mapping.json"


def read_mapping():
    mapping = {}
    with open(mapping_file) as fd:
        for key, value in json.load(fd).items():
            mapping[int(key)] = value

    return mapping


def write_mapping(mapping):
    """
    Makes sure the mapping is a lookup from dataset id (integer under 10K) to common name
    """
    assert isinstance(mapping, dict)

    for key, value in mapping.items():
        assert isinstance(key, int)
        assert key < 10000
        assert isinstance(value, str)
        assert "_" not in value

    with open(mapping_file, "w") as fd:
        json.dump(mapping, fd, indent=4)


@dataclasses.dataclass
class Taxon:
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str
    # id from inaturalist.org (taxa.csv)
    website_id: int = -1
    # id from the inat21 dataset
    dataset_id: int = -1
    # common name (from VernacularNames-english.csv)
    common_name: str = ""

    @property
    def taxonomic_name(self):
        return " ".join(
            [
                self.kingdom.capitalize(),
                self.phylum.capitalize(),
                self.cls.capitalize(),
                self.order.capitalize(),
                self.family.capitalize(),
                self.genus.capitalize(),
                self.species.lower(),
            ]
        )

    @property
    def scientific_name(self):
        return " ".join([self.genus.capitalize(), self.species.lower()])


def dataset_class_to_taxon(cls):
    index, *tiers = cls.split("_")
    index = int(index, base=10)
    return Taxon(*tiers, dataset_id=index)
