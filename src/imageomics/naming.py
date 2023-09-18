import csv
import dataclasses
import json
import os
import re

from tqdm import tqdm


@dataclasses.dataclass
class Taxon:
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str

    def __post_init__(self):
        assert isinstance(self.kingdom, str)
        assert isinstance(self.phylum, str)
        assert isinstance(self.cls, str)
        assert isinstance(self.order, str)
        assert isinstance(self.family, str)
        assert isinstance(self.genus, str)
        assert isinstance(self.species, str)

    @property
    def taxonomic(self):
        name = " ".join(
            [
                self.kingdom.capitalize(),
                self.phylum.capitalize(),
                self.cls.capitalize(),
                self.order.capitalize(),
                self.family.capitalize(),
                self.genus.capitalize(),
                self.species.lower(),
            ]
        ).strip()

        # Remove double whitespaces
        return " ".join(name.split())

    @property
    def scientific(self):
        name = " ".join(
            [
                self.genus.capitalize(),
                self.species.lower(),
            ]
        ).strip()

        # Remove double whitespaces
        return " ".join(name.split())

    @property
    def tagged(self):
        return [
            (key, value)
            for key, value in [
                ("kingdom", self.kingdom.capitalize()),
                ("phylum", self.phylum.capitalize()),
                ("class", self.cls.capitalize()),
                ("order", self.order.capitalize()),
                ("family", self.family.capitalize()),
                ("genus", self.genus.capitalize()),
                ("species", self.species.lower()),
            ]
            if value
        ]


class NameLookup:
    """Lookup from a key to scientific name, taxonomic name and key name.
    Meant to be subclassed for each different data source.
    """

    def scientific(self, key: str):  # str | None
        raise NotImplementedError()

    def taxonomic(self, key: str):  # str | None
        raise NotImplementedError()

    def common(self, key: str):  # str | None
        raise NotImplementedError()

    def tagged(self):  # list[tuple[str, str]]
        raise NotImplementedError()

    def keys(self):  # list[str]
        raise NotImplementedError()


class BioscanNameLookup(NameLookup):
    def __init__(
        self,
        bioscan_metadata_path="/fs/scratch/PAS2136/bioscan/BIOSCAN_Insect_Dataset_metadata.jsonld",
    ):
        self.lookup = {}

        with open(bioscan_metadata_path) as fd:
            bioscan_metadata = json.load(fd)

        for row in tqdm(bioscan_metadata, desc="Loading Bioscan metadata"):
            taxon = Taxon(
                "Animalia",
                "Arthropoda",
                "Insecta",
                row["order"] if row["order"] != "not_classified" else "",
                row["family"] if row["family"] != "not_classified" else "",
                row["genus"] if row["genus"] != "not_classified" else "",
                row["species"] if row["species"] != "not_classified" else "",
            )
            self.lookup[row["image_file"]] = taxon

    def scientific(self, key):
        if key not in self.lookup:
            return None

        return self.lookup[key].scientific

    def taxonomic(self, key):
        if key not in self.lookup:
            return None

        return self.lookup[key].taxonomic

    def tagged(self, key):
        if key not in self.lookup:
            return None

        return self.lookup[key].tagged

    def common(self, key):
        return None

    def keys(self):
        return list(self.lookup.keys())


class iNaturalistNameLookup(NameLookup):
    def __init__(
        self,
        taxa_csv="data/inat/dwca/taxa.csv",
        vernacularnames_english_csv="data/inat/dwca/VernacularNames-english.csv",
    ):
        self._scientific = {}
        self._taxonomic = {}
        self._common = {}

        with open(taxa_csv, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["taxonRank"] != "species":
                    continue

                king = row["kingdom"]
                phyl = row["phylum"]
                cls = row["class"]
                ord = row["order"]
                fam = row["family"]
                genus = row["genus"]
                species = row["specificEpithet"]

                scientific = f"{genus.capitalize()} {species.lower()}"
                self._scientific[int(row["id"])] = scientific
                self._taxonomic[int(row["id"])] = " ".join(
                    [
                        king.capitalize(),
                        phyl.capitalize(),
                        cls.capitalize(),
                        ord.capitalize(),
                        fam.capitalize(),
                        genus.capitalize(),
                        species.lower(),
                    ]
                )

        with open(vernacularnames_english_csv, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self._common[int(row["id"])] = row["vernacularName"]

    def scientific(self, key):
        if key not in self._scientific:
            return None

        return self._scientific[key]

    def common(self, key):
        if key not in self._common:
            return None

        return self._common[key]

    def taxonomic(self, key):
        if key not in self._taxonomic:
            return None

        return self._taxonomic[key]

    def keys(self):  # list[int]
        return list(set(self._scientific) | set(self._taxonomic) | set(self._common))


class iNat21NameLookup(NameLookup):
    def __init__(self, inat21_root="/fs/ess/PAS2136/foundation_model/inat21/raw/train"):
        self._taxa = {}

        for clsdir in os.listdir(inat21_root):
            index, king, phyl, cls, ord, fam, genus, species = clsdir.split("_")
            index = int(index, base=10)
            taxon = Taxon(king, phyl, cls, ord, fam, genus, species)
            self._taxa[clsdir] = taxon

    def scientific(self, key):
        if key not in self._taxa:
            return None

        return self._taxa[key].scientific

    def taxonomic(self, key):
        if key not in self._taxa:
            return None

        return self._taxa[key].taxonomic

    def tagged(self, key):
        if key not in self._taxa:
            return None

        return self._taxa[key].tagged

    def common(self, key):
        return None

    def keys(self):  # list[str]
        return list(self._taxa.keys())


def tagged_to_scientific(tagged):
    genus, species = "", ""
    for tier, value in tagged:
        if tier == "genus":
            genus = value
        if tier == "species":
            species = value

    if not genus and not species:
        # Just use whatever taxonomic information we have.
        return " ".join(value.capitalize() for tier, value in tagged)

    return f"{genus.capitalize()} {species.lower()}".strip()


def taxonomic_to_scientific(taxonomic):
    tiers = taxonomic.split()

    # Assume that if there are at least 7 "words" in the taxonomic name, then the last
    # two are likely to be genus and species. But if there are fewer than 7, it's more
    # likely that it's missing species, genus, family, etc. in that order.
    if len(tiers) < 7:
        return ""

    *_, genus, species = tiers
    return f"{genus.capitalize()} {species.lower()}".strip()


def strip_html(string):
    """Just removes <i>...</i> tags and similar. Not complete."""
    return re.sub(r"<.*?>(.*?)</.*?>", r"\1", string)
