import csv
import dataclasses
import re

from tqdm import tqdm

from . import naming

eol_filename_pattern = re.compile(r"(\d+)_(\d+)_eol.*jpg")


@dataclasses.dataclass(frozen=True)
class ImageFilename:
    """
    Represents a filename like 12784812_51655800_eol-full-size-copy.jpg
    """

    content_id: int
    page_id: int
    ext: str
    raw: str

    @classmethod
    def from_filename(cls, filename):
        match = eol_filename_pattern.match(filename)
        if not match:
            raise ValueError(filename)
        return cls(int(match.group(1)), int(match.group(2)), "jpg", filename)


@dataclasses.dataclass(frozen=True)
class VernacularName:
    common: str
    canonical: str
    preferred: bool


class VernacularNameLookup:
    def __init__(self, vernacularnames_csv="data/eol/vernacularnames.csv"):
        self.lookup = {}

        with open(vernacularnames_csv) as fd:
            reader = csv.reader(fd)
            next(reader)  # skip header row
            for row, (page_id, canonical, common, lang, _, _, preferred) in enumerate(
                tqdm(reader, desc=vernacularnames_csv)
            ):
                if lang != "eng":  # only keep english
                    continue

                page_id = int(page_id)
                preferred = bool(preferred)
                name = VernacularName(
                    naming.strip_html(common), naming.strip_html(canonical), preferred
                )

                if page_id not in self.lookup:
                    self.lookup[page_id] = name
                else:
                    if self.lookup[page_id].preferred:
                        continue  # already have a preferred name
                    else:
                        if preferred:
                            # update with preferrred name
                            self.lookup[page_id] = name
                        else:
                            continue  # use previous unpreferred name

    def __getitem__(self, page_id):
        if page_id not in self.lookup:
            return None
        return self.lookup[page_id]

    def __contains__(self, page_id):
        return page_id in self.lookup

    def __iter__(self):
        yield from iter(self.lookup)


def parse_all_hierarchies_in_graph(path):  # dict[str, naming.Taxon]
    """Parses the all_hierarchies_in_graph.csv file.

    Returns a dictionary from page_id to a Taxon"""
    dct = {}
    # scientific names
    with open(path) as fd:
        reader = csv.reader(fd)
        next(reader)  # skip header row
        for page_id, tier_chain, rank_chain, _ in tqdm(reader, desc=path):
            king = ""
            phyl = ""
            cls = ""
            ord = ""
            fam = ""
            genus = ""
            species = ""

            for tier, rank in zip(tier_chain.split("->"), rank_chain.split("->")):
                if rank == "kingdom" and not king:
                    king = tier
                if rank == "phylum" and not phyl:
                    phyl = tier

                # There are lots of different "class" levels in the chain, and some of
                # them are not between phyl and ord, so we need this extra check
                if rank == "class" and not cls and phyl and not ord:
                    cls = tier

                if rank == "order" and not ord:
                    ord = tier
                if rank == "family" and not fam:
                    fam = tier
                if rank == "genus" and not genus:
                    genus = tier
                if rank == "species" and not species:
                    species = tier

            taxon = naming.Taxon(king, phyl, cls, ord, fam, genus, species)
            page_id = int(page_id)

            if page_id not in dct:
                dct[page_id] = taxon

    return dct


class EolNameLookup(naming.NameLookup):
    def __init__(
        self,
        *,
        hierarchies_path="/fs/scratch/PAS2136/eol/data/interim/all_hierarchies_in_graph.csv",
        media_cargo_archive_map_csv="/fs/ess/PAS2136/eol/data/interim/media_cargo_archive_map.csv",
        pages_csv="data/eol/trait_bank/pages.csv",
        provider_ids_csv="data/eol/provider_ids.csv",
        scraped_page_ids_csv="data/eol/scraped_page_ids.csv",
        taxon_tab="data/eol/dh21/taxon.tab",
        vernacularnames_csv="data/eol/vernacularnames.csv",
    ):
        self._common = {}
        self._taxonomic = {}
        self._scientific = {}
        self._tagged = {}

        # scientific names
        with open(provider_ids_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc=provider_ids_csv):
                if not row["page_id"]:
                    continue
                page_id = int(row["page_id"])

                name = naming.strip_html(row["preferred_canonical_for_page"])
                if not name:
                    continue

                if page_id not in self._scientific:
                    self._scientific[page_id] = name

        # scientific names
        with open(taxon_tab) as fd:
            reader = csv.DictReader(fd, delimiter="\t")
            for row in tqdm(reader, desc=taxon_tab):
                if not row["eolID"]:
                    continue
                page_id = int(row["eolID"])

                name = naming.strip_html(row["scientificName"])
                if not name:
                    continue

                if page_id not in self._scientific:
                    self._scientific[page_id] = name

        # scientific names
        with open(pages_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc=pages_csv):
                if not row["page_id"]:
                    continue
                page_id = int(row["page_id"])

                name = naming.strip_html(row["canonical"])
                if not name:
                    continue

                if page_id not in self._scientific:
                    self._scientific[page_id] = name

        # scientific and common names
        vernacular_names = VernacularNameLookup(vernacularnames_csv)
        for page_id in vernacular_names:
            name = vernacular_names[page_id]
            if page_id not in self._scientific:
                self._scientific[page_id] = name.canonical

            if page_id not in self._common:
                self._common[page_id] = name.common

        # scientific and taxonomic names
        for page_id, taxon in parse_all_hierarchies_in_graph(hierarchies_path).items():
            if page_id not in self._taxonomic:
                # Sometimes the hierarchies file ignores the species and only has genus
                # If this is the case, try to find the species from the scientific name
                if not taxon.species:
                    scientific = self.scientific(page_id) or ""
                    split = scientific.split()  # genus, species
                    if len(split) == 2 and split[0].lower() == taxon.genus.lower():
                        taxon.species = split[1]

                self._taxonomic[page_id] = taxon.taxonomic

            if page_id not in self._scientific:
                self._scientific[page_id] = taxon.scientific
            if page_id not in self._tagged:
                self._tagged[page_id] = taxon.tagged

        # scientific names
        with open(scraped_page_ids_csv) as fd:
            reader = csv.DictReader(fd)
            for row in tqdm(reader, desc=scraped_page_ids_csv):
                page_id = int(row["page_id"])
                name = naming.strip_html(row["scientific_name"])

                if page_id not in self._scientific:
                    self._scientific[page_id] = name

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

    def tagged(self, key):
        if key not in self._tagged:
            return None

        return self._tagged[key]

    def keys(self):  # list[int]
        return list(
            set(self._scientific)
            | set(self._taxonomic)
            | set(self._common)
            | set(self._tagged)
        )
