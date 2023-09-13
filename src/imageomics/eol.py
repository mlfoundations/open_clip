import csv
import dataclasses
import re

from tqdm import tqdm

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
    vernacularnames_csv = "data/eol/vernacularnames.csv"

    def __init__(self, vernacularnames_csv=None):
        if vernacularnames_csv is not None:
            self.vernacularnames_csv = vernacularnames_csv

        self.lookup = {}

        with open(self.vernacularnames_csv) as fd:
            reader = csv.reader(fd)
            next(reader)  # skip header row
            for row, (page_id, canonical, common, lang, _, _, preferred) in enumerate(
                tqdm(reader, desc=self.vernacularnames_csv)
            ):
                if lang != "eng":  # only keep english
                    continue

                page_id = int(page_id)
                preferred = bool(preferred)
                name = VernacularName(common, canonical, preferred)

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
