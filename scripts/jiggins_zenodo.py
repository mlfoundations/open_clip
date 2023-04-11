import collections
import concurrent.futures
import csv
import json
import os
import random

import requests
from tqdm.auto import tqdm

master_file = "data/butterflies/jiggins_zenodo_master.csv"
desc_file = "data/butterflies/heliconius.json"
output_file = "data/butterflies/butterfly-images.json"
data_root = "/local/scratch/cv_datasets/butterfly/clip_eval"

good_projects = (
    "2677821",
    "2707828",
    "2682458",
    "2682669",
    "2684906",
    "2686762",
    "1748277",
    "3082688",
    "1247307",
    "2548678",
    "2702457",
    "2714333",
)

bad_projects = (
    "4291095",  # green backround,
    "5561246",  # not all four wings
    "3477891",  # no images
)

min_count = 5
max_count = 25
max_workers = 4


class BoundedExecutor:
    def __init__(self, pool_cls=concurrent.futures.ThreadPoolExecutor):
        self._pool = pool_cls(max_workers=max_workers)
        self._futures = []

    def submit(self, *args, **kwargs) -> None:
        self._futures.append(self._pool.submit(*args, **kwargs))

    def shutdown(self, **kwargs) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True, **kwargs)

    def finish(self, *, desc: str = "", progress=True):
        if progress:
            iterator = tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._futures),
                desc=desc,
            )
        else:
            iterator = concurrent.cutures.as_completed(self._futures)

        return [future.result() for future in iterator]


def download_img(link, out):
    # don't do anything if downloaded already
    if os.path.isfile(out):
        return

    # download
    r = requests.get(link)
    if r.status_code == 404:
        print(link, "404")
        return

    if r.status_code == 429:
        time.sleep(10)
        print("trying again:", link)
        download_img(link, out)
        return

    r.raise_for_status()

    # write to disk
    with open(out, "wb") as fd:
        fd.write(r.content)


def main():
    with open(desc_file) as fd:
        desc = json.load(fd)

    species = [sp.lower() for sp in desc["data"].keys()]
    images = collections.defaultdict(list)

    with open(master_file) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            name = row["Taxonomic.Name"].lower()
            if name == "na":
                continue

            if not name:
                continue

            if "ssp." in name:
                continue

            if row["Side"].lower() != "dorsal":
                continue

            for sp in species:
                if sp not in name:
                    continue

                link = row["zenodo_link"]
                if any(proj_id in link for proj_id in bad_projects):
                    print(link, sp)
                    continue

                if not any(proj_id in link for proj_id in good_projects):
                    breakpoint()

                image = row["Image_name"]
                if image.endswith("CR2"):
                    continue
                if "whitestandard" in image:
                    continue
                images[sp].append(f"{link}/files/{image}")

    expected = sorted(
        [
            ("charithonia", 9),
            ("cydno", 114),
            ("doris", 6),
            ("erato", 167),
            ("hecale", 31),
            ("ismenius", 17),
            ("sapho", 4),
            ("sara", 188),
        ]
    )

    actual = [(sp, len(i)) for sp, i in sorted(images.items())]
    assert actual == expected

    random.seed(1337)

    filtered = {}
    for sp, img_list in sorted(images.items()):
        if len(img_list) < min_count:
            continue
        shuffled = random.sample(img_list, k=len(img_list))
        shuffled = shuffled[:max_count]

        filtered[sp] = shuffled

    images = [
        {"species": sp, "images": sorted(imgs)} for sp, imgs in sorted(filtered.items())
    ]
    with open(output_file, "w") as fd:
        json.dump(images, fd, indent=4)

    # Download all the images to a regular dataset
    try:
        pool = BoundedExecutor()
        for dct in images:
            pool.submit(
                os.makedirs, os.path.join(data_root, dct["species"]), exist_ok=True
            )
        pool.finish(desc="Making directories")

        for dct in images:
            for i, img in enumerate(dct["images"]):
                pool.submit(
                    download_img,
                    img,
                    os.path.join(data_root, dct["species"], f"{i:04d}.jpg"),
                )
        pool.finish(desc="Downloading images")
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
