"""
Converts CUB 2011 into two splits and resizes the images appropriately.

There are only 12K images, so this script only takes about a minute.
But if you wanted to do this for more pictures, it would need to be refactored.
"""

import os
import pathlib

from PIL import Image
from tqdm.auto import tqdm

import evaluation.data

cub_root = pathlib.Path("/local/scratch/cv_datasets/cub2011/original/CUB_200_2011")
resize_size = (224, 224)

output_root = pathlib.Path("/local/scratch/cv_datasets/cub2011/traintest224")

os.makedirs(output_root, exist_ok=True)


def parse_train_test(path):
    train, test = set(), set()
    with open(path) as fd:
        for line in fd:
            image_id, is_train = line.split()
            if is_train == "1":
                train.add(int(image_id))
            else:
                test.add(int(image_id))

    return {"train": train, "test": test}


if __name__ == "__main__":
    splits = evaluation.data.parse_train_test(cub_root / "train_test_split.txt")

    with open(cub_root / "images.txt") as fd:
        for line in tqdm(list(fd)):
            img_id, img_relpath = line.split()
            img_id = int(img_id)

            img_abspath = cub_root / "images" / img_relpath
            img = Image.open(img_abspath).resize(resize_size)

            img_outpath = None
            for split, ids in splits.items():
                if img_id in ids:
                    img_outpath = output_root / split / img_relpath
            assert img_outpath is not None

            os.makedirs(img_outpath.parent, exist_ok=True)
            img.save(img_outpath)
