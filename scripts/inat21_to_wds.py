"""
Converts a regular iNat21 dataset to a CLIP dataset in webdataset format.
"""
import os

import torchvision
import webdataset
from tqdm.auto import tqdm

inat_root = "/local/scratch/cv_datasets/inat21/rand-species-split/pretrain"
split = "train"
resize_size = (224, 224)

output_root = "/local/scratch/cv_datasets/inat21/rand-species-split/pretrain-webdataset"

# Pattern for shard names
shard_pattern = "shard-%06d.tar"
max_size = 1e9

# Make the output directory
output_root = os.path.join(output_root, split, "shards")
os.makedirs(output_root, exist_ok=True)


def parse(raw):
    index, *tiers = raw.split("_")
    return tiers


if __name__ == "__main__":
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(inat_root, split),
        transform=torchvision.transforms.Resize(resize_size, antialias=True),
    )

    sink = webdataset.ShardWriter(
        os.path.join(output_root, shard_pattern), maxsize=max_size
    )

    for i, (image, cls) in enumerate(tqdm(dataset)):
        label = parse(dataset.classes[cls])
        desc = " ".join(label)
        sink.write(
            {"__key__": f"sample{i:08d}", "jpg": image, "txt": desc}
        )

    sink.close()
