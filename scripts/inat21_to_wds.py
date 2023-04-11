"""
Converts iNat21 to a pretraining webdataset format.

* Only includes 9K training classes so there are 1K unseen classes for evaluation
* Uses both scientific and common names in the text descriptions
"""
import os

import torchvision
import webdataset
from tqdm.auto import tqdm

import evaluation.data
import imageomics.naming

inat_root = "/local/scratch/cv_datasets/inat21/raw"
split = "train"
resize_size = (224, 224)

output_root = "/local/scratch/cv_datasets/inat21/pretrain-webdataset"

# Pattern for shard names
shard_pattern = "shard-%06d.tar"
max_size = 1e9

# Make the output directory
output_root = os.path.join(output_root, split, "shards")
os.makedirs(output_root, exist_ok=True)


def parse(raw):
    index, *tiers = raw.split("_")
    return int(index, base=10), tiers


def make_descriptions(classname):
    index, tiers = parse(classname)
    scientific_name = " ".join(tiers)
    yield f"a photo of {scientific_name}."

    if index not in common_name_lookup:
        # Don't know the common name
        return

    common_name = common_name_lookup[index]
    yield f"a photo of {common_name}."


if __name__ == "__main__":
    # Set up common name lookup
    common_name_lookup = imageomics.naming.read_mapping()

    dataset = evaluation.data.PretrainingInat(
        os.path.join(inat_root, split),
        transform=torchvision.transforms.Resize(resize_size, antialias=True),
    )

    sink = webdataset.ShardWriter(
        os.path.join(output_root, shard_pattern), maxsize=max_size
    )

    for i, (image, cls) in enumerate(tqdm(dataset)):
        label = parse(dataset.classes[cls])
        for desc in make_descriptions(dataset.classes[cls]):
            sink.write({"__key__": f"sample{i:08d}", "jpg": image, "txt": desc})

    sink.close()
