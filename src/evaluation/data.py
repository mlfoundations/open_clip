import os
import random

from torchvision import datasets


def make_splits(directory) -> dict[str, list[str]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    assert len(classes) == 10_000, "iNat21 has 10K classes!"

    random.seed(1337)
    shuffled = random.sample(classes, k=len(classes))

    # Check that they're the same order on all machines
    err = "not reproducible"
    assert (
        shuffled[0]
        == "08737_Plantae_Tracheophyta_Magnoliopsida_Lamiales_Verbenaceae_Verbena_hastata"
    ), err
    assert (
        shuffled[8000]
        == "03011_Animalia_Chordata_Amphibia_Anura_Microhylidae_Kaloula_pulchra"
    ), err
    assert (
        shuffled[8999]
        == "02422_Animalia_Arthropoda_Insecta_Odonata_Gomphidae_Arigomphus_furcifer"
    ), err
    assert (
        shuffled[9000]
        == "01963_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Pyronia_bathseba"
    ), err
    assert (
        shuffled[9999]
        == "06333_Plantae_Tracheophyta_Liliopsida_Poales_Poaceae_Avena_fatua"
    ), err

    return {
        "pretraining": shuffled[0:9000],
        "seen": shuffled[8000:9000],
        "unseen": shuffled[9000:10000],
    }


class PretrainingInat(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are unseen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["pretraining"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class UnseenInat(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are unseen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["unseen"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SeenInat(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are seen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["seen"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
