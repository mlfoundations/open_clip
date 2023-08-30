"""
Helpers for parsing CUB2011 files.
"""

import dataclasses
import pathlib

import numpy as np


class CubLabel:
    @classmethod
    def from_line(cls, line):
        values = line.split()
        args = []
        for val, (key, type_) in zip(values, cls.__dict__["__annotations__"].items()):
            if type_ == bool:
                args.append(True if val == "1" else False)
            else:
                args.append(type_(val))

        return cls(*args)


@dataclasses.dataclass(frozen=True)
class ImgLabel(CubLabel):
    img_id: int
    img_name: str


@dataclasses.dataclass(frozen=True)
class TrainTestSplitLabel(CubLabel):
    img_id: int
    is_train: bool


@dataclasses.dataclass(frozen=True)
class ImgAttrLabel(CubLabel):
    img_id: int
    attr_id: int
    present: bool
    certainty: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class AttrLabel(CubLabel):
    attr_id: int
    label: str
    category: str
    value: str

    @classmethod
    def from_line(cls, line):
        attr_id, label = line.split()
        category, value = label.split("::")

        return cls(int(attr_id), label, category, value)


@dataclasses.dataclass(frozen=True)
class CubBuiltinLabels:
    imgs: list[ImgLabel]
    splits: list[TrainTestSplitLabel]
    attrs: list[AttrLabel]
    img_attrs: list[ImgAttrLabel]

    @classmethod
    def from_root(cls, root):
        root = pathlib.Path(root)

        with open(root / "images.txt") as fd:
            img_labels = [ImgLabel.from_line(line) for line in fd]

        with open(root / "train_test_split.txt") as fd:
            split_labels = [TrainTestSplitLabel.from_line(line) for line in fd]

        with open(root / "attributes" / "attributes.txt") as fd:
            attr_labels = [AttrLabel.from_line(line) for line in fd]

        with open(root / "attributes" / "image_attribute_labels.txt") as fd:
            # Slow: 3.7M lines
            img_attr_labels = [ImgAttrLabel.from_line(line) for line in fd]

        return cls(img_labels, split_labels, attr_labels, img_attr_labels)


macrocategories = [
    "bill_shape",
    "bill_length",
    "tail_shape",
    "head_pattern",
    "wing_shape",
    "size",
    "shape",
    "color",
    "pattern",
]


def get_macrocategory(attr_label):
    for i, macro in enumerate(macrocategories):
        if macro in attr_label.category:
            return macro

    raise ValueError(attr_label)


@dataclasses.dataclass(frozen=True)
class CubAttributeLabels:
    builtins: CubBuiltinLabels
    binary_labels: np.array
    train_mask: np.array
    test_mask: np.array

    categories: list[str]
    category_mask: np.array

    macrocategories: list[str]
    macrocategory_mask: np.array

    @classmethod
    def from_root(cls, root: str):
        # Initialize cub labels.
        cub_labels = CubBuiltinLabels.from_root(root)

        # Make binary attribute labels.
        binary_labels = np.zeros(
            (len(cub_labels.imgs), len(cub_labels.attrs)), dtype=int
        )
        for img_attr_label in cub_labels.img_attrs:
            # Skip labels with "guessing"
            if img_attr_label.certainty == 2:
                continue

            # Skip attributes that are not present, they're already 0
            if not img_attr_label.present:
                continue

            # 1 indicates "not visible" so we should have skipped it as it's not present
            assert img_attr_label.certainty != 1

            # Both img_ids and attr_ids are 1-indexed, so substract 1.
            binary_labels[img_attr_label.img_id - 1, img_attr_label.attr_id - 1] = 1

        assert binary_labels.sum() == 343741, "Binary attribute labels are wrong!"

        # Make split masks.
        train = [label.img_id for label in cub_labels.splits if label.is_train]
        train_mask = np.zeros((len(cub_labels.imgs),), dtype=bool)
        train_mask[np.array(train) - 1] = True

        test = [label.img_id for label in cub_labels.splits if not label.is_train]
        test_mask = np.zeros((len(cub_labels.imgs),), dtype=bool)
        test_mask[np.array(test) - 1] = True

        # There should be no overlap between splits
        assert np.logical_xor(train_mask, test_mask).all()

        # Make category masks
        categories = list(set(label.category for label in cub_labels.attrs))
        category_mask = np.zeros((len(categories), len(cub_labels.attrs)), dtype=bool)
        for i, category in enumerate(categories):
            for attr_label in cub_labels.attrs:
                if attr_label.category == category:
                    category_mask[i, attr_label.attr_id - 1] = True

        # Make macrocategory masks
        macro_mask = np.zeros((len(macrocategories), len(cub_labels.attrs)), dtype=bool)
        for i, macro in enumerate(macrocategories):
            for attr_label in cub_labels.attrs:
                if get_macrocategory(attr_label) == macro:
                    macro_mask[i, attr_label.attr_id - 1] = True

        return cls(
            cub_labels,
            binary_labels,
            train_mask,
            test_mask,
            categories,
            category_mask,
            macrocategories,
            macro_mask,
        )
