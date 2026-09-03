from pathlib import Path
import json
import math
import random

import numpy as np
import open_clip
import torch

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder

from contextual_prompt_bank import (
    get_prompts,
    validate_prompt_bank,
)


TEST_DIR = Path(
    r"C:\Projects\stanford40_split\test"
)

RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

RANDOM_FILE = (
    RESULTS_ROOT
    / "random_crops"
    / "test_random_crop_similarities.npz"
)

SELECTION_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "selected_contextual_random_strategy.json"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "context_prompts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "test_contextual_random_same_crop.npz"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "test_contextual_random_same_crop_report.txt"
)


def calculate_metrics(labels, similarities):
    predictions = similarities.argmax(axis=1)

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def deterministic_random_crop(
    image,
    image_index,
    crop_index,
    num_crops,
    crop_scale,
    crop_ratio,
    random_seed,
):
    width, height = image.size
    area = height * width

    log_ratio = (
        math.log(crop_ratio[0]),
        math.log(crop_ratio[1]),
    )

    rng = random.Random(
        random_seed
        + image_index * num_crops
        + crop_index
    )

    for _ in range(10):
        target_area = area * rng.uniform(
            crop_scale[0],
            crop_scale[1],
        )

        aspect_ratio = math.exp(
            rng.uniform(
                log_ratio[0],
                log_ratio[1],
            )
        )

        crop_width = round(
            math.sqrt(
                target_area * aspect_ratio
            )
        )

        crop_height = round(
            math.sqrt(
                target_area / aspect_ratio
            )
        )

        if (
            0 < crop_width <= width
            and 0 < crop_height <= height
        ):
            left = rng.randint(
                0,
                width - crop_width,
            )

            top = rng.randint(
                0,
                height - crop_height,
            )

            return image.crop(
                (
                    left,
                    top,
                    left + crop_width,
                    top + crop_height,
                )
            )

    input_ratio = width / height

    if input_ratio < crop_ratio[0]:
        crop_width = width
        crop_height = round(
            crop_width / crop_ratio[0]
        )

    elif input_ratio > crop_ratio[1]:
        crop_height = height
        crop_width = round(
            crop_height * crop_ratio[1]
        )

    else:
        crop_width = width
        crop_height = height

    left = max(
        0,
        (width - crop_width) // 2,
    )

    top = max(
        0,
        (height - crop_height) // 2,
    )

    return image.crop(
        (
            left,
            top,
            left + crop_width,
            top + crop_height,
        )
    )


def encode_text(
    model,
    tokenizer,
    prompts,
    device,
):
    tokens = tokenizer(
        prompts
    ).to(device)

    with torch.inference_mode():
        features = model.encode_text(
            tokens
        )

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            )
        )

    return features


def main():
    print()
    print(
        "FROZEN P3 + RANDOM-CROP TEST"
    )
    print("=" * 90)

    # ------------------------------------------------------------
    # Validation-selected strategy check
    # ------------------------------------------------------------

    selection = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        selection["selection_split"]
        != "validation"
    ):
        raise ValueError(
            "Random-crop text strategy "
            "was not selected on validation."
        )

    if (
        selection["selected_strategy"]
        != "p3"
    ):
        raise ValueError(
            "Expected frozen P3 strategy."
        )

    if (
        selection["crop_selector"]
        != "frozen_p0_best_confidence"
    ):
        raise ValueError(
            "Unexpected crop-selection strategy."
        )

    # ------------------------------------------------------------
    # Frozen original random-crop test
    # ------------------------------------------------------------

    with np.load(
        RANDOM_FILE,
        allow_pickle=True,
    ) as data:

        labels = data[
            "labels"
        ].astype(np.int64)

        filenames = data[
            "filenames"
        ].astype(str)

        class_names = data[
            "class_names"
        ].astype(str)

        selected_indices = data[
            "best_confidence_indices"
        ].astype(np.int64)

        p0_scores = data[
            "best_confidence_similarities"
        ].astype(np.float32)

        num_crops = int(
            data["num_crops"]
        )

        crop_scale = tuple(
            float(x)
            for x in data[
                "crop_scale"
            ]
        )

        crop_ratio = tuple(
            float(x)
            for x in data[
                "crop_ratio"
            ]
        )

        random_seed = int(
            data["random_seed"]
        )

        model_name = str(
            data["model_name"]
        )

        pretrained = str(
            data["pretrained"]
        )

    if len(labels) != 1921:
        raise ValueError(
            f"Expected 1921 test images, "
            f"got {len(labels)}."
        )

    print(
        f"Images:          {len(labels)}"
    )
    print(
        f"Classes:         {len(class_names)}"
    )
    print(
        f"Model:           {model_name}"
    )
    print(
        f"Pretrained:      {pretrained}"
    )
    print(
        f"Random crops:    {num_crops}"
    )
    print(
        f"Crop scale:      {crop_scale}"
    )
    print(
        f"Random seed:     {random_seed}"
    )
    print(
        "Crop selector:   frozen P0 best-confidence"
    )
    print(
        "Text strategy:   frozen P3 ensemble"
    )

    # ------------------------------------------------------------
    # Dataset integrity
    # ------------------------------------------------------------

    dataset = ImageFolder(
        TEST_DIR
    )

    if not np.array_equal(
        np.asarray(
            dataset.classes
        ),
        class_names,
    ):
        raise ValueError(
            "Test class ordering differs "
            "from frozen random experiment."
        )

    fresh_labels = np.asarray(
        [
            label
            for _, label
            in dataset.samples
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        fresh_labels,
        labels,
    ):
        raise ValueError(
            "Test labels differ from "
            "frozen random experiment."
        )

    fresh_filenames = np.asarray([
        str(
            Path(path).relative_to(
                TEST_DIR
            )
        ).replace("\\", "/")
        for path, _
        in dataset.samples
    ])

    if not np.array_equal(
        fresh_filenames,
        filenames,
    ):
        raise ValueError(
            "Test filenames/order differ "
            "from frozen random experiment."
        )

    validate_prompt_bank(
        dataset.classes
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:          {device}"
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
    )

    tokenizer = (
        open_clip.get_tokenizer(
            model_name
        )
    )

    model.eval()

    # ------------------------------------------------------------
    # Frozen P3 representation
    # ------------------------------------------------------------

    prompts_p0 = get_prompts(
        dataset.classes,
        "p0",
    )

    prompts_p1 = get_prompts(
        dataset.classes,
        "p1",
    )

    prompts_p2 = get_prompts(
        dataset.classes,
        "p2",
    )

    print()
    print(
        "Encoding frozen P3 text ensemble..."
    )

    text_p0 = encode_text(
        model,
        tokenizer,
        prompts_p0,
        device,
    )

    text_p1 = encode_text(
        model,
        tokenizer,
        prompts_p1,
        device,
    )

    text_p2 = encode_text(
        model,
        tokenizer,
        prompts_p2,
        device,
    )

    text_p3 = (
        text_p0
        + text_p1
        + text_p2
    ) / 3.0

    text_p3 = (
        text_p3
        / text_p3.norm(
            dim=-1,
            keepdim=True,
        )
    )

    # ------------------------------------------------------------
    # Reconstruct SAME frozen crop per test image
    # ------------------------------------------------------------

    p3_rows = []

    print()
    print(
        "Reconstructing frozen selected "
        "test crops..."
    )

    with torch.inference_mode():
        for image_index, (
            image_path,
            _,
        ) in enumerate(
            dataset.samples
        ):

            crop_index = int(
                selected_indices[
                    image_index
                ]
            )

            with Image.open(
                image_path
            ) as loaded:

                image = loaded.convert(
                    "RGB"
                )

                crop = deterministic_random_crop(
                    image=image,
                    image_index=image_index,
                    crop_index=crop_index,
                    num_crops=num_crops,
                    crop_scale=crop_scale,
                    crop_ratio=crop_ratio,
                    random_seed=random_seed,
                )

            tensor = preprocess(
                crop
            ).unsqueeze(0).to(
                device
            )

            image_feature = (
                model.encode_image(
                    tensor
                )
            )

            image_feature = (
                image_feature
                / image_feature.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            scores = (
                image_feature
                @ text_p3.T
            )

            p3_rows.append(
                scores[
                    0
                ]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            processed = (
                image_index + 1
            )

            if (
                processed % 100 == 0
                or processed
                == len(dataset)
            ):
                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                )

    p3_scores = np.stack(
        p3_rows,
        axis=0,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    p0_result = calculate_metrics(
        labels,
        p0_scores,
    )

    p3_result = calculate_metrics(
        labels,
        p3_scores,
    )

    # Integrity check against frozen random result
    if (
        abs(
            p0_result["accuracy"]
            - float(
                np.load(
                    RANDOM_FILE,
                    allow_pickle=True,
                )["accuracy"]
            )
        )
        > 1e-8
    ):
        raise ValueError(
            "Frozen P0 random accuracy "
            "integrity check failed."
        )

    accuracy_delta = (
        p3_result["accuracy"]
        - p0_result["accuracy"]
    ) * 100

    macro_delta = (
        p3_result["macro_f1"]
        - p0_result["macro_f1"]
    ) * 100

    weighted_delta = (
        p3_result["weighted_f1"]
        - p0_result["weighted_f1"]
    ) * 100

    lines = [
        "",
        "FROZEN P3 + RANDOM-CROP TEST RESULT",
        "=" * 90,
        f"Images: {len(labels)}",
        f"Classes: {len(class_names)}",
        (
            "Crop selection: frozen P0 "
            "best-confidence crop"
        ),
        (
            "Text strategy: frozen "
            "P3 embedding ensemble"
        ),
        "",
        "TEST PERFORMANCE",
        "-" * 90,
        (
            f"{'Text representation':<28}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
        (
            f"{'P0 Basic':<28}"
            f"{p0_result['accuracy'] * 100:>13.2f}%"
            f"{p0_result['macro_f1'] * 100:>13.2f}%"
            f"{p0_result['weighted_f1'] * 100:>15.2f}%"
        ),
        (
            f"{'P3 Ensemble':<28}"
            f"{p3_result['accuracy'] * 100:>13.2f}%"
            f"{p3_result['macro_f1'] * 100:>13.2f}%"
            f"{p3_result['weighted_f1'] * 100:>15.2f}%"
        ),
        "",
        "TEST DELTAS",
        "-" * 90,
        (
            f"Accuracy:    "
            f"{accuracy_delta:+.2f} pp"
        ),
        (
            f"Macro-F1:    "
            f"{macro_delta:+.2f} pp"
        ),
        (
            f"Weighted-F1: "
            f"{weighted_delta:+.2f} pp"
        ),
    ]

    report = "\n".join(
        lines
    )

    print(
        report
    )

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels,
        filenames=filenames,
        class_names=class_names,

        selected_crop_indices=(
            selected_indices
        ),

        p0_similarities=(
            p0_scores
        ),

        p3_similarities=(
            p3_scores
        ),

        p0_predictions=(
            p0_result[
                "predictions"
            ]
        ),

        p3_predictions=(
            p3_result[
                "predictions"
            ]
        ),

        p0_accuracy=np.asarray(
            p0_result[
                "accuracy"
            ]
        ),

        p3_accuracy=np.asarray(
            p3_result[
                "accuracy"
            ]
        ),

        p0_macro_f1=np.asarray(
            p0_result[
                "macro_f1"
            ]
        ),

        p3_macro_f1=np.asarray(
            p3_result[
                "macro_f1"
            ]
        ),

        p0_weighted_f1=np.asarray(
            p0_result[
                "weighted_f1"
            ]
        ),

        p3_weighted_f1=np.asarray(
            p3_result[
                "weighted_f1"
            ]
        ),

        crop_selector=np.asarray(
            "frozen_p0_best_confidence"
        ),

        text_strategy=np.asarray(
            "p3"
        ),
    )

    print()
    print(
        f"NPZ saved to:    {OUTPUT_NPZ}"
    )

    print(
        f"Report saved to: {OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
