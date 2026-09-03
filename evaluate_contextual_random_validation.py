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


VALIDATION_DIR = Path(
    r"C:\Projects\stanford40_split\validation"
)

RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

RANDOM_FILE = (
    RESULTS_ROOT
    / "random_crops"
    / "validation_random_crop_similarities.npz"
)

PROMPT_SELECTION_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "selected_contextual_prompt_strategy.json"
)

OUTPUT_DIR = RESULTS_ROOT / "context_prompts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "validation_contextual_random_same_crop.npz"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "selected_contextual_random_strategy.json"
)


def calculate_metrics(labels, similarities):
    predictions = similarities.argmax(axis=1)

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(labels, predictions),
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

    # Same fallback as original random-crop experiment.
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
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    return features


def main():
    print()
    print(
        "CONTEXTUAL P3 + FROZEN RANDOM-CROP VALIDATION"
    )
    print("=" * 90)

    # ------------------------------------------------------------
    # Verify frozen P3 prompt selection
    # ------------------------------------------------------------

    prompt_selection = json.loads(
        PROMPT_SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        prompt_selection["selection_split"]
        != "validation"
    ):
        raise ValueError(
            "Contextual prompt strategy was not "
            "selected on validation."
        )

    if (
        prompt_selection["selected_strategy"]
        != "p3"
    ):
        raise ValueError(
            "Expected frozen P3 prompt strategy."
        )

    # ------------------------------------------------------------
    # Load frozen random validation experiment
    # ------------------------------------------------------------

    with np.load(
        RANDOM_FILE,
        allow_pickle=True,
    ) as data:

        labels = data["labels"].astype(
            np.int64
        )

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
            for x in data["crop_scale"]
        )

        crop_ratio = tuple(
            float(x)
            for x in data["crop_ratio"]
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

    print(f"Images:          {len(labels)}")
    print(f"Classes:         {len(class_names)}")
    print(f"Model:           {model_name}")
    print(f"Pretrained:      {pretrained}")
    print(f"Random crops:    {num_crops}")
    print(f"Crop scale:      {crop_scale}")
    print(f"Crop ratio:      {crop_ratio}")
    print(f"Random seed:     {random_seed}")
    print(
        "Crop selector:   frozen P0 best-confidence"
    )
    print(
        "New text view:   frozen P3 ensemble"
    )

    if len(labels) != 1521:
        raise ValueError(
            f"Expected 1521 validation images, "
            f"found {len(labels)}."
        )

    # ------------------------------------------------------------
    # Dataset integrity
    # ------------------------------------------------------------

    dataset = ImageFolder(
        VALIDATION_DIR
    )

    if not np.array_equal(
        np.asarray(dataset.classes),
        class_names,
    ):
        raise ValueError(
            "Class ordering differs from "
            "frozen random-crop experiment."
        )

    fresh_labels = np.asarray(
        [
            label
            for _, label in dataset.samples
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        fresh_labels,
        labels,
    ):
        raise ValueError(
            "Validation labels differ from "
            "frozen random experiment."
        )

    fresh_filenames = np.asarray([
        str(
            Path(path).relative_to(
                VALIDATION_DIR
            )
        ).replace("\\", "/")
        for path, _ in dataset.samples
    ])

    if not np.array_equal(
        fresh_filenames,
        filenames,
    ):
        raise ValueError(
            "Validation file order differs "
            "from frozen random experiment."
        )

    validate_prompt_bank(
        dataset.classes
    )

    # ------------------------------------------------------------
    # OpenCLIP
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device:          {device}")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
    )

    tokenizer = open_clip.get_tokenizer(
        model_name
    )

    model.eval()

    # ------------------------------------------------------------
    # Frozen P3 text embedding
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
        "Encoding frozen P3 text representation..."
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

    text_p3 = text_p3 / text_p3.norm(
        dim=-1,
        keepdim=True,
    )

    # ------------------------------------------------------------
    # Reconstruct SAME crop chosen by frozen P0
    # ------------------------------------------------------------

    p3_rows = []

    print()
    print(
        "Reconstructing the frozen selected crop "
        "for each validation image..."
    )

    with torch.inference_mode():
        for image_index, (
            image_path,
            _,
        ) in enumerate(dataset.samples):

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

            crop_tensor = preprocess(
                crop
            ).unsqueeze(0).to(
                device
            )

            image_feature = (
                model.encode_image(
                    crop_tensor
                )
            )

            image_feature = (
                image_feature
                / image_feature.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            similarities = (
                image_feature
                @ text_p3.T
            )

            p3_rows.append(
                similarities[
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
    # Compare P0 and P3 on SAME crops
    # ------------------------------------------------------------

    p0_result = calculate_metrics(
        labels,
        p0_scores,
    )

    p3_result = calculate_metrics(
        labels,
        p3_scores,
    )

    print()
    print(
        "VALIDATION: SAME P0-SELECTED RANDOM CROPS"
    )
    print("=" * 90)

    print(
        f"{'Text representation':<28}"
        f"{'Accuracy':>14}"
        f"{'Macro-F1':>14}"
        f"{'Weighted-F1':>16}"
    )

    print("-" * 90)

    print(
        f"{'P0 Basic':<28}"
        f"{p0_result['accuracy'] * 100:>13.2f}%"
        f"{p0_result['macro_f1'] * 100:>13.2f}%"
        f"{p0_result['weighted_f1'] * 100:>15.2f}%"
    )

    print(
        f"{'P3 Ensemble':<28}"
        f"{p3_result['accuracy'] * 100:>13.2f}%"
        f"{p3_result['macro_f1'] * 100:>13.2f}%"
        f"{p3_result['weighted_f1'] * 100:>15.2f}%"
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

    print()
    print(
        "P3 DELTAS VS FROZEN RANDOM P0"
    )
    print("-" * 90)

    print(
        f"Accuracy:    {accuracy_delta:+.2f} pp"
    )
    print(
        f"Macro-F1:    {macro_delta:+.2f} pp"
    )
    print(
        f"Weighted-F1: {weighted_delta:+.2f} pp"
    )

    # ------------------------------------------------------------
    # Validation-only selection
    # ------------------------------------------------------------

    p0_key = (
        p0_result["macro_f1"],
        p0_result["accuracy"],
        p0_result["weighted_f1"],
    )

    p3_key = (
        p3_result["macro_f1"],
        p3_result["accuracy"],
        p3_result["weighted_f1"],
    )

    if p3_key > p0_key:
        selected = "p3"
        selected_result = p3_result
    else:
        selected = "p0"
        selected_result = p0_result

    print()
    print("=" * 90)
    print(
        "SELECTED RANDOM-CROP TEXT STRATEGY: "
        f"{selected.upper()}"
    )

    print(
        "Selection rule: "
        "Macro-F1 -> Accuracy -> Weighted-F1"
    )

    # ------------------------------------------------------------
    # Save validation result
    # ------------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        labels=labels,
        filenames=filenames,
        class_names=class_names,
        selected_crop_indices=(
            selected_indices
        ),
        p0_similarities=p0_scores,
        p3_similarities=p3_scores,
        p0_predictions=(
            p0_result["predictions"]
        ),
        p3_predictions=(
            p3_result["predictions"]
        ),
        selected_strategy=np.asarray(
            selected
        ),
        selected_accuracy=np.asarray(
            selected_result[
                "accuracy"
            ]
        ),
        selected_macro_f1=np.asarray(
            selected_result[
                "macro_f1"
            ]
        ),
        selected_weighted_f1=np.asarray(
            selected_result[
                "weighted_f1"
            ]
        ),
        crop_selector=np.asarray(
            "frozen_p0_best_confidence"
        ),
    )

    metadata = {
        "selection_split": "validation",
        "crop_selector": (
            "frozen_p0_best_confidence"
        ),
        "selection_rule": (
            "macro_f1_then_accuracy_then_weighted_f1"
        ),
        "selected_strategy": selected,
        "p0_accuracy": p0_result[
            "accuracy"
        ],
        "p0_macro_f1": p0_result[
            "macro_f1"
        ],
        "p0_weighted_f1": p0_result[
            "weighted_f1"
        ],
        "p3_accuracy": p3_result[
            "accuracy"
        ],
        "p3_macro_f1": p3_result[
            "macro_f1"
        ],
        "p3_weighted_f1": p3_result[
            "weighted_f1"
        ],
        "accuracy_delta_pp": (
            accuracy_delta
        ),
        "macro_f1_delta_pp": (
            macro_delta
        ),
        "weighted_f1_delta_pp": (
            weighted_delta
        ),
        "num_images": len(labels),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"NPZ saved to:  {OUTPUT_NPZ}"
    )
    print(
        f"JSON saved to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
