from pathlib import Path
import json

import numpy as np
import open_clip
import torch

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torchvision.datasets import ImageFolder

from contextual_prompt_bank import (
    get_prompts,
    validate_prompt_bank,
)


# ============================================================
# Configuration
# ============================================================

ACTOR_DIR = Path(
    r"C:\Projects\stanford40_views\validation_actor_20"
)

RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "upper_body"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FULL_P3_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_prompt_results.npz"
)

RANDOM_P3_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_random_same_crop.npz"
)

PROMPT_SELECTION_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "selected_contextual_prompt_strategy.json"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "validation_upper_body_p3_results.npz"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "validation_upper_body_p3_report.txt"
)


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 16
NUM_WORKERS = 0

EXPECTED_IMAGES = 1521
EXPECTED_CLASSES = 40

# ------------------------------------------------------------
# Frozen BEFORE validation.
#
# Existing actor images already include a 20% actor margin.
# Keep the full horizontal width and retain the top 65%.
# ------------------------------------------------------------

UPPER_BODY_FRACTION = 0.65


# ============================================================
# Dataset
# ============================================================

class ActorUpperBodyDataset(Dataset):
    """
    Return both:
      1. original actor-centred crop
      2. deterministic upper-body crop

    The upper-body view retains the full actor-crop width
    and the top 65% of its height.
    """

    def __init__(
        self,
        actor_dir,
        preprocess,
    ):
        base = ImageFolder(
            actor_dir
        )

        self.samples = base.samples
        self.classes = base.classes
        self.class_to_idx = (
            base.class_to_idx
        )

        self.actor_dir = actor_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        path, label = (
            self.samples[index]
        )

        with Image.open(
            path
        ) as loaded:
            image = loaded.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

            upper_height = max(
                1,
                round(
                    height
                    * UPPER_BODY_FRACTION
                ),
            )

            upper_body = image.crop(
                (
                    0,
                    0,
                    width,
                    upper_height,
                )
            )

            actor_tensor = (
                self.preprocess(
                    image
                )
            )

            upper_tensor = (
                self.preprocess(
                    upper_body
                )
            )

        relative_path = str(
            Path(path).relative_to(
                self.actor_dir
            )
        ).replace(
            "\\",
            "/",
        )

        return (
            actor_tensor,
            upper_tensor,
            label,
            relative_path,
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    similarities,
):
    predictions = (
        similarities.argmax(
            axis=1
        )
    )

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
        features = (
            model.encode_text(
                tokens
            )
        )

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            )
        )

    return features


def build_p3_text_features(
    model,
    tokenizer,
    class_names,
    device,
):
    """
    Recreate the already frozen P3:
    normalized mean of P0 + P1 + P2.
    """

    prompts_p0 = get_prompts(
        class_names,
        "p0",
    )

    prompts_p1 = get_prompts(
        class_names,
        "p1",
    )

    prompts_p2 = get_prompts(
        class_names,
        "p2",
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

    return text_p3


# ============================================================
# Main
# ============================================================

def main():
    print()
    print(
        "UPPER-BODY P3 VALIDATION EXPERIMENT"
    )
    print("=" * 92)

    print(
        f"Upper-body fraction: "
        f"{UPPER_BODY_FRACTION:.2f}"
    )

    print(
        "Definition: top 65% of "
        "existing actor_20 crop"
    )

    # --------------------------------------------------------
    # Verify contextual P3 was validation-selected
    # --------------------------------------------------------

    selection = json.loads(
        PROMPT_SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        selection[
            "selection_split"
        ]
        != "validation"
    ):
        raise ValueError(
            "Contextual prompts were "
            "not validation-selected."
        )

    if (
        selection[
            "selected_strategy"
        ]
        != "p3"
    ):
        raise ValueError(
            "Expected frozen P3 "
            "prompt representation."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Pretrained: {PRETRAINED}"
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
        )
    )

    tokenizer = (
        open_clip.get_tokenizer(
            MODEL_NAME
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ActorUpperBodyDataset(
        ACTOR_DIR,
        preprocess,
    )

    if (
        len(dataset)
        != EXPECTED_IMAGES
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_IMAGES} images, "
            f"found {len(dataset)}."
        )

    if (
        len(dataset.classes)
        != EXPECTED_CLASSES
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_CLASSES} classes, "
            f"found "
            f"{len(dataset.classes)}."
        )

    validate_prompt_bank(
        dataset.classes
    )

    print(
        f"Images:  {len(dataset)}"
    )

    print(
        f"Classes: "
        f"{len(dataset.classes)}"
    )

    # --------------------------------------------------------
    # Load frozen Full-P3 / Random-P3 for alignment
    # --------------------------------------------------------

    with np.load(
        FULL_P3_FILE,
        allow_pickle=True,
    ) as full_data:

        full_labels = (
            full_data[
                "labels"
            ].astype(
                np.int64
            )
        )

        full_classes = (
            full_data[
                "class_names"
            ].astype(str)
        )

        full_filenames = (
            full_data[
                "filenames"
            ].astype(str)
        )

        full_scores = (
            full_data[
                "p3_similarities"
            ].astype(
                np.float32
            )
        )

    with np.load(
        RANDOM_P3_FILE,
        allow_pickle=True,
    ) as random_data:

        random_labels = (
            random_data[
                "labels"
            ].astype(
                np.int64
            )
        )

        random_classes = (
            random_data[
                "class_names"
            ].astype(str)
        )

        random_filenames = (
            random_data[
                "filenames"
            ].astype(str)
        )

        random_scores = (
            random_data[
                "p3_similarities"
            ].astype(
                np.float32
            )
        )

    if not np.array_equal(
        full_labels,
        random_labels,
    ):
        raise ValueError(
            "Full-P3 and Random-P3 "
            "validation labels differ."
        )

    if not np.array_equal(
        full_classes,
        random_classes,
    ):
        raise ValueError(
            "Full-P3 and Random-P3 "
            "class order differs."
        )

    if not np.array_equal(
        full_filenames,
        random_filenames,
    ):
        raise ValueError(
            "Full-P3 and Random-P3 "
            "filename order differs."
        )

    if not np.array_equal(
        np.asarray(
            dataset.classes
        ),
        full_classes,
    ):
        raise ValueError(
            "Actor dataset class order "
            "differs from frozen results."
        )

    expected_filenames = np.asarray([
        str(
            Path(path).relative_to(
                ACTOR_DIR
            )
        ).replace(
            "\\",
            "/",
        )
        for path, _
        in dataset.samples
    ])

    if not np.array_equal(
        expected_filenames,
        full_filenames,
    ):
        raise ValueError(
            "Actor dataset filename "
            "order differs from "
            "frozen validation split."
        )

    actor_labels = np.asarray(
        [
            label
            for _, label
            in dataset.samples
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        actor_labels,
        full_labels,
    ):
        raise ValueError(
            "Actor dataset labels differ "
            "from frozen validation split."
        )

    # --------------------------------------------------------
    # Frozen P3 text features
    # --------------------------------------------------------

    print()
    print(
        "Encoding frozen P3 "
        "text representation..."
    )

    text_p3 = (
        build_p3_text_features(
            model,
            tokenizer,
            dataset.classes,
            device,
        )
    )

    # --------------------------------------------------------
    # Encode Actor + UpperBody together
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    actor_score_rows = []
    upper_score_rows = []
    labels_rows = []
    filenames_rows = []

    processed = 0

    print()
    print(
        "Encoding Actor-P3 and "
        "UpperBody-P3..."
    )

    with torch.inference_mode():

        for (
            actor_images,
            upper_images,
            labels,
            filenames,
        ) in loader:

            actor_images = (
                actor_images.to(
                    device,
                    non_blocking=True,
                )
            )

            upper_images = (
                upper_images.to(
                    device,
                    non_blocking=True,
                )
            )

            # Actor view
            actor_features = (
                model.encode_image(
                    actor_images
                )
            )

            actor_features = (
                actor_features
                / actor_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            actor_scores = (
                actor_features
                @ text_p3.T
            )

            # Upper-body view
            upper_features = (
                model.encode_image(
                    upper_images
                )
            )

            upper_features = (
                upper_features
                / upper_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            upper_scores = (
                upper_features
                @ text_p3.T
            )

            actor_score_rows.append(
                actor_scores
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            upper_score_rows.append(
                upper_scores
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            labels_rows.append(
                labels.numpy()
            )

            filenames_rows.extend(
                filenames
            )

            processed += len(
                labels
            )

            if (
                processed % 200
                == 0
                or processed
                == len(dataset)
            ):
                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(dataset)}"
                )

    actor_scores = np.concatenate(
        actor_score_rows,
        axis=0,
    )

    upper_scores = np.concatenate(
        upper_score_rows,
        axis=0,
    )

    fresh_labels = np.concatenate(
        labels_rows,
        axis=0,
    ).astype(
        np.int64
    )

    fresh_filenames = np.asarray(
        filenames_rows
    )

    if not np.array_equal(
        fresh_labels,
        full_labels,
    ):
        raise ValueError(
            "Encoded label order changed."
        )

    if not np.array_equal(
        fresh_filenames,
        full_filenames,
    ):
        raise ValueError(
            "Encoded filename order changed."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    full_result = (
        calculate_metrics(
            full_labels,
            full_scores,
        )
    )

    actor_result = (
        calculate_metrics(
            full_labels,
            actor_scores,
        )
    )

    upper_result = (
        calculate_metrics(
            full_labels,
            upper_scores,
        )
    )

    random_result = (
        calculate_metrics(
            full_labels,
            random_scores,
        )
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    methods = [
        (
            "Full-P3",
            full_result,
        ),
        (
            "Actor-P3",
            actor_result,
        ),
        (
            "UpperBody-P3",
            upper_result,
        ),
        (
            "Random-P3",
            random_result,
        ),
    ]

    lines = [
        "",
        (
            "UPPER-BODY P3 "
            "VALIDATION RESULT"
        ),
        "=" * 92,
        (
            f"Images: "
            f"{len(full_labels)}"
        ),
        (
            f"Classes: "
            f"{len(full_classes)}"
        ),
        (
            "Upper-body rule: "
            "top 65% of actor_20 crop"
        ),
        "",
        "VALIDATION PERFORMANCE",
        "-" * 92,
        (
            f"{'Method':<28}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
    ]

    for name, result in methods:
        lines.append(
            (
                f"{name:<28}"
                f"{result['accuracy'] * 100:>13.2f}%"
                f"{result['macro_f1'] * 100:>13.2f}%"
                f"{result['weighted_f1'] * 100:>15.2f}%"
            )
        )

    lines.extend([
        "",
        "UPPER-BODY DELTAS",
        "-" * 92,
        (
            "UpperBody vs Actor "
            "Accuracy: "
            f"{(upper_result['accuracy'] - actor_result['accuracy']) * 100:+.2f} pp"
        ),
        (
            "UpperBody vs Actor "
            "Macro-F1: "
            f"{(upper_result['macro_f1'] - actor_result['macro_f1']) * 100:+.2f} pp"
        ),
        (
            "UpperBody vs Full "
            "Macro-F1: "
            f"{(upper_result['macro_f1'] - full_result['macro_f1']) * 100:+.2f} pp"
        ),
        (
            "UpperBody vs Random "
            "Macro-F1: "
            f"{(upper_result['macro_f1'] - random_result['macro_f1']) * 100:+.2f} pp"
        ),
    ])

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=full_labels,
        filenames=full_filenames,
        class_names=full_classes,

        actor_p3_similarities=(
            actor_scores
        ),

        upper_body_p3_similarities=(
            upper_scores
        ),

        actor_p3_predictions=(
            actor_result[
                "predictions"
            ]
        ),

        upper_body_p3_predictions=(
            upper_result[
                "predictions"
            ]
        ),

        actor_accuracy=np.asarray(
            actor_result[
                "accuracy"
            ]
        ),

        actor_macro_f1=np.asarray(
            actor_result[
                "macro_f1"
            ]
        ),

        actor_weighted_f1=np.asarray(
            actor_result[
                "weighted_f1"
            ]
        ),

        upper_body_accuracy=np.asarray(
            upper_result[
                "accuracy"
            ]
        ),

        upper_body_macro_f1=np.asarray(
            upper_result[
                "macro_f1"
            ]
        ),

        upper_body_weighted_f1=np.asarray(
            upper_result[
                "weighted_f1"
            ]
        ),

        upper_body_fraction=np.asarray(
            UPPER_BODY_FRACTION
        ),

        model_name=np.asarray(
            MODEL_NAME
        ),

        pretrained=np.asarray(
            PRETRAINED
        ),

        text_strategy=np.asarray(
            "p3"
        ),
    )

    print()
    print(
        f"NPZ saved to:    "
        f"{OUTPUT_NPZ}"
    )

    print(
        f"Report saved to: "
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
