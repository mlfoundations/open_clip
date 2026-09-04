from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path(
    r"C:\Projects\stanford40_split\test"
)

RESULTS_DIR = Path(
    r"C:\Projects\stanford40_results\resnet50_baseline_clean"
)

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "best_resnet50_validation.pt"
)

SELECTION_PATH = (
    RESULTS_DIR
    / "selected_resnet50_validation.json"
)

OUTPUT_NPZ = (
    RESULTS_DIR
    / "test_resnet50_results.npz"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "test_resnet50_predictions.csv"
)

OUTPUT_REPORT = (
    RESULTS_DIR
    / "test_resnet50_report.txt"
)


EXPECTED_TEST = 1921
EXPECTED_CLASSES = 40

BATCH_SIZE = 16
NUM_WORKERS = 0


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
):

    return {
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


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "FROZEN SUPERVISED RESNET-50 TEST"
    )
    print("=" * 92)

    # --------------------------------------------------------
    # Confirm validation-only selection
    # --------------------------------------------------------

    selection = json.loads(
        SELECTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        selection["selection_split"]
        != "validation"
    ):
        raise ValueError(
            "Checkpoint was not selected "
            "using validation."
        )

    if selection["test_used"]:
        raise ValueError(
            "Selection metadata says "
            "test data was used."
        )

    print(
        f"Selected epoch: "
        f"{selection['selected_epoch']}"
    )

    print(
        "Selection split: validation"
    )

    print(
        "Test used for model selection: NO"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Same deterministic transform used for validation
    # --------------------------------------------------------

    test_transform = transforms.Compose([

        transforms.Resize(
            232
        ),

        transforms.CenterCrop(
            224
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.485,
                0.456,
                0.406,
            ),
            std=(
                0.229,
                0.224,
                0.225,
            ),
        ),
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = ImageFolder(
        TEST_DIR,
        transform=test_transform,
    )

    if len(test_dataset) != EXPECTED_TEST:

        raise ValueError(
            f"Expected {EXPECTED_TEST} "
            f"test images, "
            f"found {len(test_dataset)}."
        )

    if (
        len(test_dataset.classes)
        != EXPECTED_CLASSES
    ):

        raise ValueError(
            f"Expected {EXPECTED_CLASSES} "
            f"classes, "
            f"found "
            f"{len(test_dataset.classes)}."
        )

    print()
    print(
        "TEST DATASET"
    )

    print(
        f"Images:  {len(test_dataset)}"
    )

    print(
        f"Classes: "
        f"{len(test_dataset.classes)}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_classes = (
        checkpoint[
            "classes"
        ]
    )

    checkpoint_mapping = (
        checkpoint[
            "class_to_idx"
        ]
    )

    if (
        test_dataset.classes
        != checkpoint_classes
    ):

        raise ValueError(
            "Test class ordering differs "
            "from training checkpoint."
        )

    if (
        test_dataset.class_to_idx
        != checkpoint_mapping
    ):

        raise ValueError(
            "Test class-to-index mapping "
            "differs from checkpoint."
        )

    print(
        "Class-order integrity: PASSED"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['selected_epoch']}"
    )

    print(
        f"Checkpoint validation Macro-F1: "
        f"{checkpoint['validation_macro_f1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        EXPECTED_CLASSES,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Test inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_logits = []
    all_filenames = []

    processed = 0

    print()
    print(
        "Evaluating frozen checkpoint..."
    )

    with torch.inference_mode():

        for batch_index, (
            images,
            labels,
        ) in enumerate(
            test_loader
        ):

            images = images.to(
                device
            )

            logits = model(
                images
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            all_labels.append(
                labels.numpy()
            )

            all_predictions.append(
                predictions
                .cpu()
                .numpy()
            )

            all_logits.append(
                logits
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            batch_start = (
                batch_index
                * BATCH_SIZE
            )

            batch_end = (
                batch_start
                + len(labels)
            )

            for path, _ in (
                test_dataset.samples[
                    batch_start:batch_end
                ]
            ):

                all_filenames.append(
                    str(
                        Path(path).relative_to(
                            TEST_DIR
                        )
                    ).replace(
                        "\\",
                        "/",
                    )
                )

            processed += len(
                labels
            )

            if (
                processed % 200
                < BATCH_SIZE
                or processed
                == len(test_dataset)
            ):

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(test_dataset)}"
                )

    labels_np = np.concatenate(
        all_labels
    ).astype(
        np.int64
    )

    predictions_np = np.concatenate(
        all_predictions
    ).astype(
        np.int64
    )

    logits_np = np.concatenate(
        all_logits,
        axis=0,
    )

    filenames_np = np.asarray(
        all_filenames
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    result = calculate_metrics(
        labels_np,
        predictions_np,
    )

    validation_accuracy = float(
        selection[
            "validation_accuracy"
        ]
    )

    validation_macro = float(
        selection[
            "validation_macro_f1"
        ]
    )

    validation_weighted = float(
        selection[
            "validation_weighted_f1"
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = [
        "",
        (
            "FROZEN SUPERVISED "
            "RESNET-50 TEST RESULT"
        ),
        "=" * 92,

        (
            f"Images: "
            f"{len(labels_np)}"
        ),

        (
            f"Selected epoch: "
            f"{selection['selected_epoch']}"
        ),

        (
            "Model selection: "
            "validation only"
        ),

        "",
        "PERFORMANCE",
        "-" * 92,

        (
            f"{'Split':<20}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),

        (
            f"{'Validation':<20}"
            f"{validation_accuracy * 100:>13.2f}%"
            f"{validation_macro * 100:>13.2f}%"
            f"{validation_weighted * 100:>15.2f}%"
        ),

        (
            f"{'Test':<20}"
            f"{result['accuracy'] * 100:>13.2f}%"
            f"{result['macro_f1'] * 100:>13.2f}%"
            f"{result['weighted_f1'] * 100:>15.2f}%"
        ),

        "",
        "GENERALIZATION DELTA",
        "-" * 92,

        (
            "Accuracy:    "
            f"{(result['accuracy'] - validation_accuracy) * 100:+.2f} pp"
        ),

        (
            "Macro-F1:    "
            f"{(result['macro_f1'] - validation_macro) * 100:+.2f} pp"
        ),

        (
            "Weighted-F1: "
            f"{(result['weighted_f1'] - validation_weighted) * 100:+.2f} pp"
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

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels_np,

        predictions=(
            predictions_np
        ),

        logits=(
            logits_np
        ),

        filenames=(
            filenames_np
        ),

        class_names=np.asarray(
            test_dataset.classes
        ),

        accuracy=np.asarray(
            result[
                "accuracy"
            ]
        ),

        macro_f1=np.asarray(
            result[
                "macro_f1"
            ]
        ),

        weighted_f1=np.asarray(
            result[
                "weighted_f1"
            ]
        ),

        selected_epoch=np.asarray(
            selection[
                "selected_epoch"
            ]
        ),

        validation_accuracy=np.asarray(
            validation_accuracy
        ),

        validation_macro_f1=np.asarray(
            validation_macro
        ),

        validation_weighted_f1=np.asarray(
            validation_weighted
        ),

        model_type=np.asarray(
            "supervised_non_vlm_resnet50"
        ),

        pretraining=np.asarray(
            "IMAGENET1K_V2"
        ),
    )

    # --------------------------------------------------------
    # Prediction CSV
    # --------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "filename",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "correct",
        ])

        for (
            filename,
            label,
            prediction,
        ) in zip(
            filenames_np,
            labels_np,
            predictions_np,
        ):

            writer.writerow([
                filename,
                int(label),

                test_dataset.classes[
                    label
                ],

                int(prediction),

                test_dataset.classes[
                    prediction
                ],

                bool(
                    label
                    == prediction
                ),
            ])

    print()
    print(
        f"NPZ:    "
        f"{OUTPUT_NPZ}"
    )

    print(
        f"CSV:    "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Report: "
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
