from pathlib import Path
import csv
import json

import numpy as np
import open_clip
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
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

OUTPUT_DIR = RESULTS_ROOT / "context_prompts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FILE = (
    OUTPUT_DIR
    / "selected_contextual_prompt_strategy.json"
)

BASELINE_FILE = (
    RESULTS_ROOT
    / "fixed_fusion"
    / "test_full_similarities.npz"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "test_contextual_prompt_results.npz"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "test_contextual_prompt_predictions.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "test_contextual_prompt_report.txt"
)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 32
NUM_WORKERS = 0


def calculate_metrics(labels, scores):
    predictions = scores.argmax(axis=1)

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
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    return features


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    selection = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if selection["selection_split"] != "validation":
        raise ValueError(
            "Prompt strategy was not selected "
            "using validation."
        )

    selected_strategy = selection[
        "selected_strategy"
    ]

    if selected_strategy != "p3":
        raise ValueError(
            f"Expected frozen P3 strategy, "
            f"got {selected_strategy}."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("FROZEN CONTEXTUAL PROMPT TEST")
    print("=" * 84)
    print(f"Device:          {device}")
    print(f"Model:           {MODEL_NAME}")
    print(f"Pretrained:      {PRETRAINED}")
    print(f"Frozen strategy: {selected_strategy.upper()}")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
        )
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    model.eval()

    dataset = ImageFolder(
        TEST_DIR,
        transform=preprocess,
    )

    class_names = dataset.classes

    validate_prompt_bank(class_names)

    print(f"Images:          {len(dataset)}")
    print(f"Classes:         {len(class_names)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------
    # Recreate frozen text representations
    # ------------------------------------------------------------

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

    print()
    print("Encoding frozen prompt representations...")

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
    # Encode test images once
    # ------------------------------------------------------------

    all_features = []
    all_labels = []

    print()
    print("Encoding test images once...")

    processed = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            features = model.encode_image(
                images
            )

            features = features / features.norm(
                dim=-1,
                keepdim=True,
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            processed += len(labels)

            if (
                processed % 200 == 0
                or processed == len(dataset)
            ):
                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                )

    image_features = np.concatenate(
        all_features,
        axis=0,
    ).astype(np.float32)

    labels = np.concatenate(
        all_labels,
        axis=0,
    ).astype(np.int64)

    filenames = np.asarray([
        str(
            Path(path).relative_to(
                TEST_DIR
            )
        ).replace("\\", "/")
        for path, _ in dataset.samples
    ])

    # ------------------------------------------------------------
    # Only evaluate frozen P0 control and P3 winner
    # ------------------------------------------------------------

    p0_scores = (
        image_features
        @ text_p0.cpu().numpy().T
    ).astype(np.float32)

    p3_scores = (
        image_features
        @ text_p3.cpu().numpy().T
    ).astype(np.float32)

    p0_metrics = calculate_metrics(
        labels,
        p0_scores,
    )

    p3_metrics = calculate_metrics(
        labels,
        p3_scores,
    )

    # ------------------------------------------------------------
    # Integrity check against Phase-1 frozen test
    # ------------------------------------------------------------

    with np.load(
        BASELINE_FILE,
        allow_pickle=True,
    ) as baseline:

        if not np.array_equal(
            labels,
            baseline["labels"],
        ):
            raise ValueError(
                "Test labels differ from "
                "Phase-1 baseline."
            )

        if not np.array_equal(
            np.asarray(class_names),
            baseline["class_names"].astype(str),
        ):
            raise ValueError(
                "Class order differs from "
                "Phase-1 baseline."
            )

        baseline_accuracy = float(
            baseline["accuracy"]
        )

        baseline_macro = float(
            baseline["macro_f1"]
        )

        baseline_weighted = float(
            baseline["weighted_f1"]
        )

    accuracy_delta_check = (
        p0_metrics["accuracy"]
        - baseline_accuracy
    ) * 100

    macro_delta_check = (
        p0_metrics["macro_f1"]
        - baseline_macro
    ) * 100

    print()
    print("P0 TEST INTEGRITY CHECK")
    print("-" * 84)

    print(
        "Frozen Phase-1: "
        f"Accuracy={baseline_accuracy * 100:.2f}% | "
        f"Macro-F1={baseline_macro * 100:.2f}% | "
        f"Weighted-F1={baseline_weighted * 100:.2f}%"
    )

    print(
        "Fresh P0:       "
        f"Accuracy={p0_metrics['accuracy'] * 100:.2f}% | "
        f"Macro-F1={p0_metrics['macro_f1'] * 100:.2f}% | "
        f"Weighted-F1={p0_metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        "P0 delta:       "
        f"Accuracy={accuracy_delta_check:+.4f} pp | "
        f"Macro-F1={macro_delta_check:+.4f} pp"
    )

    if (
        abs(accuracy_delta_check) > 0.001
        or abs(macro_delta_check) > 0.001
    ):
        raise ValueError(
            "P0 test reproduction failed."
        )

    # ------------------------------------------------------------
    # Final frozen result
    # ------------------------------------------------------------

    accuracy_gain = (
        p3_metrics["accuracy"]
        - p0_metrics["accuracy"]
    ) * 100

    macro_gain = (
        p3_metrics["macro_f1"]
        - p0_metrics["macro_f1"]
    ) * 100

    weighted_gain = (
        p3_metrics["weighted_f1"]
        - p0_metrics["weighted_f1"]
    ) * 100

    lines = [
        "",
        "FROZEN CONTEXTUAL PROMPT TEST RESULT",
        "=" * 84,
        f"Images: {len(labels)}",
        f"Classes: {len(class_names)}",
        "Frozen strategy: P3 embedding ensemble",
        "",
        "TEST PERFORMANCE",
        "-" * 84,
        (
            f"{'Method':<22}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
        (
            f"{'P0 Basic':<22}"
            f"{p0_metrics['accuracy'] * 100:>13.2f}%"
            f"{p0_metrics['macro_f1'] * 100:>13.2f}%"
            f"{p0_metrics['weighted_f1'] * 100:>15.2f}%"
        ),
        (
            f"{'P3 Ensemble':<22}"
            f"{p3_metrics['accuracy'] * 100:>13.2f}%"
            f"{p3_metrics['macro_f1'] * 100:>13.2f}%"
            f"{p3_metrics['weighted_f1'] * 100:>15.2f}%"
        ),
        "",
        "TEST DELTAS",
        "-" * 84,
        (
            f"Accuracy gain vs P0:    "
            f"{accuracy_gain:+.2f} pp"
        ),
        (
            f"Macro-F1 gain vs P0:    "
            f"{macro_gain:+.2f} pp"
        ),
        (
            f"Weighted-F1 gain vs P0: "
            f"{weighted_gain:+.2f} pp"
        ),
    ]

    report = "\n".join(lines)

    print(report)

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        labels=labels,
        filenames=filenames,
        class_names=np.asarray(
            class_names
        ),
        p0_similarities=p0_scores,
        p3_similarities=p3_scores,
        predictions=p3_metrics[
            "predictions"
        ],
        accuracy=np.asarray(
            p3_metrics["accuracy"]
        ),
        macro_f1=np.asarray(
            p3_metrics["macro_f1"]
        ),
        weighted_f1=np.asarray(
            p3_metrics["weighted_f1"]
        ),
        selected_strategy=np.asarray(
            "p3"
        ),
        model_name=np.asarray(
            MODEL_NAME
        ),
        pretrained=np.asarray(
            PRETRAINED
        ),
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "true_label",
            "p0_prediction",
            "p3_prediction",
            "p0_correct",
            "p3_correct",
        ])

        for i in range(len(labels)):
            p0_prediction = (
                p0_metrics[
                    "predictions"
                ][i]
            )

            p3_prediction = (
                p3_metrics[
                    "predictions"
                ][i]
            )

            writer.writerow([
                filenames[i],
                class_names[labels[i]],
                class_names[p0_prediction],
                class_names[p3_prediction],
                bool(
                    p0_prediction
                    == labels[i]
                ),
                bool(
                    p3_prediction
                    == labels[i]
                ),
            ])

    print()
    print(f"NPZ saved to:    {OUTPUT_NPZ}")
    print(f"CSV saved to:    {OUTPUT_CSV}")
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
