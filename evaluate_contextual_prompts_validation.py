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


VALIDATION_DIR = Path(
    r"C:\Projects\stanford40_split\validation"
)

RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

OUTPUT_DIR = RESULTS_ROOT / "context_prompts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = (
    RESULTS_ROOT
    / "fixed_fusion"
    / "validation_full_similarities.npz"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "validation_contextual_prompt_results.npz"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "validation_contextual_prompt_comparison.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "selected_contextual_prompt_strategy.json"
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


def encode_prompts(
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
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {VALIDATION_DIR}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("CONTEXTUAL PROMPT VALIDATION EXPERIMENT")
    print("=" * 84)
    print(f"Device:     {device}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Pretrained: {PRETRAINED}")

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
        VALIDATION_DIR,
        transform=preprocess,
    )

    class_names = dataset.classes

    validate_prompt_bank(class_names)

    print(f"Images:     {len(dataset)}")
    print(f"Classes:    {len(class_names)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------
    # P0 / P1 / P2 text representations
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
    print("Encoding P0 basic prompts...")
    text_p0 = encode_prompts(
        model,
        tokenizer,
        prompts_p0,
        device,
    )

    print("Encoding P1 action prompts...")
    text_p1 = encode_prompts(
        model,
        tokenizer,
        prompts_p1,
        device,
    )

    print("Encoding P2 contextual prompts...")
    text_p2 = encode_prompts(
        model,
        tokenizer,
        prompts_p2,
        device,
    )

    # ------------------------------------------------------------
    # P3 = normalized text-embedding ensemble
    # ------------------------------------------------------------

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
    # Encode each validation image ONCE
    # ------------------------------------------------------------

    all_features = []
    all_labels = []

    print()
    print("Encoding validation images once...")

    processed = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            features = model.encode_image(images)

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
                    f"Processed {processed}/{len(dataset)}"
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
                VALIDATION_DIR
            )
        ).replace("\\", "/")
        for path, _ in dataset.samples
    ])

    # ------------------------------------------------------------
    # Calculate similarities
    # ------------------------------------------------------------

    text_features = {
        "p0": text_p0.cpu().numpy(),
        "p1": text_p1.cpu().numpy(),
        "p2": text_p2.cpu().numpy(),
        "p3": text_p3.cpu().numpy(),
    }

    similarities = {
        strategy: (
            image_features @ matrix.T
        ).astype(np.float32)
        for strategy, matrix
        in text_features.items()
    }

    metrics = {
        strategy: calculate_metrics(
            labels,
            scores,
        )
        for strategy, scores
        in similarities.items()
    }

    # Macro-F1 -> Accuracy -> Weighted-F1
    selected_strategy = max(
        metrics,
        key=lambda strategy: (
            metrics[strategy]["macro_f1"],
            metrics[strategy]["accuracy"],
            metrics[strategy]["weighted_f1"],
        ),
    )

    selected = metrics[
        selected_strategy
    ]

    # ------------------------------------------------------------
    # P0 integrity check
    # ------------------------------------------------------------

    print()
    print("PHASE-1 P0 INTEGRITY CHECK")
    print("-" * 84)

    with np.load(
        BASELINE_FILE,
        allow_pickle=True,
    ) as baseline:

        if not np.array_equal(
            labels,
            baseline["labels"],
        ):
            raise ValueError(
                "Labels differ from Phase-1 validation baseline."
            )

        if not np.array_equal(
            np.asarray(class_names),
            baseline["class_names"].astype(str),
        ):
            raise ValueError(
                "Class order differs from Phase-1 baseline."
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

    print(
        "Frozen Phase-1: "
        f"Accuracy={baseline_accuracy * 100:.2f}% | "
        f"Macro-F1={baseline_macro * 100:.2f}% | "
        f"Weighted-F1={baseline_weighted * 100:.2f}%"
    )

    print(
        "Fresh P0:       "
        f"Accuracy={metrics['p0']['accuracy'] * 100:.2f}% | "
        f"Macro-F1={metrics['p0']['macro_f1'] * 100:.2f}% | "
        f"Weighted-F1={metrics['p0']['weighted_f1'] * 100:.2f}%"
    )

    accuracy_delta = (
        metrics["p0"]["accuracy"]
        - baseline_accuracy
    ) * 100

    macro_delta = (
        metrics["p0"]["macro_f1"]
        - baseline_macro
    ) * 100

    print(
        "P0 delta:       "
        f"Accuracy={accuracy_delta:+.4f} pp | "
        f"Macro-F1={macro_delta:+.4f} pp"
    )

    # Fail rather than continue if P0 is not reproducible.
    if (
        abs(accuracy_delta) > 0.001
        or abs(macro_delta) > 0.001
    ):
        raise ValueError(
            "Fresh P0 does not reproduce the frozen "
            "Phase-1 baseline. Stop before selecting prompts."
        )

    # ------------------------------------------------------------
    # Print validation comparison
    # ------------------------------------------------------------

    descriptions = {
        "p0": "Basic",
        "p1": "Action",
        "p2": "Context",
        "p3": "Ensemble",
    }

    print()
    print("VALIDATION PROMPT COMPARISON")
    print("=" * 84)

    print(
        f"{'Strategy':<18}"
        f"{'Accuracy':>14}"
        f"{'Macro-F1':>14}"
        f"{'Weighted-F1':>16}"
    )

    print("-" * 84)

    rows = []

    for strategy in [
        "p0",
        "p1",
        "p2",
        "p3",
    ]:
        result = metrics[strategy]

        display_name = (
            f"{strategy.upper()} "
            f"{descriptions[strategy]}"
        )

        print(
            f"{display_name:<18}"
            f"{result['accuracy'] * 100:>13.2f}%"
            f"{result['macro_f1'] * 100:>13.2f}%"
            f"{result['weighted_f1'] * 100:>15.2f}%"
        )

        rows.append({
            "strategy": strategy,
            "description": descriptions[
                strategy
            ],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result[
                "weighted_f1"
            ],
        })

    print()
    print("=" * 84)
    print(
        "SELECTED VALIDATION STRATEGY: "
        f"{selected_strategy.upper()} "
        f"({descriptions[selected_strategy]})"
    )

    print(
        f"Accuracy:    "
        f"{selected['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1:    "
        f"{selected['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted-F1: "
        f"{selected['weighted_f1'] * 100:.2f}%"
    )

    print(
        "Macro-F1 gain vs P0: "
        f"{(selected['macro_f1'] - metrics['p0']['macro_f1']) * 100:+.2f} pp"
    )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "description",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Save score matrices
    # ------------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels,
        filenames=filenames,
        class_names=np.asarray(
            class_names
        ),

        prompts_p0=np.asarray(
            prompts_p0
        ),
        prompts_p1=np.asarray(
            prompts_p1
        ),
        prompts_p2=np.asarray(
            prompts_p2
        ),

        p0_similarities=similarities[
            "p0"
        ],
        p1_similarities=similarities[
            "p1"
        ],
        p2_similarities=similarities[
            "p2"
        ],
        p3_similarities=similarities[
            "p3"
        ],

        selected_strategy=np.asarray(
            selected_strategy
        ),
        selected_similarities=(
            similarities[
                selected_strategy
            ]
        ),
        selected_predictions=(
            selected["predictions"]
        ),

        selected_accuracy=np.asarray(
            selected["accuracy"]
        ),
        selected_macro_f1=np.asarray(
            selected["macro_f1"]
        ),
        selected_weighted_f1=np.asarray(
            selected["weighted_f1"]
        ),

        model_name=np.asarray(
            MODEL_NAME
        ),
        pretrained=np.asarray(
            PRETRAINED
        ),
    )

    # ------------------------------------------------------------
    # Save frozen strategy
    # ------------------------------------------------------------

    selected_json = {
        "selection_split": "validation",
        "selection_rule": (
            "macro_f1_then_accuracy_then_weighted_f1"
        ),
        "selected_strategy": (
            selected_strategy
        ),
        "strategy_description": (
            descriptions[
                selected_strategy
            ]
        ),
        "validation_accuracy": (
            selected["accuracy"]
        ),
        "validation_macro_f1": (
            selected["macro_f1"]
        ),
        "validation_weighted_f1": (
            selected["weighted_f1"]
        ),
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "num_classes": len(
            class_names
        ),
        "num_validation_images": len(
            dataset
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            selected_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Results NPZ saved to: {OUTPUT_NPZ}"
    )
    print(
        f"Comparison CSV saved to: {OUTPUT_CSV}"
    )
    print(
        f"Frozen strategy saved to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
