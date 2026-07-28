from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


RESULTS_DIR = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
)

FULL_FILE = RESULTS_DIR / "test_full_similarities.npz"
ACTOR_FILE = RESULTS_DIR / "test_actor_similarities.npz"

WEIGHT_FILE = RESULTS_DIR / "selected_fixed_fusion_weight.json"

OUTPUT_FILE = RESULTS_DIR / "test_fixed_fusion_results.npz"
REPORT_FILE = RESULTS_DIR / "test_fixed_fusion_report.txt"

EXPECTED_SHAPE = (1921, 40)


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = {
        "similarities",
        "labels",
        "filenames",
        "class_names",
        "prompts",
    }

    missing = required.difference(data.files)

    if missing:
        raise KeyError(
            f"{path.name} is missing keys: {sorted(missing)}"
        )

    return {
        key: data[key]
        for key in required
    }


def calculate_metrics(labels, scores):
    predictions = scores.argmax(axis=1)

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


def main():
    full = load_npz(FULL_FILE)
    actor = load_npz(ACTOR_FILE)

    if full["similarities"].shape != EXPECTED_SHAPE:
        raise ValueError(
            "Unexpected full score shape: "
            f"{full['similarities'].shape}"
        )

    if actor["similarities"].shape != EXPECTED_SHAPE:
        raise ValueError(
            "Unexpected actor score shape: "
            f"{actor['similarities'].shape}"
        )

    for key in [
        "labels",
        "filenames",
        "class_names",
        "prompts",
    ]:
        if not np.array_equal(full[key], actor[key]):
            raise ValueError(
                f"Full and actor test data are not aligned: {key}"
            )

    if not WEIGHT_FILE.exists():
        raise FileNotFoundError(
            f"Weight file not found: {WEIGHT_FILE}"
        )

    with WEIGHT_FILE.open("r", encoding="utf-8") as file:
        weight_record = json.load(file)

    alpha_full = float(weight_record["alpha_full"])
    weight_actor = float(weight_record["weight_actor"])

    if not np.isclose(alpha_full, 0.60):
        raise ValueError(
            f"Expected frozen full weight 0.60, got {alpha_full}"
        )

    if not np.isclose(weight_actor, 0.40):
        raise ValueError(
            f"Expected frozen actor weight 0.40, got {weight_actor}"
        )

    labels = full["labels"]
    full_scores = full["similarities"]
    actor_scores = actor["similarities"]

    fused_scores = (
        alpha_full * full_scores
        + weight_actor * actor_scores
    )

    full_metrics = calculate_metrics(labels, full_scores)
    actor_metrics = calculate_metrics(labels, actor_scores)
    fusion_metrics = calculate_metrics(labels, fused_scores)

    class_names = [
        str(name)
        for name in full["class_names"]
    ]

    report = classification_report(
        labels,
        fusion_metrics["predictions"],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        fusion_metrics["predictions"],
    )

    np.savez_compressed(
        OUTPUT_FILE,
        fused_similarities=fused_scores,
        labels=labels,
        predictions=fusion_metrics["predictions"],
        filenames=full["filenames"],
        class_names=full["class_names"],
        alpha_full=np.asarray(alpha_full),
        weight_actor=np.asarray(weight_actor),
        accuracy=np.asarray(fusion_metrics["accuracy"]),
        macro_f1=np.asarray(fusion_metrics["macro_f1"]),
        weighted_f1=np.asarray(
            fusion_metrics["weighted_f1"]
        ),
        confusion_matrix=matrix,
    )

    accuracy_change = (
        fusion_metrics["accuracy"]
        - full_metrics["accuracy"]
    )

    macro_f1_change = (
        fusion_metrics["macro_f1"]
        - full_metrics["macro_f1"]
    )

    summary = (
        "OFFICIAL TEST RESULTS\n"
        + "=" * 72
        + "\n"
        + (
            "Full view   | "
            f"Accuracy: {full_metrics['accuracy'] * 100:.2f}% | "
            f"Macro F1: {full_metrics['macro_f1'] * 100:.2f}% | "
            f"Weighted F1: "
            f"{full_metrics['weighted_f1'] * 100:.2f}%\n"
        )
        + (
            "Actor view  | "
            f"Accuracy: {actor_metrics['accuracy'] * 100:.2f}% | "
            f"Macro F1: {actor_metrics['macro_f1'] * 100:.2f}% | "
            f"Weighted F1: "
            f"{actor_metrics['weighted_f1'] * 100:.2f}%\n"
        )
        + (
            "Fixed fusion| "
            f"Accuracy: {fusion_metrics['accuracy'] * 100:.2f}% | "
            f"Macro F1: {fusion_metrics['macro_f1'] * 100:.2f}% | "
            f"Weighted F1: "
            f"{fusion_metrics['weighted_f1'] * 100:.2f}%\n"
        )
        + "=" * 72
        + "\n"
        + f"Frozen full-image weight: {alpha_full:.2f}\n"
        + f"Frozen actor-view weight: {weight_actor:.2f}\n"
        + (
            "Accuracy change versus full view: "
            f"{accuracy_change * 100:+.2f} percentage points\n"
        )
        + (
            "Macro F1 change versus full view: "
            f"{macro_f1_change * 100:+.2f} percentage points\n"
        )
    )

    print(summary)
    print("FUSION CLASSIFICATION REPORT")
    print(report)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write(summary)
        file.write("\nFUSION CLASSIFICATION REPORT\n")
        file.write(report)

    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Report saved to:  {REPORT_FILE}")
    print("\nThe frozen weight was applied once without test tuning.")


if __name__ == "__main__":
    main()