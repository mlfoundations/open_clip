from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


RESULTS_ROOT = Path(r"C:\Projects\stanford40_results")
FIXED_DIR = RESULTS_ROOT / "fixed_fusion"
RANDOM_DIR = RESULTS_ROOT / "random_crops"

FULL_FILE = FIXED_DIR / "test_full_similarities.npz"
ACTOR_FILE = FIXED_DIR / "test_actor_similarities.npz"
PHASE1_FUSION_FILE = FIXED_DIR / "test_fixed_fusion_results.npz"
RANDOM_FILE = RANDOM_DIR / "test_random_crop_similarities.npz"

WEIGHTS_FILE = RANDOM_DIR / "selected_three_view_weights.json"

OUTPUT_NPZ = RANDOM_DIR / "test_three_view_fusion_results.npz"
OUTPUT_CSV = RANDOM_DIR / "test_three_view_fusion_predictions.csv"
OUTPUT_REPORT = RANDOM_DIR / "test_three_view_fusion_report.txt"


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


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
    # ------------------------------------------------------------
    # Load frozen weights
    # ------------------------------------------------------------

    weights = json.loads(
        WEIGHTS_FILE.read_text(encoding="utf-8")
    )

    if weights["selection_split"] != "validation":
        raise ValueError(
            "Fusion weights were not selected on validation."
        )

    if weights["random_strategy"] != "best_confidence":
        raise ValueError(
            "Expected frozen random strategy best_confidence."
        )

    w_full = float(weights["w_full"])
    w_actor = float(weights["w_actor"])
    w_random = float(weights["w_random"])

    if not np.isclose(
        w_full + w_actor + w_random,
        1.0,
    ):
        raise ValueError("Fusion weights do not sum to 1.")

    # ------------------------------------------------------------
    # Load frozen test scores
    # ------------------------------------------------------------

    full = load_npz(FULL_FILE)
    actor = load_npz(ACTOR_FILE)
    phase1 = load_npz(PHASE1_FUSION_FILE)
    random_data = load_npz(RANDOM_FILE)

    labels = full["labels"]
    filenames = full["filenames"].astype(str)
    class_names = full["class_names"].astype(str)

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------

    if not np.array_equal(labels, actor["labels"]):
        raise ValueError("Full and actor labels differ.")

    if not np.array_equal(labels, phase1["labels"]):
        raise ValueError("Full and Phase-1 fusion labels differ.")

    if "labels" in random_data:
        if not np.array_equal(labels, random_data["labels"]):
            raise ValueError("Full and random labels differ.")

    if not np.array_equal(
        class_names,
        actor["class_names"].astype(str),
    ):
        raise ValueError("Full and actor class order differs.")

    if not np.array_equal(
        class_names,
        phase1["class_names"].astype(str),
    ):
        raise ValueError("Full and Phase-1 class order differs.")

    if "class_names" in random_data:
        if not np.array_equal(
            class_names,
            random_data["class_names"].astype(str),
        ):
            raise ValueError(
                "Full and random class order differs."
            )

    frozen_random_strategy = str(
        random_data["frozen_strategy"].item()
    )

    if frozen_random_strategy != "best_confidence":
        raise ValueError(
            f"Unexpected test random strategy: "
            f"{frozen_random_strategy}"
        )

    full_scores = full["similarities"]
    actor_scores = actor["similarities"]
    phase1_scores = phase1["fused_similarities"]

    if "selected_similarities" in random_data:
        random_scores = random_data["selected_similarities"]
    else:
        random_scores = random_data[
            "best_confidence_similarities"
        ]

    expected_shape = full_scores.shape

    for name, scores in {
        "full": full_scores,
        "actor": actor_scores,
        "phase1": phase1_scores,
        "random": random_scores,
    }.items():
        if scores.shape != expected_shape:
            raise ValueError(
                f"{name} shape {scores.shape} "
                f"does not match {expected_shape}"
            )

    # ------------------------------------------------------------
    # Frozen 3-view fusion
    # ------------------------------------------------------------

    three_view_scores = (
        w_full * full_scores
        + w_actor * actor_scores
        + w_random * random_scores
    )

    methods = {
        "Full image": calculate_metrics(
            labels,
            full_scores,
        ),
        "Actor crop": calculate_metrics(
            labels,
            actor_scores,
        ),
        "Phase-1 fusion": calculate_metrics(
            labels,
            phase1_scores,
        ),
        "Random best-confidence": calculate_metrics(
            labels,
            random_scores,
        ),
        "Three-view fusion": calculate_metrics(
            labels,
            three_view_scores,
        ),
    }

    final = methods["Three-view fusion"]

    # ------------------------------------------------------------
    # Print final result
    # ------------------------------------------------------------

    lines = [
        "FROZEN THREE-VIEW FUSION TEST RESULT",
        "=" * 88,
        f"Images: {len(labels)}",
        f"Classes: {len(class_names)}",
        "",
        "FROZEN VALIDATION-SELECTED WEIGHTS",
        "-" * 88,
        f"Full:   {w_full:.2f}",
        f"Actor:  {w_actor:.2f}",
        f"Random: {w_random:.2f}",
        "",
        "TEST PERFORMANCE",
        "-" * 88,
        (
            f"{'Method':<30}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
    ]

    for name, result in methods.items():
        lines.append(
            f"{name:<30}"
            f"{result['accuracy'] * 100:>13.2f}%"
            f"{result['macro_f1'] * 100:>13.2f}%"
            f"{result['weighted_f1'] * 100:>15.2f}%"
        )

    random_result = methods["Random best-confidence"]
    phase1_result = methods["Phase-1 fusion"]

    lines.extend([
        "",
        "TEST DELTAS",
        "-" * 88,
        (
            "Three-view vs Phase-1 fusion accuracy: "
            f"{(final['accuracy'] - phase1_result['accuracy']) * 100:+.2f} pp"
        ),
        (
            "Three-view vs Phase-1 fusion Macro-F1: "
            f"{(final['macro_f1'] - phase1_result['macro_f1']) * 100:+.2f} pp"
        ),
        (
            "Three-view vs Random accuracy: "
            f"{(final['accuracy'] - random_result['accuracy']) * 100:+.2f} pp"
        ),
        (
            "Three-view vs Random Macro-F1: "
            f"{(final['macro_f1'] - random_result['macro_f1']) * 100:+.2f} pp"
        ),
    ])

    report = "\n".join(lines)

    print()
    print(report)

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        fused_similarities=three_view_scores,
        labels=labels,
        predictions=final["predictions"],
        filenames=filenames,
        class_names=class_names,
        w_full=np.asarray(w_full),
        w_actor=np.asarray(w_actor),
        w_random=np.asarray(w_random),
        accuracy=np.asarray(final["accuracy"]),
        macro_f1=np.asarray(final["macro_f1"]),
        weighted_f1=np.asarray(final["weighted_f1"]),
        confusion_matrix=confusion_matrix(
            labels,
            final["predictions"],
        ),
    )

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
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
            "prediction",
            "correct",
        ])

        for i in range(len(labels)):
            prediction = final["predictions"][i]

            writer.writerow([
                filenames[i],
                class_names[labels[i]],
                class_names[prediction],
                bool(prediction == labels[i]),
            ])

    print()
    print(f"NPZ saved to:    {OUTPUT_NPZ}")
    print(f"CSV saved to:    {OUTPUT_CSV}")
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
