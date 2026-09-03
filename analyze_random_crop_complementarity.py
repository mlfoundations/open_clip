from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


RESULTS_ROOT = Path(r"C:\Projects\stanford40_results")
FIXED_DIR = RESULTS_ROOT / "fixed_fusion"
RANDOM_DIR = RESULTS_ROOT / "random_crops"

FULL_FILE = FIXED_DIR / "test_full_similarities.npz"
ACTOR_FILE = FIXED_DIR / "test_actor_similarities.npz"
FUSION_FILE = FIXED_DIR / "test_fixed_fusion_results.npz"
RANDOM_FILE = RANDOM_DIR / "test_random_crop_similarities.npz"

CSV_OUTPUT = RANDOM_DIR / "random_crop_complementarity.csv"
REPORT_OUTPUT = RANDOM_DIR / "random_crop_complementarity.txt"


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


def transition_counts(base_correct, random_correct):
    return {
        "both_correct": int(np.sum(base_correct & random_correct)),
        "base_wrong_random_correct": int(
            np.sum((~base_correct) & random_correct)
        ),
        "base_correct_random_wrong": int(
            np.sum(base_correct & (~random_correct))
        ),
        "both_wrong": int(
            np.sum((~base_correct) & (~random_correct))
        ),
    }


def percentage(count, total):
    return 100.0 * count / total


def main():
    full = load_npz(FULL_FILE)
    actor = load_npz(ACTOR_FILE)
    fusion = load_npz(FUSION_FILE)
    random_data = load_npz(RANDOM_FILE)

    labels = full["labels"]
    filenames = full["filenames"].astype(str)
    class_names = full["class_names"].astype(str)

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------

    if not np.array_equal(labels, actor["labels"]):
        raise ValueError("Full and actor labels are not aligned.")

    if not np.array_equal(labels, fusion["labels"]):
        raise ValueError("Full and fusion labels are not aligned.")

    if "labels" in random_data:
        if not np.array_equal(labels, random_data["labels"]):
            raise ValueError(
                "Full and random-crop labels are not aligned."
            )

    if not np.array_equal(
        class_names,
        actor["class_names"].astype(str),
    ):
        raise ValueError("Full and actor class order differs.")

    if not np.array_equal(
        class_names,
        fusion["class_names"].astype(str),
    ):
        raise ValueError("Full and fusion class order differs.")

    if "class_names" in random_data:
        if not np.array_equal(
            class_names,
            random_data["class_names"].astype(str),
        ):
            raise ValueError(
                "Random-crop class order differs from Phase 1."
            )

    full_scores = full["similarities"]
    actor_scores = actor["similarities"]
    fusion_scores = fusion["fused_similarities"]

    if "selected_similarities" in random_data:
        random_scores = random_data["selected_similarities"]
    elif "best_confidence_similarities" in random_data:
        random_scores = random_data[
            "best_confidence_similarities"
        ]
    else:
        raise KeyError(
            "Random-crop score matrix not found."
        )

    expected_shape = full_scores.shape

    for name, scores in {
        "full": full_scores,
        "actor": actor_scores,
        "fusion": fusion_scores,
        "random": random_scores,
    }.items():
        if scores.shape != expected_shape:
            raise ValueError(
                f"{name} score shape {scores.shape} "
                f"does not match {expected_shape}."
            )

    frozen_strategy = (
        str(random_data["frozen_strategy"].item())
        if "frozen_strategy" in random_data
        else "unknown"
    )

    if frozen_strategy != "best_confidence":
        raise ValueError(
            "Expected frozen random strategy "
            f"'best_confidence', got '{frozen_strategy}'."
        )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = {
        "Full image": calculate_metrics(labels, full_scores),
        "Actor crop": calculate_metrics(labels, actor_scores),
        "Phase-1 fusion": calculate_metrics(
            labels,
            fusion_scores,
        ),
        "Random best-confidence": calculate_metrics(
            labels,
            random_scores,
        ),
    }

    full_pred = metrics["Full image"]["predictions"]
    actor_pred = metrics["Actor crop"]["predictions"]
    fusion_pred = metrics["Phase-1 fusion"]["predictions"]
    random_pred = metrics[
        "Random best-confidence"
    ]["predictions"]

    full_correct = full_pred == labels
    actor_correct = actor_pred == labels
    fusion_correct = fusion_pred == labels
    random_correct = random_pred == labels

    n = len(labels)

    # ------------------------------------------------------------
    # Pairwise disagreement
    # ------------------------------------------------------------

    disagreements = {
        "Full vs Random": int(
            np.sum(full_pred != random_pred)
        ),
        "Actor vs Random": int(
            np.sum(actor_pred != random_pred)
        ),
        "Fusion vs Random": int(
            np.sum(fusion_pred != random_pred)
        ),
        "Full vs Actor": int(
            np.sum(full_pred != actor_pred)
        ),
    }

    # ------------------------------------------------------------
    # Correction / harm analysis
    # ------------------------------------------------------------

    full_transition = transition_counts(
        full_correct,
        random_correct,
    )

    actor_transition = transition_counts(
        actor_correct,
        random_correct,
    )

    fusion_transition = transition_counts(
        fusion_correct,
        random_correct,
    )

    # ------------------------------------------------------------
    # Oracle / complementarity
    # ------------------------------------------------------------

    oracle_far = (
        full_correct
        | actor_correct
        | random_correct
    )

    oracle_fusion_random = (
        fusion_correct
        | random_correct
    )

    unique_random_far = (
        random_correct
        & (~full_correct)
        & (~actor_correct)
    )

    all_far_wrong = (
        (~full_correct)
        & (~actor_correct)
        & (~random_correct)
    )

    fusion_wrong_random_correct = (
        (~fusion_correct)
        & random_correct
    )

    fusion_correct_random_wrong = (
        fusion_correct
        & (~random_correct)
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    lines = []

    lines.extend([
        "RANDOM-CROP COMPLEMENTARITY ANALYSIS",
        "=" * 88,
        f"Images: {n}",
        f"Classes: {len(class_names)}",
        f"Frozen random strategy: {frozen_strategy}",
        (
            "Phase-1 fusion weights: "
            f"Full={float(fusion['alpha_full']):.2f}, "
            f"Actor={float(fusion['weight_actor']):.2f}"
        ),
        "",
        "TEST PERFORMANCE",
        "-" * 88,
        (
            f"{'Method':<30}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
    ])

    for name, result in metrics.items():
        lines.append(
            f"{name:<30}"
            f"{result['accuracy'] * 100:>13.2f}%"
            f"{result['macro_f1'] * 100:>13.2f}%"
            f"{result['weighted_f1'] * 100:>15.2f}%"
        )

    lines.extend([
        "",
        "PAIRWISE PREDICTION DISAGREEMENT",
        "-" * 88,
    ])

    for name, count in disagreements.items():
        lines.append(
            f"{name:<30}"
            f"{count:>6d} images "
            f"({percentage(count, n):6.2f}%)"
        )

    def append_transition(title, values):
        lines.extend([
            "",
            title,
            "-" * 88,
            (
                f"Both correct:                 "
                f"{values['both_correct']:4d} "
                f"({percentage(values['both_correct'], n):6.2f}%)"
            ),
            (
                f"Baseline wrong -> Random correct: "
                f"{values['base_wrong_random_correct']:4d} "
                f"({percentage(values['base_wrong_random_correct'], n):6.2f}%)"
            ),
            (
                f"Baseline correct -> Random wrong: "
                f"{values['base_correct_random_wrong']:4d} "
                f"({percentage(values['base_correct_random_wrong'], n):6.2f}%)"
            ),
            (
                f"Both wrong:                   "
                f"{values['both_wrong']:4d} "
                f"({percentage(values['both_wrong'], n):6.2f}%)"
            ),
        ])

    append_transition(
        "FULL IMAGE VS RANDOM",
        full_transition,
    )

    append_transition(
        "ACTOR CROP VS RANDOM",
        actor_transition,
    )

    append_transition(
        "PHASE-1 FUSION VS RANDOM",
        fusion_transition,
    )

    lines.extend([
        "",
        "COMPLEMENTARITY / ORACLE",
        "-" * 88,
        (
            "Random correct while Full AND Actor wrong: "
            f"{int(unique_random_far.sum()):4d} "
            f"({percentage(int(unique_random_far.sum()), n):6.2f}%)"
        ),
        (
            "Phase-1 Fusion wrong -> Random correct:     "
            f"{int(fusion_wrong_random_correct.sum()):4d} "
            f"({percentage(int(fusion_wrong_random_correct.sum()), n):6.2f}%)"
        ),
        (
            "Phase-1 Fusion correct -> Random wrong:     "
            f"{int(fusion_correct_random_wrong.sum()):4d} "
            f"({percentage(int(fusion_correct_random_wrong.sum()), n):6.2f}%)"
        ),
        (
            "All Full + Actor + Random wrong:            "
            f"{int(all_far_wrong.sum()):4d} "
            f"({percentage(int(all_far_wrong.sum()), n):6.2f}%)"
        ),
        "",
        (
            "Oracle Full + Actor + Random accuracy:      "
            f"{oracle_far.mean() * 100:.2f}%"
        ),
        (
            "Oracle Phase-1 Fusion + Random accuracy:    "
            f"{oracle_fusion_random.mean() * 100:.2f}%"
        ),
    ])

    report = "\n".join(lines)

    print()
    print(report)

    REPORT_OUTPUT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Per-image CSV for later qualitative analysis
    # ------------------------------------------------------------

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "true_label_index",
            "true_label",
            "full_prediction",
            "actor_prediction",
            "fusion_prediction",
            "random_prediction",
            "full_correct",
            "actor_correct",
            "fusion_correct",
            "random_correct",
            "fusion_wrong_random_correct",
            "fusion_correct_random_wrong",
        ])

        for i in range(n):
            writer.writerow([
                filenames[i],
                int(labels[i]),
                class_names[labels[i]],
                class_names[full_pred[i]],
                class_names[actor_pred[i]],
                class_names[fusion_pred[i]],
                class_names[random_pred[i]],
                bool(full_correct[i]),
                bool(actor_correct[i]),
                bool(fusion_correct[i]),
                bool(random_correct[i]),
                bool(
                    fusion_wrong_random_correct[i]
                ),
                bool(
                    fusion_correct_random_wrong[i]
                ),
            ])

    print()
    print(f"CSV saved to:    {CSV_OUTPUT}")
    print(f"Report saved to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
