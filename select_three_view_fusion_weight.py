from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


RESULTS_ROOT = Path(r"C:\Projects\stanford40_results")
FIXED_DIR = RESULTS_ROOT / "fixed_fusion"
RANDOM_DIR = RESULTS_ROOT / "random_crops"

FULL_FILE = FIXED_DIR / "validation_full_similarities.npz"
ACTOR_FILE = FIXED_DIR / "validation_actor_similarities.npz"
RANDOM_FILE = RANDOM_DIR / "validation_random_crop_similarities.npz"

CSV_OUTPUT = RANDOM_DIR / "validation_three_view_weight_search.csv"
JSON_OUTPUT = RANDOM_DIR / "selected_three_view_weights.json"

STEP = 0.05


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def metrics(labels, scores):
    pred = scores.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_f1": f1_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            pred,
            average="weighted",
            zero_division=0,
        ),
    }


def main():
    full = load_npz(FULL_FILE)
    actor = load_npz(ACTOR_FILE)
    random_data = load_npz(RANDOM_FILE)

    labels = full["labels"]
    full_scores = full["similarities"]
    actor_scores = actor["similarities"]

    if "selected_similarities" not in random_data:
        raise KeyError(
            "selected_similarities missing from random validation file."
        )

    random_scores = random_data["selected_similarities"]

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------

    if not np.array_equal(labels, actor["labels"]):
        raise ValueError("Full and actor validation labels differ.")

    if not np.array_equal(labels, random_data["labels"]):
        raise ValueError("Full and random validation labels differ.")

    if full_scores.shape != actor_scores.shape:
        raise ValueError("Full and actor score shapes differ.")

    if full_scores.shape != random_scores.shape:
        raise ValueError(
            f"Random shape {random_scores.shape} differs "
            f"from Phase-1 shape {full_scores.shape}."
        )

    full_classes = full["class_names"].astype(str)
    actor_classes = actor["class_names"].astype(str)
    random_classes = random_data["class_names"].astype(str)

    if not np.array_equal(full_classes, actor_classes):
        raise ValueError("Full and actor class order differs.")

    if not np.array_equal(full_classes, random_classes):
        raise ValueError("Full and random class order differs.")

    strategy = str(random_data["selected_strategy"].item())

    if strategy != "best_confidence":
        raise ValueError(
            f"Expected best_confidence, got {strategy}."
        )

    # ------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------

    full_metrics = metrics(labels, full_scores)
    actor_metrics = metrics(labels, actor_scores)
    random_metrics = metrics(labels, random_scores)

    print()
    print("THREE-VIEW FUSION VALIDATION SEARCH")
    print("=" * 88)
    print(f"Images:          {len(labels)}")
    print(f"Classes:         {len(full_classes)}")
    print(f"Random strategy: {strategy}")
    print(f"Weight step:     {STEP:.2f}")

    print()
    print("INDIVIDUAL VALIDATION RESULTS")
    print("-" * 88)

    for name, result in [
        ("Full", full_metrics),
        ("Actor", actor_metrics),
        ("Random", random_metrics),
    ]:
        print(
            f"{name:<12}"
            f"Accuracy={result['accuracy'] * 100:6.2f}% | "
            f"Macro-F1={result['macro_f1'] * 100:6.2f}% | "
            f"Weighted-F1={result['weighted_f1'] * 100:6.2f}%"
        )

    # ------------------------------------------------------------
    # Exhaustive simplex search
    # ------------------------------------------------------------

    rows = []

    units = int(round(1.0 / STEP))

    for full_units in range(units + 1):
        for actor_units in range(units - full_units + 1):

            random_units = units - full_units - actor_units

            w_full = full_units / units
            w_actor = actor_units / units
            w_random = random_units / units

            fused = (
                w_full * full_scores
                + w_actor * actor_scores
                + w_random * random_scores
            )

            result = metrics(labels, fused)

            rows.append({
                "w_full": w_full,
                "w_actor": w_actor,
                "w_random": w_random,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            })

    # Primary criterion:
    # Macro-F1, then Accuracy, then Weighted-F1.
    rows.sort(
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["weighted_f1"],
        ),
        reverse=True,
    )

    best = rows[0]

    print()
    print("TOP 10 VALIDATION WEIGHT COMBINATIONS")
    print("-" * 88)

    print(
        f"{'Rank':<6}"
        f"{'Full':>8}"
        f"{'Actor':>8}"
        f"{'Random':>9}"
        f"{'Accuracy':>12}"
        f"{'Macro-F1':>12}"
        f"{'Weighted-F1':>14}"
    )

    for rank, row in enumerate(rows[:10], start=1):
        print(
            f"{rank:<6}"
            f"{row['w_full']:>8.2f}"
            f"{row['w_actor']:>8.2f}"
            f"{row['w_random']:>9.2f}"
            f"{row['accuracy'] * 100:>11.2f}%"
            f"{row['macro_f1'] * 100:>11.2f}%"
            f"{row['weighted_f1'] * 100:>13.2f}%"
        )

    print()
    print("=" * 88)
    print("SELECTED THREE-VIEW WEIGHTS")
    print("=" * 88)

    print(f"Full:        {best['w_full']:.2f}")
    print(f"Actor:       {best['w_actor']:.2f}")
    print(f"Random:      {best['w_random']:.2f}")
    print()
    print(f"Accuracy:    {best['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:    {best['macro_f1'] * 100:.2f}%")
    print(f"Weighted-F1: {best['weighted_f1'] * 100:.2f}%")

    print()
    print(
        "Macro-F1 gain vs Random alone: "
        f"{(best['macro_f1'] - random_metrics['macro_f1']) * 100:+.2f} pp"
    )

    # ------------------------------------------------------------
    # Save complete validation search
    # ------------------------------------------------------------

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "w_full",
                "w_actor",
                "w_random",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    selected = {
        "selection_split": "validation",
        "selection_metric": "macro_f1_then_accuracy_then_weighted_f1",
        "weight_step": STEP,
        "random_strategy": strategy,
        "w_full": best["w_full"],
        "w_actor": best["w_actor"],
        "w_random": best["w_random"],
        "validation_accuracy": best["accuracy"],
        "validation_macro_f1": best["macro_f1"],
        "validation_weighted_f1": best["weighted_f1"],
    }

    JSON_OUTPUT.write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Search CSV saved to: {CSV_OUTPUT}")
    print(f"Frozen weights saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
