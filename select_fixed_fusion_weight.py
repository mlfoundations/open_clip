from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


RESULTS_DIR = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
)

FULL_FILE = RESULTS_DIR / "validation_full_similarities.npz"
ACTOR_FILE = RESULTS_DIR / "validation_actor_similarities.npz"

CSV_OUTPUT = RESULTS_DIR / "validation_fusion_weight_search.csv"
JSON_OUTPUT = RESULTS_DIR / "selected_fixed_fusion_weight.json"

# 0.00, 0.05, ..., 1.00
ALPHAS = np.linspace(0.0, 1.0, 21)


def load_results(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=True)

    required_keys = {
        "similarities",
        "labels",
        "filenames",
        "class_names",
        "prompts",
    }

    missing_keys = required_keys.difference(data.files)

    if missing_keys:
        raise KeyError(
            f"{path.name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return {
        "similarities": data["similarities"],
        "labels": data["labels"],
        "filenames": data["filenames"],
        "class_names": data["class_names"],
        "prompts": data["prompts"],
    }


def calculate_metrics(labels, similarities):
    predictions = similarities.argmax(axis=1)

    return {
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full = load_results(FULL_FILE)
    actor = load_results(ACTOR_FILE)

    if full["similarities"].shape != (1521, 40):
        raise ValueError(
            "Unexpected full-view matrix shape: "
            f"{full['similarities'].shape}"
        )

    if actor["similarities"].shape != (1521, 40):
        raise ValueError(
            "Unexpected actor-view matrix shape: "
            f"{actor['similarities'].shape}"
        )

    alignment_checks = {
        "labels": np.array_equal(
            full["labels"],
            actor["labels"],
        ),
        "filenames": np.array_equal(
            full["filenames"],
            actor["filenames"],
        ),
        "class_names": np.array_equal(
            full["class_names"],
            actor["class_names"],
        ),
        "prompts": np.array_equal(
            full["prompts"],
            actor["prompts"],
        ),
    }

    failed_checks = [
        name
        for name, passed in alignment_checks.items()
        if not passed
    ]

    if failed_checks:
        raise ValueError(
            "Full and actor files are not aligned for: "
            f"{failed_checks}"
        )

    labels = full["labels"]
    full_scores = full["similarities"]
    actor_scores = actor["similarities"]

    full_metrics = calculate_metrics(labels, full_scores)
    actor_metrics = calculate_metrics(labels, actor_scores)

    print("VALIDATION SINGLE-VIEW RESULTS")
    print("=" * 68)
    print(
        f"Full view  | Accuracy: "
        f"{full_metrics['accuracy'] * 100:.2f}% | "
        f"Macro F1: {full_metrics['macro_f1'] * 100:.2f}% | "
        f"Weighted F1: "
        f"{full_metrics['weighted_f1'] * 100:.2f}%"
    )
    print(
        f"Actor view | Accuracy: "
        f"{actor_metrics['accuracy'] * 100:.2f}% | "
        f"Macro F1: {actor_metrics['macro_f1'] * 100:.2f}% | "
        f"Weighted F1: "
        f"{actor_metrics['weighted_f1'] * 100:.2f}%"
    )

    search_results = []

    print("\nFIXED FUSION WEIGHT SEARCH")
    print("=" * 68)
    print(
        "Alpha  Full weight  Actor weight  "
        "Accuracy  Macro F1  Weighted F1"
    )
    print("-" * 68)

    for alpha_value in ALPHAS:
        alpha = float(np.round(alpha_value, 2))
        actor_weight = 1.0 - alpha

        fused_scores = (
            alpha * full_scores
            + actor_weight * actor_scores
        )

        metrics = calculate_metrics(labels, fused_scores)

        result = {
            "alpha_full": alpha,
            "weight_actor": actor_weight,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

        search_results.append(result)

        print(
            f"{alpha:>5.2f}"
            f"{alpha:>13.2f}"
            f"{actor_weight:>14.2f}"
            f"{metrics['accuracy'] * 100:>10.2f}%"
            f"{metrics['macro_f1'] * 100:>10.2f}%"
            f"{metrics['weighted_f1'] * 100:>12.2f}%"
        )

    # Primary criterion: Macro F1
    # First tie-breaker: accuracy
    # Second tie-breaker: larger full-image weight
    best_result = max(
        search_results,
        key=lambda result: (
            result["macro_f1"],
            result["accuracy"],
            result["alpha_full"],
        ),
    )

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "alpha_full",
                "weight_actor",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(search_results)

    selected_weight_record = {
        "selection_partition": "validation",
        "selection_metric": "macro_f1",
        "tie_breaker_1": "accuracy",
        "tie_breaker_2": "larger_full_image_weight",
        "alpha_full": best_result["alpha_full"],
        "weight_actor": best_result["weight_actor"],
        "validation_accuracy": best_result["accuracy"],
        "validation_macro_f1": best_result["macro_f1"],
        "validation_weighted_f1": best_result["weighted_f1"],
        "full_view_validation_accuracy": full_metrics["accuracy"],
        "full_view_validation_macro_f1": full_metrics["macro_f1"],
        "actor_view_validation_accuracy": actor_metrics["accuracy"],
        "actor_view_validation_macro_f1": actor_metrics["macro_f1"],
        "formula": (
            "S_fusion = alpha_full * S_full "
            "+ weight_actor * S_actor"
        ),
    }

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected_weight_record,
            file,
            indent=4,
        )

    macro_improvement = (
        best_result["macro_f1"]
        - full_metrics["macro_f1"]
    )

    accuracy_improvement = (
        best_result["accuracy"]
        - full_metrics["accuracy"]
    )

    print("\n" + "=" * 68)
    print("SELECTED FIXED FUSION WEIGHT")
    print("=" * 68)
    print(
        f"Full-image weight (alpha): "
        f"{best_result['alpha_full']:.2f}"
    )
    print(
        f"Actor-view weight:         "
        f"{best_result['weight_actor']:.2f}"
    )
    print(
        f"Validation accuracy:       "
        f"{best_result['accuracy'] * 100:.2f}%"
    )
    print(
        f"Validation Macro F1:       "
        f"{best_result['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Validation Weighted F1:    "
        f"{best_result['weighted_f1'] * 100:.2f}%"
    )
    print(
        f"Accuracy change vs full:   "
        f"{accuracy_improvement * 100:+.2f} points"
    )
    print(
        f"Macro F1 change vs full:   "
        f"{macro_improvement * 100:+.2f} points"
    )
    print(f"\nSearch table saved to: {CSV_OUTPUT}")
    print(f"Selected weight saved to: {JSON_OUTPUT}")
    print("\nThe selected weight is now frozen for test evaluation.")


if __name__ == "__main__":
    main()
