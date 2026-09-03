from pathlib import Path
import csv
import json

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

FULL_PATH = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_prompt_results.npz"
)

RANDOM_PATH = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_random_same_crop.npz"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "context_prompts"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "validation_confidence_gate_search.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "selected_confidence_gate.json"
)


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


def top1_top2_margin(scores):
    top_two = np.partition(
        scores,
        -2,
        axis=1,
    )[:, -2:]

    top_two.sort(
        axis=1
    )

    return (
        top_two[:, 1]
        - top_two[:, 0]
    )


def main():
    print()
    print(
        "VALIDATION CONFIDENCE-GATE THRESHOLD SEARCH"
    )
    print("=" * 94)

    # ------------------------------------------------------------
    # Load Full-P3
    # ------------------------------------------------------------

    with np.load(
        FULL_PATH,
        allow_pickle=True,
    ) as full:

        labels_full = full[
            "labels"
        ].astype(np.int64)

        classes_full = full[
            "class_names"
        ].astype(str)

        filenames_full = full[
            "filenames"
        ].astype(str)

        full_scores = full[
            "p3_similarities"
        ].astype(np.float32)

    # ------------------------------------------------------------
    # Load Random-P3
    # ------------------------------------------------------------

    with np.load(
        RANDOM_PATH,
        allow_pickle=True,
    ) as random_data:

        labels_random = random_data[
            "labels"
        ].astype(np.int64)

        classes_random = random_data[
            "class_names"
        ].astype(str)

        filenames_random = random_data[
            "filenames"
        ].astype(str)

        random_scores = random_data[
            "p3_similarities"
        ].astype(np.float32)

    # ------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------

    if not np.array_equal(
        labels_full,
        labels_random,
    ):
        raise ValueError(
            "Validation labels differ."
        )

    if not np.array_equal(
        classes_full,
        classes_random,
    ):
        raise ValueError(
            "Class order differs."
        )

    if not np.array_equal(
        filenames_full,
        filenames_random,
    ):
        raise ValueError(
            "Validation filenames differ."
        )

    labels = labels_full

    # ------------------------------------------------------------
    # Predictions / margins
    # ------------------------------------------------------------

    full_pred = full_scores.argmax(
        axis=1
    )

    random_pred = random_scores.argmax(
        axis=1
    )

    full_margin = top1_top2_margin(
        full_scores
    )

    random_margin = top1_top2_margin(
        random_scores
    )

    disagreement = (
        full_pred
        != random_pred
    )

    full_more_confident = (
        full_margin
        > random_margin
    )

    eligible = (
        disagreement
        & full_more_confident
    )

    # ------------------------------------------------------------
    # Random-P3 baseline
    # ------------------------------------------------------------

    baseline = calculate_metrics(
        labels,
        random_pred,
    )

    print(
        f"Images:                {len(labels)}"
    )

    print(
        f"Prediction disagreement: "
        f"{disagreement.sum()}"
    )

    print(
        f"Gate-eligible cases:     "
        f"{eligible.sum()}"
    )

    print()
    print(
        "RANDOM-P3 BASELINE"
    )
    print("-" * 94)

    print(
        f"Accuracy:    "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1:    "
        f"{baseline['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted-F1: "
        f"{baseline['weighted_f1'] * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Candidate thresholds
    #
    # Default is Random-P3.
    #
    # Switch to Full-P3 ONLY if:
    #
    # 1. predictions disagree
    # 2. Full margin > Random margin
    # 3. Random margin <= threshold
    #
    # ------------------------------------------------------------

    eligible_margins = np.unique(
        random_margin[
            eligible
        ]
    )

    candidates = [
        None
    ]

    candidates.extend(
        float(x)
        for x in eligible_margins
    )

    rows = []

    # ------------------------------------------------------------
    # No-gate baseline candidate
    # ------------------------------------------------------------

    baseline_row = {
        "threshold": None,
        "strategy": "random_p3_only",
        "num_switches": 0,
        "rescues": 0,
        "harms": 0,
        "net_correct": 0,
        "accuracy": baseline[
            "accuracy"
        ],
        "macro_f1": baseline[
            "macro_f1"
        ],
        "weighted_f1": baseline[
            "weighted_f1"
        ],
    }

    rows.append(
        baseline_row
    )

    # ------------------------------------------------------------
    # Threshold search
    # ------------------------------------------------------------

    for threshold in candidates[1:]:

        use_full = (
            eligible
            & (
                random_margin
                <= threshold
            )
        )

        predictions = (
            random_pred.copy()
        )

        predictions[
            use_full
        ] = full_pred[
            use_full
        ]

        result = calculate_metrics(
            labels,
            predictions,
        )

        random_correct = (
            random_pred
            == labels
        )

        full_correct = (
            full_pred
            == labels
        )

        rescues = int(
            (
                use_full
                & ~random_correct
                & full_correct
            ).sum()
        )

        harms = int(
            (
                use_full
                & random_correct
                & ~full_correct
            ).sum()
        )

        rows.append({
            "threshold": threshold,
            "strategy": (
                "confidence_gate"
            ),
            "num_switches": int(
                use_full.sum()
            ),
            "rescues": rescues,
            "harms": harms,
            "net_correct": (
                rescues - harms
            ),
            "accuracy": result[
                "accuracy"
            ],
            "macro_f1": result[
                "macro_f1"
            ],
            "weighted_f1": result[
                "weighted_f1"
            ],
        })

    # ------------------------------------------------------------
    # Selection:
    #
    # 1 Macro-F1
    # 2 Accuracy
    # 3 Weighted-F1
    # 4 fewer switches if otherwise tied
    # ------------------------------------------------------------

    selected = max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["weighted_f1"],
            -row["num_switches"],
        ),
    )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["weighted_f1"],
            -row["num_switches"],
        ),
        reverse=True,
    )

    # ------------------------------------------------------------
    # Print top 10
    # ------------------------------------------------------------

    print()
    print(
        "TOP VALIDATION GATES"
    )
    print("=" * 94)

    print(
        f"{'Rank':<6}"
        f"{'Threshold':>12}"
        f"{'Switch':>9}"
        f"{'Rescue':>9}"
        f"{'Harm':>8}"
        f"{'Net':>7}"
        f"{'Acc':>10}"
        f"{'Macro-F1':>12}"
    )

    print("-" * 94)

    for rank, row in enumerate(
        ranked[:10],
        start=1,
    ):
        if row[
            "threshold"
        ] is None:
            threshold_text = (
                "NONE"
            )
        else:
            threshold_text = (
                f"{row['threshold']:.6f}"
            )

        print(
            f"{rank:<6}"
            f"{threshold_text:>12}"
            f"{row['num_switches']:>9}"
            f"{row['rescues']:>9}"
            f"{row['harms']:>8}"
            f"{row['net_correct']:>+7}"
            f"{row['accuracy'] * 100:>9.2f}%"
            f"{row['macro_f1'] * 100:>11.2f}%"
        )

    # ------------------------------------------------------------
    # Selected result
    # ------------------------------------------------------------

    print()
    print("=" * 94)
    print(
        "SELECTED VALIDATION STRATEGY"
    )
    print("=" * 94)

    if selected[
        "threshold"
    ] is None:

        print(
            "Strategy: Random-P3 only"
        )

        print(
            "No confidence gate selected."
        )

    else:
        print(
            "Strategy: Confidence gate"
        )

        print(
            f"Frozen threshold: "
            f"{selected['threshold']:.8f}"
        )

    print(
        f"Switches:     "
        f"{selected['num_switches']}"
    )

    print(
        f"Rescues:      "
        f"{selected['rescues']}"
    )

    print(
        f"Harms:        "
        f"{selected['harms']}"
    )

    print(
        f"Net correct:  "
        f"{selected['net_correct']:+d}"
    )

    print()
    print(
        f"Accuracy:     "
        f"{selected['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1:     "
        f"{selected['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted-F1:  "
        f"{selected['weighted_f1'] * 100:.2f}%"
    )

    print()
    print(
        "DELTA VS RANDOM-P3"
    )
    print("-" * 94)

    print(
        f"Accuracy:    "
        f"{(selected['accuracy'] - baseline['accuracy']) * 100:+.2f} pp"
    )

    print(
        f"Macro-F1:    "
        f"{(selected['macro_f1'] - baseline['macro_f1']) * 100:+.2f} pp"
    )

    print(
        f"Weighted-F1: "
        f"{(selected['weighted_f1'] - baseline['weighted_f1']) * 100:+.2f} pp"
    )

    # ------------------------------------------------------------
    # Save complete threshold search
    # ------------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "threshold",
            "strategy",
            "num_switches",
            "rescues",
            "harms",
            "net_correct",
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )

    # ------------------------------------------------------------
    # Freeze selected validation configuration
    # ------------------------------------------------------------

    metadata = {
        "selection_split": (
            "validation"
        ),
        "gate_rule": (
            "default_random_p3;"
            "switch_to_full_p3_if_predictions_disagree"
            "_and_full_margin_gt_random_margin"
            "_and_random_margin_lte_threshold"
        ),
        "selection_rule": (
            "macro_f1_then_accuracy_"
            "then_weighted_f1_"
            "then_fewer_switches"
        ),
        "selected_strategy": (
            selected[
                "strategy"
            ]
        ),
        "selected_threshold": (
            selected[
                "threshold"
            ]
        ),
        "num_switches": (
            selected[
                "num_switches"
            ]
        ),
        "rescues": (
            selected[
                "rescues"
            ]
        ),
        "harms": (
            selected[
                "harms"
            ]
        ),
        "net_correct": (
            selected[
                "net_correct"
            ]
        ),
        "validation_accuracy": (
            selected[
                "accuracy"
            ]
        ),
        "validation_macro_f1": (
            selected[
                "macro_f1"
            ]
        ),
        "validation_weighted_f1": (
            selected[
                "weighted_f1"
            ]
        ),
        "baseline_random_p3_accuracy": (
            baseline[
                "accuracy"
            ]
        ),
        "baseline_random_p3_macro_f1": (
            baseline[
                "macro_f1"
            ]
        ),
        "baseline_random_p3_weighted_f1": (
            baseline[
                "weighted_f1"
            ]
        ),
        "num_validation_images": int(
            len(labels)
        ),
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
        f"Search CSV saved to: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Frozen gate saved to: "
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
