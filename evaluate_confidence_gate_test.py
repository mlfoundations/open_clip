from pathlib import Path
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
    / "test_contextual_prompt_results.npz"
)

RANDOM_PATH = (
    RESULTS_ROOT
    / "context_prompts"
    / "test_contextual_random_same_crop.npz"
)

GATE_PATH = (
    RESULTS_ROOT
    / "context_prompts"
    / "selected_confidence_gate.json"
)

OUTPUT_NPZ = (
    RESULTS_ROOT
    / "context_prompts"
    / "test_confidence_gate_results.npz"
)

OUTPUT_REPORT = (
    RESULTS_ROOT
    / "context_prompts"
    / "test_confidence_gate_report.txt"
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

    top_two.sort(axis=1)

    return (
        top_two[:, 1]
        - top_two[:, 0]
    )


def main():
    print()
    print(
        "FROZEN CONFIDENCE-GATED FUSION TEST"
    )
    print("=" * 92)

    # ------------------------------------------------------------
    # Frozen validation-selected gate
    # ------------------------------------------------------------

    gate = json.loads(
        GATE_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        gate["selection_split"]
        != "validation"
    ):
        raise ValueError(
            "Gate was not selected "
            "on validation."
        )

    if (
        gate["selected_strategy"]
        != "confidence_gate"
    ):
        raise ValueError(
            "Validation did not select "
            "the confidence gate."
        )

    threshold = float(
        gate["selected_threshold"]
    )

    print(
        f"Frozen threshold: {threshold:.8f}"
    )

    # ------------------------------------------------------------
    # Full-P3 test scores
    # ------------------------------------------------------------

    with np.load(
        FULL_PATH,
        allow_pickle=True,
    ) as full:

        full_labels = full[
            "labels"
        ].astype(np.int64)

        full_classes = full[
            "class_names"
        ].astype(str)

        full_filenames = full[
            "filenames"
        ].astype(str)

        full_scores = full[
            "p3_similarities"
        ].astype(np.float32)

    # ------------------------------------------------------------
    # Random-P3 test scores
    # ------------------------------------------------------------

    with np.load(
        RANDOM_PATH,
        allow_pickle=True,
    ) as random_data:

        random_labels = random_data[
            "labels"
        ].astype(np.int64)

        random_classes = random_data[
            "class_names"
        ].astype(str)

        random_filenames = random_data[
            "filenames"
        ].astype(str)

        random_scores = random_data[
            "p3_similarities"
        ].astype(np.float32)

    # ------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------

    if not np.array_equal(
        full_labels,
        random_labels,
    ):
        raise ValueError(
            "Test labels differ."
        )

    if not np.array_equal(
        full_classes,
        random_classes,
    ):
        raise ValueError(
            "Test class order differs."
        )

    if not np.array_equal(
        full_filenames,
        random_filenames,
    ):
        raise ValueError(
            "Test filename order differs."
        )

    labels = full_labels

    print(
        f"Images:           {len(labels)}"
    )

    print(
        f"Classes:          {len(full_classes)}"
    )

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
        full_pred != random_pred
    )

    full_more_confident = (
        full_margin
        > random_margin
    )

    use_full = (
        disagreement
        & full_more_confident
        & (
            random_margin
            <= threshold
        )
    )

    gated_pred = (
        random_pred.copy()
    )

    gated_pred[
        use_full
    ] = full_pred[
        use_full
    ]

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    random_result = calculate_metrics(
        labels,
        random_pred,
    )

    gated_result = calculate_metrics(
        labels,
        gated_pred,
    )

    random_correct = (
        random_pred == labels
    )

    full_correct = (
        full_pred == labels
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

    neutral_switches = int(
        use_full.sum()
        - rescues
        - harms
    )

    accuracy_delta = (
        gated_result["accuracy"]
        - random_result["accuracy"]
    ) * 100

    macro_delta = (
        gated_result["macro_f1"]
        - random_result["macro_f1"]
    ) * 100

    weighted_delta = (
        gated_result["weighted_f1"]
        - random_result["weighted_f1"]
    ) * 100

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    lines = [
        "",
        "FROZEN CONFIDENCE-GATED FUSION TEST RESULT",
        "=" * 92,
        f"Images: {len(labels)}",
        f"Frozen threshold: {threshold:.8f}",
        (
            "Default view: Random-P3; "
            "fallback view: Full-P3"
        ),
        "",
        "TEST PERFORMANCE",
        "-" * 92,
        (
            f"{'Method':<30}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
        (
            f"{'Random-P3':<30}"
            f"{random_result['accuracy'] * 100:>13.2f}%"
            f"{random_result['macro_f1'] * 100:>13.2f}%"
            f"{random_result['weighted_f1'] * 100:>15.2f}%"
        ),
        (
            f"{'Confidence gate':<30}"
            f"{gated_result['accuracy'] * 100:>13.2f}%"
            f"{gated_result['macro_f1'] * 100:>13.2f}%"
            f"{gated_result['weighted_f1'] * 100:>15.2f}%"
        ),
        "",
        "GATE BEHAVIOUR",
        "-" * 92,
        f"Prediction disagreements: {int(disagreement.sum())}",
        f"Switched to Full-P3:       {int(use_full.sum())}",
        f"Rescues:                   {rescues}",
        f"Harms:                     {harms}",
        f"Neutral switches:           {neutral_switches}",
        f"Net correct:                {rescues - harms:+d}",
        "",
        "TEST DELTAS VS RANDOM-P3",
        "-" * 92,
        f"Accuracy:    {accuracy_delta:+.2f} pp",
        f"Macro-F1:    {macro_delta:+.2f} pp",
        f"Weighted-F1: {weighted_delta:+.2f} pp",
    ]

    report = "\n".join(
        lines
    )

    print(report)

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels,
        filenames=full_filenames,
        class_names=full_classes,

        full_p3_similarities=(
            full_scores
        ),

        random_p3_similarities=(
            random_scores
        ),

        full_predictions=(
            full_pred
        ),

        random_predictions=(
            random_pred
        ),

        gated_predictions=(
            gated_pred
        ),

        full_margin=(
            full_margin
        ),

        random_margin=(
            random_margin
        ),

        use_full=(
            use_full
        ),

        threshold=np.asarray(
            threshold
        ),

        random_accuracy=np.asarray(
            random_result[
                "accuracy"
            ]
        ),

        gated_accuracy=np.asarray(
            gated_result[
                "accuracy"
            ]
        ),

        random_macro_f1=np.asarray(
            random_result[
                "macro_f1"
            ]
        ),

        gated_macro_f1=np.asarray(
            gated_result[
                "macro_f1"
            ]
        ),

        random_weighted_f1=np.asarray(
            random_result[
                "weighted_f1"
            ]
        ),

        gated_weighted_f1=np.asarray(
            gated_result[
                "weighted_f1"
            ]
        ),
    )

    print()
    print(
        f"NPZ saved to:    {OUTPUT_NPZ}"
    )

    print(
        f"Report saved to: {OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
