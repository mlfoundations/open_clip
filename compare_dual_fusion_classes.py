from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


RESULTS_DIR = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
)

FULL_FILE = RESULTS_DIR / "test_full_similarities.npz"
FUSION_FILE = RESULTS_DIR / "test_fixed_fusion_results.npz"

CSV_FILE = RESULTS_DIR / "dual_fusion_class_comparison.csv"
TEXT_FILE = RESULTS_DIR / "dual_fusion_class_comparison.txt"

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40


def calculate_class_metrics(labels, predictions):
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=np.arange(EXPECTED_CLASSES),
            zero_division=0,
        )
    )

    correct = np.asarray([
        np.sum(
            (labels == class_index)
            & (predictions == class_index)
        )
        for class_index in range(EXPECTED_CLASSES)
    ])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "correct": correct,
    }


def main():
    if not FULL_FILE.exists():
        raise FileNotFoundError(FULL_FILE)

    if not FUSION_FILE.exists():
        raise FileNotFoundError(FUSION_FILE)

    full_data = np.load(FULL_FILE, allow_pickle=True)
    fusion_data = np.load(FUSION_FILE, allow_pickle=True)

    labels = full_data["labels"].astype(np.int64)
    fusion_labels = fusion_data["labels"].astype(np.int64)

    full_predictions = (
        full_data["similarities"]
        .argmax(axis=1)
        .astype(np.int64)
    )

    fusion_predictions = (
        fusion_data["predictions"]
        .astype(np.int64)
    )

    class_names = [
        str(name)
        for name in full_data["class_names"]
    ]

    if len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(labels)}."
        )

    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(class_names)}."
        )

    if not np.array_equal(labels, fusion_labels):
        raise ValueError(
            "Full-view and fusion labels are not aligned."
        )

    if (
        "filenames" in full_data.files
        and "filenames" in fusion_data.files
        and not np.array_equal(
            full_data["filenames"],
            fusion_data["filenames"],
        )
    ):
        raise ValueError(
            "Full-view and fusion filenames are not aligned."
        )

    full_metrics = calculate_class_metrics(
        labels,
        full_predictions,
    )

    fusion_metrics = calculate_class_metrics(
        labels,
        fusion_predictions,
    )

    rows = []

    for class_index, class_name in enumerate(class_names):
        full_only_correct = int(np.sum(
            (labels == class_index)
            & (full_predictions == class_index)
            & (fusion_predictions != class_index)
        ))

        fusion_only_correct = int(np.sum(
            (labels == class_index)
            & (full_predictions != class_index)
            & (fusion_predictions == class_index)
        ))

        net_correct_change = (
            fusion_only_correct - full_only_correct
        )

        row = {
            "class_index": class_index,
            "class_name": class_name,
            "support": int(
                full_metrics["support"][class_index]
            ),
            "full_correct": int(
                full_metrics["correct"][class_index]
            ),
            "fusion_correct": int(
                fusion_metrics["correct"][class_index]
            ),
            "net_correct_change": net_correct_change,
            "full_only_correct": full_only_correct,
            "fusion_only_correct": fusion_only_correct,
            "full_precision": float(
                full_metrics["precision"][class_index]
            ),
            "fusion_precision": float(
                fusion_metrics["precision"][class_index]
            ),
            "precision_change": float(
                fusion_metrics["precision"][class_index]
                - full_metrics["precision"][class_index]
            ),
            "full_recall": float(
                full_metrics["recall"][class_index]
            ),
            "fusion_recall": float(
                fusion_metrics["recall"][class_index]
            ),
            "recall_change": float(
                fusion_metrics["recall"][class_index]
                - full_metrics["recall"][class_index]
            ),
            "full_f1": float(
                full_metrics["f1"][class_index]
            ),
            "fusion_f1": float(
                fusion_metrics["f1"][class_index]
            ),
            "f1_change": float(
                fusion_metrics["f1"][class_index]
                - full_metrics["f1"][class_index]
            ),
        }

        rows.append(row)

    sorted_rows = sorted(
        rows,
        key=lambda item: item["f1_change"],
        reverse=True,
    )

    fieldnames = list(rows[0].keys())

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    improved = [
        row for row in rows
        if row["f1_change"] > 1e-12
    ]

    declined = [
        row for row in rows
        if row["f1_change"] < -1e-12
    ]

    unchanged = [
        row for row in rows
        if abs(row["f1_change"]) <= 1e-12
    ]

    lines = [
        "DUAL-FUSION CLASS-WISE COMPARISON",
        "=" * 88,
        (
            f"Improved classes: {len(improved)} | "
            f"Declined classes: {len(declined)} | "
            f"Unchanged classes: {len(unchanged)}"
        ),
        "",
        "TOP 10 F1 IMPROVEMENTS",
        "-" * 88,
    ]

    for rank, row in enumerate(sorted_rows[:10], start=1):
        lines.append(
            f"{rank:02d}. {row['class_name']:<32} "
            f"Full: {row['full_f1'] * 100:6.2f}% | "
            f"Fusion: {row['fusion_f1'] * 100:6.2f}% | "
            f"Change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    lines.extend([
        "",
        "TOP 10 F1 DECLINES",
        "-" * 88,
    ])

    for rank, row in enumerate(
        sorted_rows[-10:][::-1],
        start=1,
    ):
        lines.append(
            f"{rank:02d}. {row['class_name']:<32} "
            f"Full: {row['full_f1'] * 100:6.2f}% | "
            f"Fusion: {row['fusion_f1'] * 100:6.2f}% | "
            f"Change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    lines.extend([
        "",
        "ALL CLASSES",
        "-" * 88,
    ])

    for row in sorted_rows:
        lines.append(
            f"{row['class_name']:<32} "
            f"F1 change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Recall change: "
            f"{row['recall_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    report = "\n".join(lines)

    with TEXT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(report)
    print(f"\nCSV saved to:    {CSV_FILE}")
    print(f"Report saved to: {TEXT_FILE}")


if __name__ == "__main__":
    main()