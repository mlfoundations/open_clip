from pathlib import Path
import csv

import numpy as np

from compare_dual_fusion_classes import calculate_class_metrics


FULL_FILE = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
    r"\test_full_similarities.npz"
)
RANDOM_FILE = Path(
    r"C:\Projects\stanford40_results\random_crops"
    r"\test_random_crop_similarities.npz"
)
OUTPUT_DIR = Path(r"C:\Projects\stanford40_results\random_crops")
CSV_FILE = OUTPUT_DIR / "random_crop_class_comparison.csv"
TEXT_FILE = OUTPUT_DIR / "random_crop_class_comparison.txt"

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40


def main():
    if not FULL_FILE.exists():
        raise FileNotFoundError(FULL_FILE)
    if not RANDOM_FILE.exists():
        raise FileNotFoundError(RANDOM_FILE)

    with np.load(FULL_FILE, allow_pickle=True) as full_data, np.load(
        RANDOM_FILE, allow_pickle=True
    ) as random_data:
        labels = full_data["labels"].astype(np.int64)
        random_labels = random_data["labels"].astype(np.int64)
        full_predictions = full_data["similarities"].argmax(axis=1)
        random_predictions = random_data["selected_predictions"].astype(
            np.int64
        )
        class_names = full_data["class_names"].astype(str)
        random_classes = random_data["class_names"].astype(str)
        full_filenames = full_data["filenames"].astype(str)
        random_filenames = random_data["filenames"].astype(str)
        strategy = str(random_data["frozen_strategy"].item())

    if len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, found {len(labels)}."
        )
    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(class_names)}."
        )
    if strategy != "best_confidence":
        raise ValueError(f"Unexpected frozen strategy: {strategy}")
    if not np.array_equal(labels, random_labels):
        raise ValueError("Full and random-crop labels are not aligned.")
    if not np.array_equal(class_names, random_classes):
        raise ValueError("Full and random-crop classes are not aligned.")
    if not np.array_equal(full_filenames, random_filenames):
        raise ValueError("Full and random-crop filenames are not aligned.")

    full_metrics = calculate_class_metrics(labels, full_predictions)
    random_metrics = calculate_class_metrics(labels, random_predictions)
    rows = []

    for class_index, class_name in enumerate(class_names):
        class_mask = labels == class_index
        full_only_correct = int(
            np.sum(
                class_mask
                & (full_predictions == class_index)
                & (random_predictions != class_index)
            )
        )
        random_only_correct = int(
            np.sum(
                class_mask
                & (full_predictions != class_index)
                & (random_predictions == class_index)
            )
        )

        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": int(full_metrics["support"][class_index]),
                "full_correct": int(full_metrics["correct"][class_index]),
                "random_correct": int(
                    random_metrics["correct"][class_index]
                ),
                "full_only_correct": full_only_correct,
                "random_only_correct": random_only_correct,
                "net_correct_change": (
                    random_only_correct - full_only_correct
                ),
                "full_precision": float(
                    full_metrics["precision"][class_index]
                ),
                "random_precision": float(
                    random_metrics["precision"][class_index]
                ),
                "precision_change": float(
                    random_metrics["precision"][class_index]
                    - full_metrics["precision"][class_index]
                ),
                "full_recall": float(full_metrics["recall"][class_index]),
                "random_recall": float(
                    random_metrics["recall"][class_index]
                ),
                "recall_change": float(
                    random_metrics["recall"][class_index]
                    - full_metrics["recall"][class_index]
                ),
                "full_f1": float(full_metrics["f1"][class_index]),
                "random_f1": float(random_metrics["f1"][class_index]),
                "f1_change": float(
                    random_metrics["f1"][class_index]
                    - full_metrics["f1"][class_index]
                ),
            }
        )

    sorted_rows = sorted(
        rows, key=lambda row: row["f1_change"], reverse=True
    )

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)

    improved = [row for row in rows if row["f1_change"] > 1e-12]
    declined = [row for row in rows if row["f1_change"] < -1e-12]
    unchanged = [
        row for row in rows if abs(row["f1_change"]) <= 1e-12
    ]

    lines = [
        "RANDOM-CROP CLASS-WISE COMPARISON",
        "=" * 100,
        f"Frozen strategy: {strategy}",
        (
            f"Improved classes: {len(improved)} | "
            f"Declined classes: {len(declined)} | "
            f"Unchanged classes: {len(unchanged)}"
        ),
        "",
        "TOP 10 F1 IMPROVEMENTS",
        "-" * 100,
    ]

    for rank, row in enumerate(sorted_rows[:10], start=1):
        lines.append(
            f"{rank:02d}. {row['class_name']:<32} "
            f"Full: {row['full_f1'] * 100:6.2f}% | "
            f"Random: {row['random_f1'] * 100:6.2f}% | "
            f"Change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    lines.extend(["", "F1 DECLINES", "-" * 100])
    for rank, row in enumerate(sorted(declined, key=lambda row: row["f1_change"]), start=1):
        lines.append(
            f"{rank:02d}. {row['class_name']:<32} "
            f"Full: {row['full_f1'] * 100:6.2f}% | "
            f"Random: {row['random_f1'] * 100:6.2f}% | "
            f"Change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    lines.extend(["", "ALL CLASSES", "-" * 100])
    for row in sorted_rows:
        lines.append(
            f"{row['class_name']:<32} "
            f"F1 change: {row['f1_change'] * 100:+6.2f} pp | "
            f"Recall change: {row['recall_change'] * 100:+6.2f} pp | "
            f"Net correct: {row['net_correct_change']:+d}"
        )

    report = "\n".join(lines)
    TEXT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nCSV saved to:    {CSV_FILE}")
    print(f"Report saved to: {TEXT_FILE}")


if __name__ == "__main__":
    main()
