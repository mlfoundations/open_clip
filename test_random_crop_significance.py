from pathlib import Path
import json

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from test_dual_fusion_significance import bootstrap_confidence_intervals


FULL_FILE = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
    r"\test_full_similarities.npz"
)
RANDOM_FILE = Path(
    r"C:\Projects\stanford40_results\random_crops"
    r"\test_random_crop_similarities.npz"
)
OUTPUT_FILE = Path(
    r"C:\Projects\stanford40_results\random_crops"
    r"\random_crop_significance.json"
)

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
        full_filenames = full_data["filenames"].astype(str)
        random_filenames = random_data["filenames"].astype(str)
        full_classes = full_data["class_names"].astype(str)
        random_classes = random_data["class_names"].astype(str)
        strategy = str(random_data["frozen_strategy"].item())

    if len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, found {len(labels)}."
        )
    if len(full_classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(full_classes)}."
        )
    if strategy != "best_confidence":
        raise ValueError(f"Unexpected frozen strategy: {strategy}")
    if not np.array_equal(labels, random_labels):
        raise ValueError("Full and random-crop labels are not aligned.")
    if not np.array_equal(full_filenames, random_filenames):
        raise ValueError("Full and random-crop filenames are not aligned.")
    if not np.array_equal(full_classes, random_classes):
        raise ValueError("Full and random-crop class orders are not aligned.")

    full_correct = full_predictions == labels
    random_correct = random_predictions == labels

    both_correct = int(np.sum(full_correct & random_correct))
    full_only_correct = int(np.sum(full_correct & ~random_correct))
    random_only_correct = int(np.sum(~full_correct & random_correct))
    both_wrong = int(np.sum(~full_correct & ~random_correct))

    discordant_total = full_only_correct + random_only_correct
    if discordant_total == 0:
        mcnemar_p_value = 1.0
    else:
        mcnemar_p_value = float(
            binomtest(
                k=min(full_only_correct, random_only_correct),
                n=discordant_total,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    class_labels = np.arange(EXPECTED_CLASSES)
    full_accuracy = accuracy_score(labels, full_predictions)
    random_accuracy = accuracy_score(labels, random_predictions)
    full_macro_f1 = f1_score(
        labels,
        full_predictions,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    random_macro_f1 = f1_score(
        labels,
        random_predictions,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )

    bootstrap = bootstrap_confidence_intervals(
        labels, full_predictions, random_predictions
    )

    results = {
        "test_images": EXPECTED_IMAGES,
        "comparison": "full_image_vs_frozen_best_confidence_random_crop",
        "frozen_strategy": strategy,
        "full_accuracy": float(full_accuracy),
        "random_crop_accuracy": float(random_accuracy),
        "accuracy_difference": float(random_accuracy - full_accuracy),
        "full_macro_f1": float(full_macro_f1),
        "random_crop_macro_f1": float(random_macro_f1),
        "macro_f1_difference": float(random_macro_f1 - full_macro_f1),
        "both_correct": both_correct,
        "full_only_correct": full_only_correct,
        "random_only_correct": random_only_correct,
        "both_wrong": both_wrong,
        "net_additional_correct": random_only_correct - full_only_correct,
        "mcnemar_exact_p_value": mcnemar_p_value,
        "mcnemar_significant_at_0.05": mcnemar_p_value < 0.05,
        **bootstrap,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 72)
    print("RANDOM-CROP PAIRED SIGNIFICANCE RESULTS")
    print("=" * 72)
    print(f"Frozen strategy:          {strategy}")
    print(f"Full accuracy:            {full_accuracy * 100:.2f}%")
    print(f"Random-crop accuracy:     {random_accuracy * 100:.2f}%")
    print(
        f"Accuracy difference:      "
        f"{(random_accuracy - full_accuracy) * 100:+.2f} pp"
    )
    print(
        "Accuracy difference CI:   "
        f"[{bootstrap['accuracy_ci_lower'] * 100:+.2f}, "
        f"{bootstrap['accuracy_ci_upper'] * 100:+.2f}] pp"
    )
    print(f"Full Macro F1:            {full_macro_f1 * 100:.2f}%")
    print(f"Random-crop Macro F1:     {random_macro_f1 * 100:.2f}%")
    print(
        f"Macro F1 difference:      "
        f"{(random_macro_f1 - full_macro_f1) * 100:+.2f} pp"
    )
    print(
        "Macro F1 difference CI:   "
        f"[{bootstrap['macro_f1_ci_lower'] * 100:+.2f}, "
        f"{bootstrap['macro_f1_ci_upper'] * 100:+.2f}] pp"
    )
    print("\nPAIRED PREDICTION TABLE")
    print("-" * 72)
    print(f"Both correct:             {both_correct}")
    print(f"Full only correct:        {full_only_correct}")
    print(f"Random crop only correct: {random_only_correct}")
    print(f"Both wrong:               {both_wrong}")
    print(
        f"Net additional correct:   "
        f"{random_only_correct - full_only_correct:+d}"
    )
    print(f"McNemar exact p-value:    {mcnemar_p_value:.10f}")
    print(f"Significant at p < 0.05:  {mcnemar_p_value < 0.05}")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
