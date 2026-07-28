from pathlib import Path
import json

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


RESULTS_DIR = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
)

FULL_FILE = RESULTS_DIR / "test_full_similarities.npz"
FUSION_FILE = RESULTS_DIR / "test_fixed_fusion_results.npz"

OUTPUT_FILE = RESULTS_DIR / "dual_fusion_significance.json"

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40

BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42


def bootstrap_confidence_intervals(
    labels,
    full_predictions,
    fusion_predictions,
):
    """Paired bootstrap confidence intervals for metric differences."""

    rng = np.random.default_rng(RANDOM_SEED)

    number_of_images = len(labels)
    class_labels = np.arange(EXPECTED_CLASSES)

    accuracy_differences = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    macro_f1_differences = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    print(
        f"\nRunning {BOOTSTRAP_SAMPLES:,} paired bootstrap samples..."
    )

    for bootstrap_index in range(BOOTSTRAP_SAMPLES):
        sampled_indices = rng.integers(
            0,
            number_of_images,
            size=number_of_images,
        )

        sampled_labels = labels[sampled_indices]
        sampled_full = full_predictions[sampled_indices]
        sampled_fusion = fusion_predictions[sampled_indices]

        full_accuracy = accuracy_score(
            sampled_labels,
            sampled_full,
        )

        fusion_accuracy = accuracy_score(
            sampled_labels,
            sampled_fusion,
        )

        full_macro_f1 = f1_score(
            sampled_labels,
            sampled_full,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )

        fusion_macro_f1 = f1_score(
            sampled_labels,
            sampled_fusion,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )

        accuracy_differences[bootstrap_index] = (
            fusion_accuracy - full_accuracy
        )

        macro_f1_differences[bootstrap_index] = (
            fusion_macro_f1 - full_macro_f1
        )

        if (bootstrap_index + 1) % 1000 == 0:
            print(
                f"Completed "
                f"{bootstrap_index + 1:,}/"
                f"{BOOTSTRAP_SAMPLES:,}"
            )

    accuracy_ci = np.percentile(
        accuracy_differences,
        [2.5, 97.5],
    )

    macro_f1_ci = np.percentile(
        macro_f1_differences,
        [2.5, 97.5],
    )

    return {
        "accuracy_ci_lower": float(accuracy_ci[0]),
        "accuracy_ci_upper": float(accuracy_ci[1]),
        "macro_f1_ci_lower": float(macro_f1_ci[0]),
        "macro_f1_ci_upper": float(macro_f1_ci[1]),
        "probability_accuracy_improvement": float(
            np.mean(accuracy_differences > 0)
        ),
        "probability_macro_f1_improvement": float(
            np.mean(macro_f1_differences > 0)
        ),
    }


def main():
    if not FULL_FILE.exists():
        raise FileNotFoundError(
            f"Full-view result file not found: {FULL_FILE}"
        )

    if not FUSION_FILE.exists():
        raise FileNotFoundError(
            f"Fusion result file not found: {FUSION_FILE}"
        )

    full_data = np.load(
        FULL_FILE,
        allow_pickle=True,
    )

    fusion_data = np.load(
        FUSION_FILE,
        allow_pickle=True,
    )

    labels = full_data["labels"].astype(np.int64)

    full_predictions = (
        full_data["similarities"].argmax(axis=1).astype(np.int64)
    )

    fusion_labels = fusion_data["labels"].astype(np.int64)
    fusion_predictions = (
        fusion_data["predictions"].astype(np.int64)
    )

    if len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} test images, "
            f"but found {len(labels)}."
        )

    if not np.array_equal(labels, fusion_labels):
        raise ValueError(
            "Full-view and fusion label arrays are not aligned."
        )

    if "filenames" in full_data.files:
        if "filenames" not in fusion_data.files:
            raise KeyError(
                "Fusion result file does not contain filenames."
            )

        if not np.array_equal(
            full_data["filenames"],
            fusion_data["filenames"],
        ):
            raise ValueError(
                "Full-view and fusion filenames are not aligned."
            )

    full_correct = full_predictions == labels
    fusion_correct = fusion_predictions == labels

    both_correct = int(
        np.sum(full_correct & fusion_correct)
    )

    full_only_correct = int(
        np.sum(full_correct & ~fusion_correct)
    )

    fusion_only_correct = int(
        np.sum(~full_correct & fusion_correct)
    )

    both_wrong = int(
        np.sum(~full_correct & ~fusion_correct)
    )

    discordant_total = (
        full_only_correct + fusion_only_correct
    )

    if discordant_total == 0:
        mcnemar_p_value = 1.0
    else:
        mcnemar_result = binomtest(
            k=min(
                full_only_correct,
                fusion_only_correct,
            ),
            n=discordant_total,
            p=0.5,
            alternative="two-sided",
        )

        mcnemar_p_value = float(
            mcnemar_result.pvalue
        )

    full_accuracy = accuracy_score(
        labels,
        full_predictions,
    )

    fusion_accuracy = accuracy_score(
        labels,
        fusion_predictions,
    )

    full_macro_f1 = f1_score(
        labels,
        full_predictions,
        labels=np.arange(EXPECTED_CLASSES),
        average="macro",
        zero_division=0,
    )

    fusion_macro_f1 = f1_score(
        labels,
        fusion_predictions,
        labels=np.arange(EXPECTED_CLASSES),
        average="macro",
        zero_division=0,
    )

    accuracy_difference = (
        fusion_accuracy - full_accuracy
    )

    macro_f1_difference = (
        fusion_macro_f1 - full_macro_f1
    )

    bootstrap_results = bootstrap_confidence_intervals(
        labels,
        full_predictions,
        fusion_predictions,
    )

    statistically_significant_mcnemar = (
        mcnemar_p_value < 0.05
    )

    accuracy_ci_excludes_zero = (
        bootstrap_results["accuracy_ci_lower"] > 0
        or bootstrap_results["accuracy_ci_upper"] < 0
    )

    macro_f1_ci_excludes_zero = (
        bootstrap_results["macro_f1_ci_lower"] > 0
        or bootstrap_results["macro_f1_ci_upper"] < 0
    )

    results = {
        "test_images": EXPECTED_IMAGES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "random_seed": RANDOM_SEED,
        "full_accuracy": float(full_accuracy),
        "fusion_accuracy": float(fusion_accuracy),
        "accuracy_difference": float(
            accuracy_difference
        ),
        "full_macro_f1": float(full_macro_f1),
        "fusion_macro_f1": float(fusion_macro_f1),
        "macro_f1_difference": float(
            macro_f1_difference
        ),
        "both_correct": both_correct,
        "full_only_correct": full_only_correct,
        "fusion_only_correct": fusion_only_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p_value": mcnemar_p_value,
        "mcnemar_significant_at_0.05": (
            statistically_significant_mcnemar
        ),
        **bootstrap_results,
        "accuracy_ci_excludes_zero": (
            accuracy_ci_excludes_zero
        ),
        "macro_f1_ci_excludes_zero": (
            macro_f1_ci_excludes_zero
        ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("DUAL-FUSION STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 72)

    print(
        f"Full-view accuracy:       "
        f"{full_accuracy * 100:.2f}%"
    )

    print(
        f"Dual-fusion accuracy:     "
        f"{fusion_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy difference:      "
        f"{accuracy_difference * 100:+.2f} percentage points"
    )

    print(
        f"Accuracy difference 95% CI: "
        f"[{bootstrap_results['accuracy_ci_lower'] * 100:+.2f}, "
        f"{bootstrap_results['accuracy_ci_upper'] * 100:+.2f}] points"
    )

    print()

    print(
        f"Full-view Macro F1:       "
        f"{full_macro_f1 * 100:.2f}%"
    )

    print(
        f"Dual-fusion Macro F1:     "
        f"{fusion_macro_f1 * 100:.2f}%"
    )

    print(
        f"Macro F1 difference:      "
        f"{macro_f1_difference * 100:+.2f} percentage points"
    )

    print(
        f"Macro F1 difference 95% CI: "
        f"[{bootstrap_results['macro_f1_ci_lower'] * 100:+.2f}, "
        f"{bootstrap_results['macro_f1_ci_upper'] * 100:+.2f}] points"
    )

    print("\nPAIRED PREDICTION TABLE")
    print("-" * 72)
    print(f"Both correct:              {both_correct}")
    print(f"Full only correct:         {full_only_correct}")
    print(f"Fusion only correct:       {fusion_only_correct}")
    print(f"Both wrong:                {both_wrong}")

    print("\nMcNemar exact test")
    print("-" * 72)
    print(f"Exact p-value:             {mcnemar_p_value:.6f}")
    print(
        "Significant at p < 0.05:  "
        f"{statistically_significant_mcnemar}"
    )

    print("\nINTERPRETATION")
    print("-" * 72)

    if statistically_significant_mcnemar:
        print(
            "McNemar's test indicates a statistically significant "
            "difference in prediction correctness."
        )
    else:
        print(
            "McNemar's test does not indicate a statistically "
            "significant difference at the 0.05 level."
        )

    if accuracy_ci_excludes_zero:
        print(
            "The bootstrap accuracy confidence interval excludes zero."
        )
    else:
        print(
            "The bootstrap accuracy confidence interval includes zero."
        )

    if macro_f1_ci_excludes_zero:
        print(
            "The bootstrap Macro F1 confidence interval excludes zero."
        )
    else:
        print(
            "The bootstrap Macro F1 confidence interval includes zero."
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()