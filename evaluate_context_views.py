from pathlib import Path
import csv
import json

import numpy as np
import open_clip
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# ============================================================
# Experiment configuration
# ============================================================

FULL_DIR = Path(r"C:\Projects\stanford40_split\test")
ACTOR_DIR = Path(r"C:\Projects\stanford40_views\actor_20")
CONTEXT_DIR = Path(r"C:\Projects\stanford40_views\context_masked")

OUTPUT_DIR = Path(r"C:\Projects\stanford40_results\context_analysis")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 16
NUM_WORKERS = 0

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40

VIEW_DIRECTORIES = {
    "full": FULL_DIR,
    "actor": ACTOR_DIR,
    "context": CONTEXT_DIR,
}


def normalise_relative_path(path):
    """Convert a relative path to a consistent platform-independent form."""
    return Path(path).as_posix()


def validate_dataset(reference_dataset, comparison_dataset, view_name):
    """
    Confirm that every view has exactly the same classes, class ordering,
    image count, labels and filenames.
    """

    if comparison_dataset.classes != reference_dataset.classes:
        raise ValueError(
            f"{view_name} class ordering does not match the full view.\n"
            f"Full classes: {reference_dataset.classes}\n"
            f"{view_name} classes: {comparison_dataset.classes}"
        )

    if comparison_dataset.class_to_idx != reference_dataset.class_to_idx:
        raise ValueError(
            f"{view_name} class-to-index mapping does not match the full view."
        )

    if len(comparison_dataset) != len(reference_dataset):
        raise ValueError(
            f"{view_name} contains {len(comparison_dataset)} images, "
            f"but the full view contains {len(reference_dataset)}."
        )

    reference_items = []
    comparison_items = []

    for path, label in reference_dataset.samples:
        relative_path = normalise_relative_path(
            Path(path).relative_to(reference_dataset.root)
        )
        reference_items.append((relative_path, label))

    for path, label in comparison_dataset.samples:
        relative_path = normalise_relative_path(
            Path(path).relative_to(comparison_dataset.root)
        )
        comparison_items.append((relative_path, label))

    if comparison_items != reference_items:
        for index, (reference_item, comparison_item) in enumerate(
            zip(reference_items, comparison_items)
        ):
            if reference_item != comparison_item:
                raise ValueError(
                    f"{view_name} image alignment mismatch at index {index}.\n"
                    f"Full: {reference_item}\n"
                    f"{view_name}: {comparison_item}"
                )

        raise ValueError(
            f"{view_name} filenames or labels do not match the full view."
        )

    print(
        f"Validated {view_name}: "
        f"{len(comparison_dataset)} aligned images, "
        f"{len(comparison_dataset.classes)} classes"
    )


def create_text_features(model, tokenizer, class_names, device):
    """Create one normalised CLIP text feature for each basic prompt."""

    readable_class_names = [
        class_name.replace("_", " ")
        for class_name in class_names
    ]

    prompts = [
        f"a photo of a person {class_name}"
        for class_name in readable_class_names
    ]

    print("\nBasic prompts:")
    for index, prompt in enumerate(prompts):
        print(f"{index:02d}: {prompt}")

    text_tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True,
        )

    return text_features, prompts


def evaluate_view(
    view_name,
    dataset,
    model,
    text_features,
    device,
):
    """Evaluate one visual view and return cosine-similarity scores."""

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_similarities = []
    all_labels = []

    print(f"\nEvaluating {view_name} view...")

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            similarities = image_features @ text_features.T

            all_similarities.append(
                similarities.cpu().numpy().astype(np.float32)
            )
            all_labels.append(
                labels.numpy().astype(np.int64)
            )

            processed = min(
                batch_number * BATCH_SIZE,
                len(dataset),
            )

            if batch_number % 10 == 0 or processed == len(dataset):
                print(
                    f"{view_name}: processed "
                    f"{processed}/{len(dataset)} images"
                )

    similarity_scores = np.concatenate(all_similarities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = similarity_scores.argmax(axis=1)

    return similarity_scores, labels, predictions


def calculate_metrics(labels, predictions, class_names):
    """Calculate overall and per-class evaluation metrics."""

    overall_accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    per_class_accuracy = {}
    per_class_correct = {}
    per_class_total = {}

    for class_index, class_name in enumerate(class_names):
        class_mask = labels == class_index
        class_total = int(class_mask.sum())
        class_correct = int(
            (predictions[class_mask] == class_index).sum()
        )

        if class_total > 0:
            class_accuracy = class_correct / class_total
        else:
            class_accuracy = 0.0

        per_class_accuracy[class_name] = class_accuracy
        per_class_correct[class_name] = class_correct
        per_class_total[class_name] = class_total

    return {
        "accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


def save_similarity_scores(
    output_path,
    similarities,
    labels,
    predictions,
    filenames,
    class_names,
    prompts,
):
    """
    Save raw cosine similarities for later fixed and adaptive fusion.

    Shape:
        similarities = [1921 images, 40 classes]
    """

    np.savez_compressed(
        output_path,
        similarities=similarities,
        labels=labels,
        predictions=predictions,
        filenames=np.asarray(filenames),
        class_names=np.asarray(class_names),
        prompts=np.asarray(prompts),
        model_name=np.asarray(MODEL_NAME),
        pretrained=np.asarray(PRETRAINED),
    )


def save_overall_results(metrics_by_view):
    output_path = OUTPUT_DIR / "overall_metrics.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "view",
            "accuracy",
            "accuracy_percent",
            "macro_f1",
            "macro_f1_percent",
            "weighted_f1",
            "weighted_f1_percent",
        ])

        for view_name in ["full", "actor", "context"]:
            metrics = metrics_by_view[view_name]

            writer.writerow([
                view_name,
                f"{metrics['accuracy']:.6f}",
                f"{metrics['accuracy'] * 100:.2f}",
                f"{metrics['macro_f1']:.6f}",
                f"{metrics['macro_f1'] * 100:.2f}",
                f"{metrics['weighted_f1']:.6f}",
                f"{metrics['weighted_f1'] * 100:.2f}",
            ])


def save_per_class_results(metrics_by_view, class_names):
    output_path = OUTPUT_DIR / "per_class_context_analysis.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "class_index",
            "class_name",
            "test_images",
            "full_correct",
            "full_accuracy",
            "full_accuracy_percent",
            "actor_correct",
            "actor_accuracy",
            "actor_accuracy_percent",
            "context_correct",
            "context_accuracy",
            "context_accuracy_percent",
            "context_reliance_score",
            "context_reliance_score_percentage_points",
            "person_importance_drop",
            "person_importance_drop_percentage_points",
        ])

        for class_index, class_name in enumerate(class_names):
            full_accuracy = (
                metrics_by_view["full"]
                ["per_class_accuracy"][class_name]
            )
            actor_accuracy = (
                metrics_by_view["actor"]
                ["per_class_accuracy"][class_name]
            )
            context_accuracy = (
                metrics_by_view["context"]
                ["per_class_accuracy"][class_name]
            )

            # CRS_c = context accuracy - actor accuracy
            context_reliance_score = (
                context_accuracy - actor_accuracy
            )

            # PID_c = full accuracy - context accuracy
            person_importance_drop = (
                full_accuracy - context_accuracy
            )

            writer.writerow([
                class_index,
                class_name,
                metrics_by_view["full"]
                ["per_class_total"][class_name],
                metrics_by_view["full"]
                ["per_class_correct"][class_name],
                f"{full_accuracy:.6f}",
                f"{full_accuracy * 100:.2f}",
                metrics_by_view["actor"]
                ["per_class_correct"][class_name],
                f"{actor_accuracy:.6f}",
                f"{actor_accuracy * 100:.2f}",
                metrics_by_view["context"]
                ["per_class_correct"][class_name],
                f"{context_accuracy:.6f}",
                f"{context_accuracy * 100:.2f}",
                f"{context_reliance_score:.6f}",
                f"{context_reliance_score * 100:.2f}",
                f"{person_importance_drop:.6f}",
                f"{person_importance_drop * 100:.2f}",
            ])


def save_experiment_information(
    class_names,
    prompts,
    filenames,
    metrics_by_view,
):
    information = {
        "model_name": MODEL_NAME,
        "pretrained_weights": PRETRAINED,
        "number_of_images": len(filenames),
        "number_of_classes": len(class_names),
        "batch_size": BATCH_SIZE,
        "class_ordering": class_names,
        "prompts": prompts,
        "view_directories": {
            name: str(path)
            for name, path in VIEW_DIRECTORIES.items()
        },
        "overall_metrics": {
            view_name: {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
            for view_name, metrics in metrics_by_view.items()
        },
        "metric_definitions": {
            "context_reliance_score": (
                "context per-class accuracy minus "
                "actor per-class accuracy"
            ),
            "person_importance_drop": (
                "full per-class accuracy minus "
                "context per-class accuracy"
            ),
        },
    }

    output_path = OUTPUT_DIR / "experiment_information.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(information, file, indent=4)


def print_results(metrics_by_view):
    print("\n" + "=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)

    print(
        f"{'View':<12}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>16}"
    )

    for view_name in ["full", "actor", "context"]:
        metrics = metrics_by_view[view_name]

        print(
            f"{view_name:<12}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['macro_f1'] * 100:>11.2f}%"
            f"{metrics['weighted_f1'] * 100:>15.2f}%"
        )

    overall_crs = (
        metrics_by_view["context"]["accuracy"]
        - metrics_by_view["actor"]["accuracy"]
    )

    overall_pid = (
        metrics_by_view["full"]["accuracy"]
        - metrics_by_view["context"]["accuracy"]
    )

    print("\nOverall diagnostic differences:")
    print(
        "Context Reliance Score "
        f"(context - actor): {overall_crs * 100:.2f} percentage points"
    )
    print(
        "Person Importance Drop "
        f"(full - context): {overall_pid * 100:.2f} percentage points"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for view_name, directory in VIEW_DIRECTORIES.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"{view_name} directory not found: {directory}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED}")
    print("Loading OpenCLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    datasets = {
        view_name: ImageFolder(directory, transform=preprocess)
        for view_name, directory in VIEW_DIRECTORIES.items()
    }

    full_dataset = datasets["full"]

    if len(full_dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} test images, "
            f"but found {len(full_dataset)}."
        )

    if len(full_dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(full_dataset.classes)}."
        )

    print("\nClass ordering:")
    for class_index, class_name in enumerate(full_dataset.classes):
        print(f"{class_index:02d}: {class_name}")

    validate_dataset(
        full_dataset,
        datasets["actor"],
        "actor",
    )
    validate_dataset(
        full_dataset,
        datasets["context"],
        "context",
    )

    class_names = full_dataset.classes

    filenames = [
        normalise_relative_path(
            Path(path).relative_to(full_dataset.root)
        )
        for path, _ in full_dataset.samples
    ]

    text_features, prompts = create_text_features(
        model,
        tokenizer,
        class_names,
        device,
    )

    metrics_by_view = {}
    reference_labels = None

    for view_name in ["full", "actor", "context"]:
        similarities, labels, predictions = evaluate_view(
            view_name,
            datasets[view_name],
            model,
            text_features,
            device,
        )

        if similarities.shape != (
            EXPECTED_IMAGES,
            EXPECTED_CLASSES,
        ):
            raise ValueError(
                f"Unexpected {view_name} similarity shape: "
                f"{similarities.shape}"
            )

        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(
                f"{view_name} labels do not match the full-view labels."
            )

        metrics_by_view[view_name] = calculate_metrics(
            labels,
            predictions,
            class_names,
        )

        similarity_output = (
            OUTPUT_DIR / f"{view_name}_similarities.npz"
        )

        save_similarity_scores(
            similarity_output,
            similarities,
            labels,
            predictions,
            filenames,
            class_names,
            prompts,
        )

        print(
            f"Saved {view_name} similarities to: "
            f"{similarity_output}"
        )

    save_overall_results(metrics_by_view)
    save_per_class_results(metrics_by_view, class_names)
    save_experiment_information(
        class_names,
        prompts,
        filenames,
        metrics_by_view,
    )

    print_results(metrics_by_view)

    print("\nFiles saved:")
    print(OUTPUT_DIR / "overall_metrics.csv")
    print(OUTPUT_DIR / "per_class_context_analysis.csv")
    print(OUTPUT_DIR / "experiment_information.json")
    print(OUTPUT_DIR / "full_similarities.npz")
    print(OUTPUT_DIR / "actor_similarities.npz")
    print(OUTPUT_DIR / "context_similarities.npz")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()