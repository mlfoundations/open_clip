from pathlib import Path

import numpy as np
import open_clip
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# ============================================================
# Configuration
# ============================================================

FULL_DIR = Path(
    r"C:\Projects\stanford40_split\test"
)

ACTOR_DIR = Path(
    r"C:\Projects\stanford40_views\actor_20"
)

OUTPUT_DIR = Path(
    r"C:\Projects\stanford40_results\fixed_fusion"
)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 16
NUM_WORKERS = 0

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40


def normalise_relative_path(path):
    """Make relative paths comparable across datasets."""
    return Path(path).as_posix()


def validate_datasets(full_dataset, actor_dataset):
    """Confirm exact alignment between full and actor views."""

    if len(full_dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} full validation images, "
            f"but found {len(full_dataset)}."
        )

    if len(actor_dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} actor validation images, "
            f"but found {len(actor_dataset)}."
        )

    if len(full_dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(full_dataset.classes)}."
        )

    if full_dataset.classes != actor_dataset.classes:
        raise ValueError(
            "Full and actor class ordering does not match."
        )

    if full_dataset.class_to_idx != actor_dataset.class_to_idx:
        raise ValueError(
            "Full and actor class-to-index mappings do not match."
        )

    full_samples = [
        (
            normalise_relative_path(
                Path(path).relative_to(FULL_DIR)
            ),
            label,
        )
        for path, label in full_dataset.samples
    ]

    actor_samples = [
        (
            normalise_relative_path(
                Path(path).relative_to(ACTOR_DIR)
            ),
            label,
        )
        for path, label in actor_dataset.samples
    ]

    if full_samples != actor_samples:
        for index, (full_item, actor_item) in enumerate(
            zip(full_samples, actor_samples)
        ):
            if full_item != actor_item:
                raise ValueError(
                    "Dataset alignment failed at index "
                    f"{index}:\n"
                    f"Full:  {full_item}\n"
                    f"Actor: {actor_item}"
                )

        raise ValueError(
            "Full and actor validation datasets are not aligned."
        )

    print(
        f"Validated: {len(full_dataset)} aligned images, "
        f"{len(full_dataset.classes)} classes"
    )


def create_text_features(
    model,
    tokenizer,
    class_names,
    device,
):
    """Create normalised CLIP features for the basic prompts."""

    readable_class_names = [
        class_name.replace("_", " ")
        for class_name in class_names
    ]

    prompts = [
        f"a photo of a person {class_name}"
        for class_name in readable_class_names
    ]

    print("\nPrompts:")
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
    """Extract cosine-similarity scores for one view."""

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_similarities = []
    all_labels = []

    print(f"\nEvaluating {view_name} validation view...")

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

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

            if (
                batch_number % 10 == 0
                or processed == len(dataset)
            ):
                print(
                    f"{view_name}: processed "
                    f"{processed}/{len(dataset)}"
                )

    similarity_scores = np.concatenate(
        all_similarities,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = similarity_scores.argmax(axis=1)

    accuracy = accuracy_score(labels, predictions)
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

    print(f"\n{view_name} validation results")
    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")

    return {
        "similarities": similarity_scores,
        "labels": labels,
        "predictions": predictions,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_view_results(
    output_path,
    results,
    filenames,
    class_names,
    prompts,
):
    """Save scores and metadata for fixed fusion."""

    np.savez_compressed(
        output_path,
        similarities=results["similarities"],
        labels=results["labels"],
        predictions=results["predictions"],
        filenames=np.asarray(filenames),
        class_names=np.asarray(class_names),
        prompts=np.asarray(prompts),
        model_name=np.asarray(MODEL_NAME),
        pretrained=np.asarray(PRETRAINED),
        accuracy=np.asarray(results["accuracy"]),
        macro_f1=np.asarray(results["macro_f1"]),
        weighted_f1=np.asarray(results["weighted_f1"]),
    )


def main():
    if not FULL_DIR.exists():
        raise FileNotFoundError(
            f"Full validation folder not found: {FULL_DIR}"
        )

    if not ACTOR_DIR.exists():
        raise FileNotFoundError(
            f"Actor validation folder not found: {ACTOR_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device:             {device}")
    print(f"Model:              {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED}")
    print(f"Batch size:         {BATCH_SIZE}")

    print("\nLoading CLIP model...")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
        )
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(device)
    model.eval()

    full_dataset = ImageFolder(
        FULL_DIR,
        transform=preprocess,
    )

    actor_dataset = ImageFolder(
        ACTOR_DIR,
        transform=preprocess,
    )

    validate_datasets(
        full_dataset,
        actor_dataset,
    )

    text_features, prompts = create_text_features(
        model,
        tokenizer,
        full_dataset.classes,
        device,
    )

    filenames = [
        normalise_relative_path(
            Path(path).relative_to(FULL_DIR)
        )
        for path, _ in full_dataset.samples
    ]

    full_results = evaluate_view(
        "full",
        full_dataset,
        model,
        text_features,
        device,
    )

    actor_results = evaluate_view(
        "actor",
        actor_dataset,
        model,
        text_features,
        device,
    )

    if not np.array_equal(
        full_results["labels"],
        actor_results["labels"],
    ):
        raise ValueError(
            "Full and actor label arrays do not match."
        )

    full_output = (
        OUTPUT_DIR / "test_full_similarities.npz"
    )

    actor_output = (
        OUTPUT_DIR / "test_actor_similarities.npz"
    )

    save_view_results(
        full_output,
        full_results,
        filenames,
        full_dataset.classes,
        prompts,
    )

    save_view_results(
        actor_output,
        actor_results,
        filenames,
        full_dataset.classes,
        prompts,
    )

    print("\n" + "=" * 64)
    print("VALIDATION SCORE EXTRACTION COMPLETED")
    print("=" * 64)
    print(
        "Similarity matrix shape: "
        f"{full_results['similarities'].shape}"
    )
    print(f"Full scores saved to:  {full_output}")
    print(f"Actor scores saved to: {actor_output}")


if __name__ == "__main__":
    main()
