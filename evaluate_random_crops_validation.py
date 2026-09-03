from pathlib import Path
import math
import random

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder


# ============================================================
# Configuration
# ============================================================

FULL_DIR = Path(r"C:\Projects\stanford40_split\validation")
OUTPUT_DIR = Path(r"C:\Projects\stanford40_results\random_crops")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

NUM_CROPS = 10
TOP_K_CROPS = 3
CROP_SCALE = (0.50, 1.00)
CROP_RATIO = (0.75, 1.3333333333333333)
RANDOM_SEED = 42

EXPECTED_IMAGES = 1521
EXPECTED_CLASSES = 40


def normalise_relative_path(path):
    """Make relative paths portable inside saved result files."""
    return Path(path).as_posix()


def validate_dataset(dataset):
    """Check that the validation partition has the expected structure."""
    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} validation images, "
            f"but found {len(dataset)}."
        )

    if len(dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(dataset.classes)}."
        )

    print(
        f"Validated: {len(dataset)} images, "
        f"{len(dataset.classes)} classes"
    )


def create_text_features(model, tokenizer, class_names, device):
    """Create normalized OpenCLIP features for the Phase 1 basic prompts."""
    readable_class_names = [
        class_name.replace("_", " ") for class_name in class_names
    ]
    prompts = [
        f"a photo of a person {class_name}"
        for class_name in readable_class_names
    ]

    text_tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    return text_features, prompts


def deterministic_random_resized_crop(image, image_index, crop_index):
    """Generate one reproducible, unbiased random resized crop."""
    width, height = image.size
    area = height * width
    log_ratio = (math.log(CROP_RATIO[0]), math.log(CROP_RATIO[1]))

    rng = random.Random(
        RANDOM_SEED + image_index * NUM_CROPS + crop_index
    )

    for _ in range(10):
        target_area = area * rng.uniform(CROP_SCALE[0], CROP_SCALE[1])
        aspect_ratio = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))

        crop_width = round(math.sqrt(target_area * aspect_ratio))
        crop_height = round(math.sqrt(target_area / aspect_ratio))

        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)
            return image.crop(
                (left, top, left + crop_width, top + crop_height)
            )

    # Deterministic centre-crop fallback for extreme image shapes.
    input_ratio = width / height
    if input_ratio < CROP_RATIO[0]:
        crop_width = width
        crop_height = round(crop_width / CROP_RATIO[0])
    elif input_ratio > CROP_RATIO[1]:
        crop_height = height
        crop_width = round(crop_height * CROP_RATIO[1])
    else:
        crop_width = width
        crop_height = height

    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def calculate_metrics(labels, similarities):
    predictions = similarities.argmax(axis=1)
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:    {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%")


def evaluate_random_crops(
    dataset,
    preprocess,
    model,
    text_features,
    device,
):
    """Extract scores for all crops and construct four crop strategies."""
    all_crop_similarities = []
    all_labels = []
    best_confidence_indices = []
    best_margin_indices = []

    logit_scale = model.logit_scale.exp().detach()

    print(
        f"\nEvaluating {NUM_CROPS} deterministic random crops "
        "per validation image..."
    )

    with torch.inference_mode():
        for image_index, (image_path, label) in enumerate(dataset.samples):
            with Image.open(image_path) as loaded_image:
                image = loaded_image.convert("RGB")
                crops = [
                    deterministic_random_resized_crop(
                        image, image_index, crop_index
                    )
                    for crop_index in range(NUM_CROPS)
                ]

            crop_batch = torch.stack(
                [preprocess(crop) for crop in crops]
            ).to(device)

            image_features = model.encode_image(crop_batch)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            similarities = image_features @ text_features.T
            probabilities = (logit_scale * similarities).softmax(dim=1)

            confidence = probabilities.max(dim=1).values
            top_two = probabilities.topk(k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]

            all_crop_similarities.append(
                similarities.cpu().numpy().astype(np.float32)
            )
            all_labels.append(label)
            best_confidence_indices.append(int(confidence.argmax().item()))
            best_margin_indices.append(int(margin.argmax().item()))

            processed = image_index + 1
            if processed % 50 == 0 or processed == len(dataset):
                print(f"Processed {processed}/{len(dataset)} images")

    crop_similarities = np.stack(all_crop_similarities, axis=0)
    labels = np.asarray(all_labels, dtype=np.int64)
    best_confidence_indices = np.asarray(
        best_confidence_indices, dtype=np.int64
    )
    best_margin_indices = np.asarray(best_margin_indices, dtype=np.int64)
    row_indices = np.arange(len(dataset))

    best_confidence_scores = crop_similarities[
        row_indices, best_confidence_indices
    ]
    best_margin_scores = crop_similarities[
        row_indices, best_margin_indices
    ]
    mean_scores = crop_similarities.mean(axis=1)

    # Rank crops by their largest cosine similarity and average the top k.
    crop_peak_scores = crop_similarities.max(axis=2)
    top_k_indices = np.argsort(crop_peak_scores, axis=1)[:, -TOP_K_CROPS:]
    top_k_scores = np.take_along_axis(
        crop_similarities,
        top_k_indices[:, :, None],
        axis=1,
    ).mean(axis=1)

    strategies = {
        "best_confidence": best_confidence_scores,
        "best_margin": best_margin_scores,
        "mean_all": mean_scores,
        "mean_top_k": top_k_scores,
    }

    metrics = {
        name: calculate_metrics(labels, scores)
        for name, scores in strategies.items()
    }

    return {
        "crop_similarities": crop_similarities,
        "labels": labels,
        "best_confidence_indices": best_confidence_indices,
        "best_margin_indices": best_margin_indices,
        "top_k_indices": top_k_indices,
        "strategies": strategies,
        "metrics": metrics,
    }


def select_strategy(metrics):
    """Select by macro F1, then accuracy, then weighted F1."""
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["macro_f1"],
            metrics[name]["accuracy"],
            metrics[name]["weighted_f1"],
        ),
    )


def main():
    if not FULL_DIR.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {FULL_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device:             {device}")
    print(f"Model:              {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED}")
    print(f"Random crops:       {NUM_CROPS}")
    print(f"Top-k crops:        {TOP_K_CROPS}")
    print(f"Crop scale:         {CROP_SCALE}")
    print(f"Random seed:        {RANDOM_SEED}")

    print("\nLoading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    # No transform here: random crops must be generated from the original PIL image.
    dataset = ImageFolder(FULL_DIR)
    validate_dataset(dataset)

    text_features, prompts = create_text_features(
        model,
        tokenizer,
        dataset.classes,
        device,
    )

    filenames = [
        normalise_relative_path(Path(path).relative_to(FULL_DIR))
        for path, _ in dataset.samples
    ]

    results = evaluate_random_crops(
        dataset,
        preprocess,
        model,
        text_features,
        device,
    )

    print("\n" + "=" * 64)
    print("RANDOM-CROP VALIDATION RESULTS")
    print("=" * 64)
    for strategy_name, strategy_metrics in results["metrics"].items():
        print_metrics(strategy_name, strategy_metrics)

    selected_strategy = select_strategy(results["metrics"])
    selected_metrics = results["metrics"][selected_strategy]
    selected_scores = results["strategies"][selected_strategy]

    output_path = OUTPUT_DIR / "validation_random_crop_similarities.npz"
    np.savez_compressed(
        output_path,
        crop_similarities=results["crop_similarities"],
        labels=results["labels"],
        filenames=np.asarray(filenames),
        class_names=np.asarray(dataset.classes),
        prompts=np.asarray(prompts),
        best_confidence_indices=results["best_confidence_indices"],
        best_margin_indices=results["best_margin_indices"],
        top_k_indices=results["top_k_indices"],
        best_confidence_similarities=results["strategies"]["best_confidence"],
        best_margin_similarities=results["strategies"]["best_margin"],
        mean_all_similarities=results["strategies"]["mean_all"],
        mean_top_k_similarities=results["strategies"]["mean_top_k"],
        selected_similarities=selected_scores,
        selected_predictions=selected_metrics["predictions"],
        selected_strategy=np.asarray(selected_strategy),
        selected_accuracy=np.asarray(selected_metrics["accuracy"]),
        selected_macro_f1=np.asarray(selected_metrics["macro_f1"]),
        selected_weighted_f1=np.asarray(selected_metrics["weighted_f1"]),
        model_name=np.asarray(MODEL_NAME),
        pretrained=np.asarray(PRETRAINED),
        num_crops=np.asarray(NUM_CROPS),
        top_k_crops=np.asarray(TOP_K_CROPS),
        crop_scale=np.asarray(CROP_SCALE),
        crop_ratio=np.asarray(CROP_RATIO),
        random_seed=np.asarray(RANDOM_SEED),
    )

    print("\n" + "=" * 64)
    print(f"Selected validation strategy: {selected_strategy}")
    print(
        f"Selected Macro F1:           "
        f"{selected_metrics['macro_f1'] * 100:.2f}%"
    )
    print(f"Results saved to:            {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
