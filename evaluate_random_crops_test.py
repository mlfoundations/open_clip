from pathlib import Path

import numpy as np
import open_clip
import torch
from torchvision.datasets import ImageFolder

import evaluate_random_crops_validation as random_crop


TEST_DIR = Path(r"C:\Projects\stanford40_split\test")
VALIDATION_RESULTS = Path(
    r"C:\Projects\stanford40_results\random_crops"
    r"\validation_random_crop_similarities.npz"
)
OUTPUT_DIR = Path(r"C:\Projects\stanford40_results\random_crops")
OUTPUT_PATH = OUTPUT_DIR / "test_random_crop_similarities.npz"

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40


def load_frozen_validation_configuration():
    if not VALIDATION_RESULTS.exists():
        raise FileNotFoundError(
            f"Validation result file not found: {VALIDATION_RESULTS}"
        )

    with np.load(VALIDATION_RESULTS) as validation:
        frozen = {
            "strategy": str(validation["selected_strategy"].item()),
            "class_names": validation["class_names"].astype(str),
            "prompts": validation["prompts"].astype(str),
            "model_name": str(validation["model_name"].item()),
            "pretrained": str(validation["pretrained"].item()),
            "num_crops": int(validation["num_crops"].item()),
            "top_k_crops": int(validation["top_k_crops"].item()),
            "crop_scale": tuple(validation["crop_scale"].tolist()),
            "crop_ratio": tuple(validation["crop_ratio"].tolist()),
            "random_seed": int(validation["random_seed"].item()),
        }

    expected = {
        "strategy": "best_confidence",
        "model_name": random_crop.MODEL_NAME,
        "pretrained": random_crop.PRETRAINED,
        "num_crops": random_crop.NUM_CROPS,
        "top_k_crops": random_crop.TOP_K_CROPS,
        "crop_scale": random_crop.CROP_SCALE,
        "crop_ratio": random_crop.CROP_RATIO,
        "random_seed": random_crop.RANDOM_SEED,
    }

    for key, expected_value in expected.items():
        if frozen[key] != expected_value:
            raise ValueError(
                f"Frozen validation configuration mismatch for {key}: "
                f"saved={frozen[key]!r}, code={expected_value!r}"
            )

    return frozen


def validate_test_dataset(dataset, frozen):
    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} test images, "
            f"but found {len(dataset)}."
        )

    if len(dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(dataset.classes)}."
        )

    if not np.array_equal(
        np.asarray(dataset.classes), frozen["class_names"]
    ):
        raise ValueError(
            "Test class order does not match the frozen validation class order."
        )

    print(
        f"Validated: {len(dataset)} test images, "
        f"{len(dataset.classes)} classes"
    )


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    frozen = load_frozen_validation_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device:                    {device}")
    print(f"Model:                     {frozen['model_name']}")
    print(f"Pretrained weights:        {frozen['pretrained']}")
    print(f"Frozen strategy:           {frozen['strategy']}")
    print(f"Frozen random crops:       {frozen['num_crops']}")
    print(f"Frozen top-k crops:        {frozen['top_k_crops']}")
    print(f"Frozen crop scale:         {frozen['crop_scale']}")
    print(f"Frozen random seed:        {frozen['random_seed']}")

    print("\nLoading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        frozen["model_name"],
        pretrained=frozen["pretrained"],
    )
    tokenizer = open_clip.get_tokenizer(frozen["model_name"])
    model = model.to(device)
    model.eval()

    dataset = ImageFolder(TEST_DIR)
    validate_test_dataset(dataset, frozen)

    text_features, prompts = random_crop.create_text_features(
        model,
        tokenizer,
        dataset.classes,
        device,
    )

    if not np.array_equal(np.asarray(prompts), frozen["prompts"]):
        raise ValueError(
            "Generated test prompts do not match frozen validation prompts."
        )

    filenames = [
        random_crop.normalise_relative_path(
            Path(path).relative_to(TEST_DIR)
        )
        for path, _ in dataset.samples
    ]

    results = random_crop.evaluate_random_crops(
        dataset,
        preprocess,
        model,
        text_features,
        device,
    )

    strategy = frozen["strategy"]
    selected_scores = results["strategies"][strategy]
    selected_metrics = results["metrics"][strategy]

    np.savez_compressed(
        OUTPUT_PATH,
        crop_similarities=results["crop_similarities"],
        labels=results["labels"],
        filenames=np.asarray(filenames),
        class_names=np.asarray(dataset.classes),
        prompts=np.asarray(prompts),
        best_confidence_indices=results["best_confidence_indices"],
        best_confidence_similarities=results["strategies"][
            "best_confidence"
        ],
        selected_similarities=selected_scores,
        selected_predictions=selected_metrics["predictions"],
        frozen_strategy=np.asarray(strategy),
        accuracy=np.asarray(selected_metrics["accuracy"]),
        macro_f1=np.asarray(selected_metrics["macro_f1"]),
        weighted_f1=np.asarray(selected_metrics["weighted_f1"]),
        model_name=np.asarray(frozen["model_name"]),
        pretrained=np.asarray(frozen["pretrained"]),
        num_crops=np.asarray(frozen["num_crops"]),
        crop_scale=np.asarray(frozen["crop_scale"]),
        crop_ratio=np.asarray(frozen["crop_ratio"]),
        random_seed=np.asarray(frozen["random_seed"]),
    )

    print("\n" + "=" * 64)
    print("FROZEN RANDOM-CROP TEST RESULT")
    print("=" * 64)
    print(f"Strategy:    {strategy}")
    print(f"Accuracy:    {selected_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:    {selected_metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1: {selected_metrics['weighted_f1'] * 100:.2f}%")
    print(f"Scores saved to: {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
