from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


TEST_DIR = Path(r"C:\Projects\stanford40_split\test")
RESULTS_DIR = Path(r"C:\Projects\open_clip\action_aware_results")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 16


# More descriptive prompts for difficult action classes
ACTION_PROMPTS = {
    "applauding":
        "a photo of a person clapping both hands together",
    "cutting vegetables":
        "a photo of a person using a knife to cut vegetables on a board",
    "phoning":
        "a photo of a person holding a phone next to their ear",
    "pouring liquid":
        "a photo of a person tilting a container and pouring liquid into another container",
    "smoking":
        "a photo of a person holding a cigarette near their mouth",
    "texting message":
        "a photo of a person looking at and typing on a mobile phone with their hands",
    "washing dishes":
        "a photo of a person washing plates or dishes at a kitchen sink",
    "waving hands":
        "a photo of a person raising an open hand and waving",
    "writing on a book":
        "a photo of a person holding a pen and writing on a book or notebook",
}


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    dataset = ImageFolder(TEST_DIR, transform=preprocess)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    class_names = [
        folder_name.replace("_", " ")
        for folder_name in dataset.classes
    ]

    prompts = [
        ACTION_PROMPTS.get(
            class_name,
            f"a photo of a person {class_name}",
        )
        for class_name in class_names
    ]

    print(f"Classes found: {len(class_names)}")
    print(f"Test images found: {len(dataset)}")
    print(f"Action-aware classes: {len(ACTION_PROMPTS)}")

    print("\nAction-aware prompts:")
    for class_name in ACTION_PROMPTS:
        print(f"- {class_name}: {ACTION_PROMPTS[class_name]}")

    print("\nCreating text features...")

    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    true_labels = []
    predicted_labels = []

    start_time = time.perf_counter()

    print("Evaluating images...")

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarities = image_features @ text_features.T
            predictions = similarities.argmax(dim=1)

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )

            if batch_number % 10 == 0:
                processed = min(
                    batch_number * BATCH_SIZE,
                    len(dataset),
                )
                print(f"Processed {processed}/{len(dataset)} images")

    elapsed_time = time.perf_counter() - start_time

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )

    class_totals = matrix.sum(axis=1)
    class_correct = matrix.diagonal()

    per_class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros_like(class_correct, dtype=float),
        where=class_totals != 0,
    )

    per_class_df = pd.DataFrame({
        "class": class_names,
        "correct": class_correct,
        "total": class_totals,
        "accuracy_percent": per_class_accuracy * 100,
    })

    per_class_df = per_class_df.sort_values(
        "accuracy_percent",
        ascending=False,
    )

    per_class_df.to_csv(
        RESULTS_DIR / "per_class_accuracy.csv",
        index=False,
    )

    summary_df = pd.DataFrame([{
        "experiment": "action-aware prompts",
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "test_images": len(dataset),
        "accuracy_percent": accuracy * 100,
        "macro_precision_percent": macro_precision * 100,
        "macro_recall_percent": macro_recall * 100,
        "macro_f1_percent": macro_f1 * 100,
        "weighted_precision_percent": weighted_precision * 100,
        "weighted_recall_percent": weighted_recall * 100,
        "weighted_f1_percent": weighted_f1 * 100,
        "evaluation_seconds": elapsed_time,
        "seconds_per_image": elapsed_time / len(dataset),
    }])

    summary_df.to_csv(
        RESULTS_DIR / "action_aware_summary.csv",
        index=False,
    )

    prompt_df = pd.DataFrame({
        "class": class_names,
        "prompt": prompts,
        "prompt_type": [
            "action-aware"
            if class_name in ACTION_PROMPTS
            else "basic"
            for class_name in class_names
        ],
    })

    prompt_df.to_csv(
        RESULTS_DIR / "prompts_used.csv",
        index=False,
    )

    prediction_df = pd.DataFrame({
        "image_path": [
            str(path)
            for path, _ in dataset.samples
        ],
        "true_class": [
            class_names[index]
            for index in true_labels
        ],
        "predicted_class": [
            class_names[index]
            for index in predicted_labels
        ],
        "correct": (
            np.array(true_labels)
            == np.array(predicted_labels)
        ),
    })

    prediction_df.to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    plt.figure(figsize=(22, 18))

    sns.heatmap(
        matrix,
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=False,
    )

    plt.title("CLIP Action-Aware Prompt Confusion Matrix")
    plt.xlabel("Predicted action")
    plt.ylabel("True action")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nAction-aware evaluation completed.")
    print(f"Top-1 accuracy:       {accuracy * 100:.2f}%")
    print(f"Macro precision:      {macro_precision * 100:.2f}%")
    print(f"Macro recall:         {macro_recall * 100:.2f}%")
    print(f"Macro F1-score:       {macro_f1 * 100:.2f}%")
    print(f"Weighted precision:   {weighted_precision * 100:.2f}%")
    print(f"Weighted recall:      {weighted_recall * 100:.2f}%")
    print(f"Weighted F1-score:    {weighted_f1 * 100:.2f}%")
    print(f"Evaluation time:      {elapsed_time:.2f} seconds")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()