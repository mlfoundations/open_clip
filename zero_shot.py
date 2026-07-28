from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Test dataset created by split_dataset.py
TEST_DIR = Path(r"C:\Projects\stanford40_split\test")

# Start with a smaller standard CLIP model
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 16


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

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
        class_name.replace("_", " ")
        for class_name in dataset.classes
    ]

    # Basic baseline prompt
    prompts = [
        f"a photo of a person {class_name}"
        for class_name in class_names
    ]

    print(f"Classes found: {len(class_names)}")
    print(f"Test images found: {len(dataset)}")
    print("Creating text features...")

    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    print("Evaluating images...")

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarities = image_features @ text_features.T
            predictions = similarities.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_number % 10 == 0:
                print(f"Processed {total}/{len(dataset)} images")

    accuracy = 100 * correct / total

    print("\nZero-shot evaluation completed.")
    print(f"Correct predictions: {correct}")
    print(f"Total test images: {total}")
    print(f"Top-1 accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()