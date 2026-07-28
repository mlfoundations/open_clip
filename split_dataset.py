from pathlib import Path
import random
import shutil

source_dir = Path(r"C:\Projects\train_FUll")
output_dir = Path(r"C:\Projects\stanford40_split")

random.seed(42)

for class_folder in sorted(source_dir.iterdir()):
    if not class_folder.is_dir():
        continue

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    random.shuffle(images)
    split_point = int(len(images) * 0.8)

    groups = {
        "train": images[:split_point],
        "test": images[split_point:]
    }

    for group_name, group_images in groups.items():
        destination = output_dir / group_name / class_folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image in group_images:
            shutil.copy2(image, destination / image.name)

    print(
        f"{class_folder.name}: "
        f"{len(groups['train'])} train, "
        f"{len(groups['test'])} test"
    )

print("\nDataset split completed successfully.")
print(f"Saved at: {output_dir}")