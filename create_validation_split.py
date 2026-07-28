from pathlib import Path
import csv
import random
import shutil


TRAIN_DIR = Path(r"C:\Projects\stanford40_split\train")
VALIDATION_DIR = Path(r"C:\Projects\stanford40_split\validation")

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    class_directories = sorted([
        directory
        for directory in TRAIN_DIR.iterdir()
        if directory.is_dir()
    ])

    if len(class_directories) != 40:
        raise ValueError(
            f"Expected 40 training classes, "
            f"but found {len(class_directories)}."
        )

    # Prevent accidental mixing with an earlier validation split.
    if VALIDATION_DIR.exists() and any(VALIDATION_DIR.iterdir()):
        raise FileExistsError(
            f"The validation directory already exists and is not empty:\n"
            f"{VALIDATION_DIR}\n"
            f"Check or rename it before running this script again."
        )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    manifest_rows = []

    total_training_images = 0
    total_validation_images = 0

    print(f"Training directory:   {TRAIN_DIR}")
    print(f"Validation directory: {VALIDATION_DIR}")
    print(f"Validation ratio:     {VALIDATION_RATIO:.0%}")
    print(f"Random seed:          {RANDOM_SEED}")
    print()

    for class_directory in class_directories:
        images = sorted([
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            raise ValueError(
                f"No images found in class: {class_directory.name}"
            )

        total_training_images += len(images)

        shuffled_images = images.copy()
        rng.shuffle(shuffled_images)

        validation_count = round(
            len(images) * VALIDATION_RATIO
        )

        validation_images = sorted(
            shuffled_images[:validation_count]
        )

        destination_directory = (
            VALIDATION_DIR / class_directory.name
        )
        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for source_path in validation_images:
            destination_path = (
                destination_directory / source_path.name
            )

            shutil.copy2(source_path, destination_path)

            manifest_rows.append({
                "class_name": class_directory.name,
                "filename": source_path.name,
                "source_path": str(source_path),
                "validation_path": str(destination_path),
            })

        total_validation_images += validation_count

        print(
            f"{class_directory.name:<30}"
            f"Train: {len(images):>3}  "
            f"Validation: {validation_count:>3}"
        )

    if total_training_images != 7611:
        raise ValueError(
            f"Expected 7,611 training images, "
            f"but found {total_training_images}."
        )

    manifest_path = (
        VALIDATION_DIR / "validation_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "filename",
                "source_path",
                "validation_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 60)
    print("VALIDATION SPLIT COMPLETED")
    print("=" * 60)
    print(f"Original training images: {total_training_images}")
    print(f"Validation images copied: {total_validation_images}")
    print(
        "Approximate remaining development-training images: "
        f"{total_training_images - total_validation_images}"
    )
    print(f"Validation classes:       {len(class_directories)}")
    print(f"Manifest saved to:        {manifest_path}")
    print("\nOriginal training images were not moved or deleted.")


if __name__ == "__main__":
    main()