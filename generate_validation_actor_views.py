from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


VALIDATION_DIR = Path(
    r"C:\Projects\stanford40_split\validation"
)

ANNOTATION_DIR = Path(
    r"C:\Projects\Stanford40\XMLAnnotations"
)

OUTPUT_DIR = Path(
    r"C:\Projects\stanford40_views\validation_actor_20"
)

CROP_MARGIN = 0.20
EXPECTED_IMAGES = 1521
EXPECTED_CLASSES = 40
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_bounding_box(xml_path):
    """Read the first human bounding box from a Stanford-40 XML file."""

    root = ET.parse(xml_path).getroot()
    bounding_box = root.find(".//object/bndbox")

    if bounding_box is None:
        raise ValueError(
            f"No bounding box found in: {xml_path}"
        )

    xmin = int(float(bounding_box.findtext("xmin")))
    ymin = int(float(bounding_box.findtext("ymin")))
    xmax = int(float(bounding_box.findtext("xmax")))
    ymax = int(float(bounding_box.findtext("ymax")))

    return xmin, ymin, xmax, ymax


def expand_bounding_box(
    box,
    image_width,
    image_height,
    margin,
):
    """Expand a bounding box while keeping it inside the image."""

    xmin, ymin, xmax, ymax = box

    box_width = xmax - xmin
    box_height = ymax - ymin

    horizontal_margin = int(box_width * margin)
    vertical_margin = int(box_height * margin)

    expanded_xmin = max(
        0,
        xmin - horizontal_margin,
    )
    expanded_ymin = max(
        0,
        ymin - vertical_margin,
    )
    expanded_xmax = min(
        image_width,
        xmax + horizontal_margin,
    )
    expanded_ymax = min(
        image_height,
        ymax + vertical_margin,
    )

    return (
        expanded_xmin,
        expanded_ymin,
        expanded_xmax,
        expanded_ymax,
    )


def main():
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {VALIDATION_DIR}"
        )

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(
            f"Annotation folder not found: {ANNOTATION_DIR}"
        )

    class_directories = sorted([
        directory
        for directory in VALIDATION_DIR.iterdir()
        if directory.is_dir()
    ])

    if len(class_directories) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} validation classes, "
            f"but found {len(class_directories)}."
        )

    # Start clean so old crops cannot contaminate the experiment.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_images = 0
    failed_images = []

    print(f"Validation folder: {VALIDATION_DIR}")
    print(f"Annotation folder: {ANNOTATION_DIR}")
    print(f"Output folder:     {OUTPUT_DIR}")
    print(f"Crop margin:       {CROP_MARGIN:.0%}")
    print()

    for class_directory in class_directories:
        output_class_directory = (
            OUTPUT_DIR / class_directory.name
        )
        output_class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_paths = sorted([
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        class_processed = 0

        for image_path in image_paths:
            xml_path = (
                ANNOTATION_DIR / f"{image_path.stem}.xml"
            )

            if not xml_path.exists():
                failed_images.append(
                    (
                        str(image_path),
                        f"XML not found: {xml_path}",
                    )
                )
                continue

            try:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")

                width, height = image.size

                original_box = read_bounding_box(xml_path)

                # Keep the original coordinates inside the image.
                xmin, ymin, xmax, ymax = original_box

                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(
                    xmin + 1,
                    min(xmax, width),
                )
                ymax = max(
                    ymin + 1,
                    min(ymax, height),
                )

                original_box = (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )

                expanded_box = expand_bounding_box(
                    original_box,
                    width,
                    height,
                    CROP_MARGIN,
                )

                actor_image = image.crop(expanded_box)

                output_path = (
                    output_class_directory / image_path.name
                )

                actor_image.save(
                    output_path,
                    quality=95,
                )

                processed_images += 1
                class_processed += 1

            except Exception as error:
                failed_images.append(
                    (
                        str(image_path),
                        str(error),
                    )
                )

        print(
            f"{class_directory.name:<30}"
            f"{class_processed:>4} actor crops"
        )

    print("\n" + "=" * 64)
    print("VALIDATION ACTOR-VIEW GENERATION")
    print("=" * 64)
    print(f"Actor crops generated: {processed_images}")
    print(f"Failed images:         {len(failed_images)}")
    print(f"Output classes:        {len(class_directories)}")
    print(f"Output folder:         {OUTPUT_DIR}")

    if failed_images:
        print("\nFailures:")

        for image_path, reason in failed_images:
            print(f"- {image_path}")
            print(f"  Reason: {reason}")

        raise RuntimeError(
            "Some validation actor crops could not be generated."
        )

    if processed_images != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} actor crops, "
            f"but generated {processed_images}."
        )

    output_class_count = len([
        directory
        for directory in OUTPUT_DIR.iterdir()
        if directory.is_dir()
    ])

    if output_class_count != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} output classes, "
            f"but found {output_class_count}."
        )

    print("\nAll validation actor crops were generated successfully.")


if __name__ == "__main__":
    main()