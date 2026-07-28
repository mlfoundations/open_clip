from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


# Existing Stanford-40 test split
TEST_DIR = Path(r"C:\Projects\stanford40_split\test")

# Official bounding-box annotations
ANNOTATION_DIR = Path(r"C:\Projects\Stanford40\XMLAnnotations")

# Generated experimental views
OUTPUT_ROOT = Path(r"C:\Projects\stanford40_views")
ACTOR_OUTPUT_DIR = OUTPUT_ROOT / "actor_20"
CONTEXT_OUTPUT_DIR = OUTPUT_ROOT / "context_masked"

# Expand actor crop by 20%
CROP_MARGIN = 0.20

# Neutral-grey colour used to hide the actor
MASK_COLOUR = (128, 128, 128)


def read_bounding_box(xml_path):
    """Read the first human bounding box from a Stanford-40 XML file."""

    root = ET.parse(xml_path).getroot()
    bounding_box = root.find(".//object/bndbox")

    if bounding_box is None:
        raise ValueError(f"No bounding box found in: {xml_path}")

    xmin = int(float(bounding_box.findtext("xmin")))
    ymin = int(float(bounding_box.findtext("ymin")))
    xmax = int(float(bounding_box.findtext("xmax")))
    ymax = int(float(bounding_box.findtext("ymax")))

    return xmin, ymin, xmax, ymax


def expand_bounding_box(box, image_width, image_height, margin):
    """Expand a bounding box while keeping it inside the image."""

    xmin, ymin, xmax, ymax = box

    box_width = xmax - xmin
    box_height = ymax - ymin

    horizontal_margin = int(box_width * margin)
    vertical_margin = int(box_height * margin)

    expanded_xmin = max(0, xmin - horizontal_margin)
    expanded_ymin = max(0, ymin - vertical_margin)
    expanded_xmax = min(image_width, xmax + horizontal_margin)
    expanded_ymax = min(image_height, ymax + vertical_margin)

    return expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(
            f"Annotation folder not found: {ANNOTATION_DIR}"
        )

    # Start clean so old generated images cannot contaminate the experiment
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    ACTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in TEST_DIR.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg"}
    )

    print(f"Test images found: {len(image_paths)}")
    print("Generating actor and context views...")

    processed = 0
    missing_annotations = []
    failed_images = []

    for image_path in image_paths:
        class_name = image_path.parent.name
        xml_path = ANNOTATION_DIR / f"{image_path.stem}.xml"

        if not xml_path.exists():
            missing_annotations.append(str(xml_path))
            continue

        actor_class_dir = ACTOR_OUTPUT_DIR / class_name
        context_class_dir = CONTEXT_OUTPUT_DIR / class_name

        actor_class_dir.mkdir(parents=True, exist_ok=True)
        context_class_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            width, height = image.size
            original_box = read_bounding_box(xml_path)

            # Keep coordinates inside the image
            xmin, ymin, xmax, ymax = original_box
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(xmin + 1, min(xmax, width))
            ymax = max(ymin + 1, min(ymax, height))
            original_box = xmin, ymin, xmax, ymax

            # Actor-focused crop retains nearby interacted objects
            expanded_box = expand_bounding_box(
                original_box,
                width,
                height,
                CROP_MARGIN,
            )

            actor_image = image.crop(expanded_box)

            # Context-only image hides the annotated actor region
            context_image = image.copy()
            drawing = ImageDraw.Draw(context_image)
            drawing.rectangle(original_box, fill=MASK_COLOUR)

            actor_image.save(
                actor_class_dir / image_path.name,
                quality=95,
            )
            context_image.save(
                context_class_dir / image_path.name,
                quality=95,
            )

            processed += 1

            if processed % 200 == 0:
                print(f"Processed {processed}/{len(image_paths)} images")

        except Exception as error:
            failed_images.append((str(image_path), str(error)))

    print("\nView generation completed.")
    print(f"Successfully processed: {processed}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Actor views saved to: {ACTOR_OUTPUT_DIR}")
    print(f"Context views saved to: {CONTEXT_OUTPUT_DIR}")

    if missing_annotations:
        print("\nFirst five missing annotations:")
        for path in missing_annotations[:5]:
            print(path)

    if failed_images:
        print("\nFirst five failures:")
        for image_path, error in failed_images[:5]:
            print(f"{image_path}: {error}")


if __name__ == "__main__":
    main()