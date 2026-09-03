from pathlib import Path
import csv
import json

import numpy as np
import open_clip
import torch

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)

from contextual_prompt_bank import (
    get_prompts,
    validate_prompt_bank,
)


# ============================================================
# Configuration
# ============================================================

ACTOR_DIR = Path(
    r"C:\Projects\stanford40_views\validation_actor_20"
)

RESULTS_ROOT = Path(
    r"C:\Projects\stanford40_results"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "pose_interaction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UPPER_BODY_FILE = (
    RESULTS_ROOT
    / "upper_body"
    / "validation_upper_body_p3_results.npz"
)

FULL_P3_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_prompt_results.npz"
)

RANDOM_P3_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "validation_contextual_random_same_crop.npz"
)

PROMPT_SELECTION_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "selected_contextual_prompt_strategy.json"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "validation_pose_interaction_p3_results.npz"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "validation_pose_interaction_crops.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "validation_pose_interaction_p3_report.txt"
)


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

EXPECTED_IMAGES = 1521
EXPECTED_CLASSES = 40

# Frozen before validation.
PERSON_SCORE_THRESHOLD = 0.70
CROP_PADDING = 0.25

# COCO keypoint indices:
#
# 0 nose
# 1 left_eye
# 2 right_eye
# 3 left_ear
# 4 right_ear
# 5 left_shoulder
# 6 right_shoulder
# 7 left_elbow
# 8 right_elbow
# 9 left_wrist
# 10 right_wrist
#
INTERACTION_KEYPOINTS = [
    0, 1, 2, 3, 4,
    5, 6,
    7, 8,
    9, 10,
]


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(labels, scores):
    predictions = scores.argmax(axis=1)

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# P3 text representation
# ============================================================

def encode_text(
    model,
    tokenizer,
    prompts,
    device,
):
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            )
        )

    return features


def create_p3_text_features(
    model,
    tokenizer,
    class_names,
    device,
):
    p0 = get_prompts(
        class_names,
        "p0",
    )

    p1 = get_prompts(
        class_names,
        "p1",
    )

    p2 = get_prompts(
        class_names,
        "p2",
    )

    t0 = encode_text(
        model,
        tokenizer,
        p0,
        device,
    )

    t1 = encode_text(
        model,
        tokenizer,
        p1,
        device,
    )

    t2 = encode_text(
        model,
        tokenizer,
        p2,
        device,
    )

    p3 = (
        t0 + t1 + t2
    ) / 3.0

    p3 = (
        p3
        / p3.norm(
            dim=-1,
            keepdim=True,
        )
    )

    return p3


# ============================================================
# Pose-guided crop
# ============================================================

def create_interaction_crop(
    image,
    detection,
):
    """
    Create crop around head + shoulder + elbow + wrist keypoints.

    If no sufficiently confident person is detected,
    return the entire actor crop unchanged.
    """

    width, height = image.size

    scores = detection["scores"]

    if len(scores) == 0:
        return (
            image.copy(),
            None,
            True,
            0.0,
        )

    best_index = int(
        scores.argmax().item()
    )

    person_score = float(
        scores[best_index].item()
    )

    if (
        person_score
        < PERSON_SCORE_THRESHOLD
    ):
        return (
            image.copy(),
            None,
            True,
            person_score,
        )

    keypoints = (
        detection["keypoints"][
            best_index
        ]
        .detach()
        .cpu()
        .numpy()
    )

    points = keypoints[
        INTERACTION_KEYPOINTS,
        :2,
    ]

    # Remove obviously invalid coordinates.
    valid = (
        np.isfinite(points).all(
            axis=1
        )
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
        & (points[:, 0] <= width)
        & (points[:, 1] <= height)
    )

    points = points[
        valid
    ]

    if len(points) < 4:
        return (
            image.copy(),
            None,
            True,
            person_score,
        )

    x_min = float(
        points[:, 0].min()
    )

    x_max = float(
        points[:, 0].max()
    )

    y_min = float(
        points[:, 1].min()
    )

    y_max = float(
        points[:, 1].max()
    )

    span_x = max(
        1.0,
        x_max - x_min,
    )

    span_y = max(
        1.0,
        y_max - y_min,
    )

    pad_x = (
        span_x
        * CROP_PADDING
    )

    pad_y = (
        span_y
        * CROP_PADDING
    )

    left = max(
        0,
        int(
            round(
                x_min - pad_x
            )
        ),
    )

    right = min(
        width,
        int(
            round(
                x_max + pad_x
            )
        ),
    )

    top = max(
        0,
        int(
            round(
                y_min - pad_y
            )
        ),
    )

    bottom = min(
        height,
        int(
            round(
                y_max + pad_y
            )
        ),
    )

    if (
        right <= left
        or bottom <= top
    ):
        return (
            image.copy(),
            None,
            True,
            person_score,
        )

    crop = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    box = (
        left,
        top,
        right,
        bottom,
    )

    return (
        crop,
        box,
        False,
        person_score,
    )


# ============================================================
# Main
# ============================================================

def main():
    print()
    print(
        "POSE-GUIDED INTERACTION P3 VALIDATION"
    )
    print("=" * 96)

    print(
        f"Person threshold: "
        f"{PERSON_SCORE_THRESHOLD}"
    )

    print(
        f"Interaction padding: "
        f"{CROP_PADDING}"
    )

    print(
        "Keypoints: head + shoulders "
        "+ elbows + wrists"
    )

    print(
        "Fallback: original actor_20 crop"
    )

    # --------------------------------------------------------
    # Verify frozen P3
    # --------------------------------------------------------

    selection = json.loads(
        PROMPT_SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        selection[
            "selection_split"
        ]
        != "validation"
    ):
        raise ValueError(
            "P3 was not validation-selected."
        )

    if (
        selection[
            "selected_strategy"
        ]
        != "p3"
    ):
        raise ValueError(
            "Expected frozen P3 strategy."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ImageFolder(
        ACTOR_DIR
    )

    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(dataset)}."
        )

    if (
        len(dataset.classes)
        != EXPECTED_CLASSES
    ):
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(dataset.classes)}."
        )

    validate_prompt_bank(
        dataset.classes
    )

    filenames = np.asarray([
        str(
            Path(path).relative_to(
                ACTOR_DIR
            )
        ).replace(
            "\\",
            "/",
        )
        for path, _
        in dataset.samples
    ])

    labels = np.asarray(
        [
            label
            for _, label
            in dataset.samples
        ],
        dtype=np.int64,
    )

    print(
        f"Images:  {len(dataset)}"
    )

    print(
        f"Classes: "
        f"{len(dataset.classes)}"
    )

    # --------------------------------------------------------
    # Load frozen reference scores
    # --------------------------------------------------------

    with np.load(
        FULL_P3_FILE,
        allow_pickle=True,
    ) as data:

        frozen_labels = data[
            "labels"
        ].astype(np.int64)

        frozen_classes = data[
            "class_names"
        ].astype(str)

        frozen_filenames = data[
            "filenames"
        ].astype(str)

        full_scores = data[
            "p3_similarities"
        ].astype(np.float32)

    with np.load(
        RANDOM_P3_FILE,
        allow_pickle=True,
    ) as data:

        random_scores = data[
            "p3_similarities"
        ].astype(np.float32)

    with np.load(
        UPPER_BODY_FILE,
        allow_pickle=True,
    ) as data:

        actor_scores = data[
            "actor_p3_similarities"
        ].astype(np.float32)

    if not np.array_equal(
        labels,
        frozen_labels,
    ):
        raise ValueError(
            "Actor labels differ from "
            "frozen validation labels."
        )

    if not np.array_equal(
        filenames,
        frozen_filenames,
    ):
        raise ValueError(
            "Actor filename order differs "
            "from frozen validation order."
        )

    if not np.array_equal(
        np.asarray(
            dataset.classes
        ),
        frozen_classes,
    ):
        raise ValueError(
            "Class order differs."
        )

    # --------------------------------------------------------
    # Device + OpenCLIP
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    clip_model, _, clip_preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
        )
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    clip_model.eval()

    p3_text = create_p3_text_features(
        clip_model,
        tokenizer,
        dataset.classes,
        device,
    )

    # --------------------------------------------------------
    # Keypoint R-CNN
    # --------------------------------------------------------

    print()
    print(
        "Loading COCO Keypoint R-CNN..."
    )

    pose_weights = (
        KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    pose_model = (
        keypointrcnn_resnet50_fpn(
            weights=pose_weights
        )
        .to(device)
    )

    pose_model.eval()

    # --------------------------------------------------------
    # Evaluate pose-guided interaction crops
    # --------------------------------------------------------

    similarity_rows = []
    metadata_rows = []

    fallback_count = 0

    print()
    print(
        "Generating and encoding "
        "pose-guided interaction crops..."
    )

    with torch.inference_mode():

        for image_index, (
            image_path,
            label,
        ) in enumerate(
            dataset.samples
        ):

            with Image.open(
                image_path
            ) as loaded:

                image = loaded.convert(
                    "RGB"
                )

                pose_input = (
                    to_tensor(
                        image
                    ).to(device)
                )

                detection = (
                    pose_model(
                        [pose_input]
                    )[0]
                )

                (
                    crop,
                    crop_box,
                    fallback,
                    person_score,
                ) = create_interaction_crop(
                    image,
                    detection,
                )

                if fallback:
                    fallback_count += 1

                clip_tensor = (
                    clip_preprocess(
                        crop
                    )
                    .unsqueeze(0)
                    .to(device)
                )

            image_feature = (
                clip_model.encode_image(
                    clip_tensor
                )
            )

            image_feature = (
                image_feature
                / image_feature.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            similarities = (
                image_feature
                @ p3_text.T
            )

            similarity_rows.append(
                similarities[
                    0
                ]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if crop_box is None:
                left = top = right = bottom = ""
            else:
                (
                    left,
                    top,
                    right,
                    bottom,
                ) = crop_box

            metadata_rows.append({
                "filename": (
                    filenames[
                        image_index
                    ]
                ),
                "true_class": (
                    dataset.classes[
                        label
                    ]
                ),
                "person_score": (
                    person_score
                ),
                "fallback_actor_crop": (
                    fallback
                ),
                "crop_left": left,
                "crop_top": top,
                "crop_right": right,
                "crop_bottom": bottom,
            })

            processed = (
                image_index + 1
            )

            if (
                processed % 50 == 0
                or processed
                == len(dataset)
            ):
                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(dataset)}"
                )

    pose_scores = np.stack(
        similarity_rows,
        axis=0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    full_result = calculate_metrics(
        labels,
        full_scores,
    )

    actor_result = calculate_metrics(
        labels,
        actor_scores,
    )

    pose_result = calculate_metrics(
        labels,
        pose_scores,
    )

    random_result = calculate_metrics(
        labels,
        random_scores,
    )

    # --------------------------------------------------------
    # Complementarity with Random-P3
    # --------------------------------------------------------

    pose_pred = pose_result[
        "predictions"
    ]

    random_pred = random_result[
        "predictions"
    ]

    pose_correct = (
        pose_pred == labels
    )

    random_correct = (
        random_pred == labels
    )

    pose_only = (
        pose_correct
        & ~random_correct
    )

    random_only = (
        ~pose_correct
        & random_correct
    )

    oracle = (
        pose_correct
        | random_correct
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = [
        "",
        (
            "POSE-GUIDED INTERACTION "
            "P3 VALIDATION RESULT"
        ),
        "=" * 96,
        f"Images: {len(labels)}",
        (
            f"Fallbacks to actor crop: "
            f"{fallback_count}"
        ),
        (
            f"Detection success: "
            f"{(1 - fallback_count / len(labels)) * 100:.2f}%"
        ),
        "",
        "VALIDATION PERFORMANCE",
        "-" * 96,
        (
            f"{'Method':<30}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
    ]

    methods = [
        (
            "Full-P3",
            full_result,
        ),
        (
            "Actor-P3",
            actor_result,
        ),
        (
            "PoseInteraction-P3",
            pose_result,
        ),
        (
            "Random-P3",
            random_result,
        ),
    ]

    for name, result in methods:
        lines.append(
            (
                f"{name:<30}"
                f"{result['accuracy'] * 100:>13.2f}%"
                f"{result['macro_f1'] * 100:>13.2f}%"
                f"{result['weighted_f1'] * 100:>15.2f}%"
            )
        )

    lines.extend([
        "",
        "POSE-INTERACTION DELTAS",
        "-" * 96,
        (
            "PoseInteraction vs Actor "
            "Macro-F1: "
            f"{(pose_result['macro_f1'] - actor_result['macro_f1']) * 100:+.2f} pp"
        ),
        (
            "PoseInteraction vs Random "
            "Macro-F1: "
            f"{(pose_result['macro_f1'] - random_result['macro_f1']) * 100:+.2f} pp"
        ),
        "",
        "COMPLEMENTARITY WITH RANDOM-P3",
        "-" * 96,
        (
            "Pose correct / Random wrong: "
            f"{pose_only.sum()}"
        ),
        (
            "Pose wrong / Random correct: "
            f"{random_only.sum()}"
        ),
        (
            "Oracle Random + Pose accuracy: "
            f"{oracle.mean() * 100:.2f}%"
        ),
    ])

    report = "\n".join(
        lines
    )

    print(report)

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels,
        filenames=filenames,
        class_names=np.asarray(
            dataset.classes
        ),

        pose_interaction_p3_similarities=(
            pose_scores
        ),

        pose_interaction_predictions=(
            pose_result[
                "predictions"
            ]
        ),

        pose_accuracy=np.asarray(
            pose_result[
                "accuracy"
            ]
        ),

        pose_macro_f1=np.asarray(
            pose_result[
                "macro_f1"
            ]
        ),

        pose_weighted_f1=np.asarray(
            pose_result[
                "weighted_f1"
            ]
        ),

        person_score_threshold=np.asarray(
            PERSON_SCORE_THRESHOLD
        ),

        crop_padding=np.asarray(
            CROP_PADDING
        ),

        fallback_count=np.asarray(
            fallback_count
        ),

        text_strategy=np.asarray(
            "p3"
        ),
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "true_class",
                "person_score",
                "fallback_actor_crop",
                "crop_left",
                "crop_top",
                "crop_right",
                "crop_bottom",
            ],
        )

        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    print()
    print(
        f"NPZ saved to:    "
        f"{OUTPUT_NPZ}"
    )

    print(
        f"CSV saved to:    "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Report saved to: "
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
