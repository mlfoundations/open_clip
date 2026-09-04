from pathlib import Path

import numpy as np
import open_clip
import torch

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)

from evaluate_pose_interaction_validation import (
    calculate_metrics,
    create_interaction_crop,
    create_p3_text_features,
    PERSON_SCORE_THRESHOLD,
    CROP_PADDING,
)

from contextual_prompt_bank import (
    validate_prompt_bank,
)


# ============================================================
# FROZEN TEST CONFIGURATION
# ============================================================

ACTOR_DIR = Path(
    r"C:\Projects\stanford40_views\test_actor_20"
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

FULL_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "test_contextual_prompt_results.npz"
)

RANDOM_FILE = (
    RESULTS_ROOT
    / "context_prompts"
    / "test_contextual_random_same_crop.npz"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "test_pose_interaction_p3_results.npz"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "test_pose_interaction_p3_report.txt"
)


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

EXPECTED_IMAGES = 1921
EXPECTED_CLASSES = 40


def main():

    print()
    print(
        "FROZEN POSE-INTERACTION P3 TEST"
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
        "Method frozen from validation"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = ImageFolder(
        ACTOR_DIR
    )

    if len(dataset) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(dataset)}."
        )

    if len(dataset.classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {len(dataset.classes)}."
        )

    validate_prompt_bank(
        dataset.classes
    )

    labels = np.asarray(
        [
            label
            for _, label
            in dataset.samples
        ],
        dtype=np.int64,
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

    print(
        f"Images:  {len(dataset)}"
    )

    print(
        f"Classes: {len(dataset.classes)}"
    )

    # ========================================================
    # Load frozen Full-P3
    # ========================================================

    with np.load(
        FULL_FILE,
        allow_pickle=True,
    ) as data:

        full_labels = data[
            "labels"
        ].astype(np.int64)

        full_classes = data[
            "class_names"
        ].astype(str)

        full_filenames = data[
            "filenames"
        ].astype(str)

        full_scores = data[
            "p3_similarities"
        ].astype(np.float32)

    # ========================================================
    # Load frozen Random-P3
    # ========================================================

    with np.load(
        RANDOM_FILE,
        allow_pickle=True,
    ) as data:

        random_labels = data[
            "labels"
        ].astype(np.int64)

        random_classes = data[
            "class_names"
        ].astype(str)

        random_filenames = data[
            "filenames"
        ].astype(str)

        random_scores = data[
            "p3_similarities"
        ].astype(np.float32)

    # ========================================================
    # Integrity
    # ========================================================

    if not np.array_equal(
        labels,
        full_labels,
    ):
        raise ValueError(
            "Actor labels differ from Full-P3."
        )

    if not np.array_equal(
        labels,
        random_labels,
    ):
        raise ValueError(
            "Actor labels differ from Random-P3."
        )

    if not np.array_equal(
        filenames,
        full_filenames,
    ):
        raise ValueError(
            "Actor filenames differ from Full-P3."
        )

    if not np.array_equal(
        filenames,
        random_filenames,
    ):
        raise ValueError(
            "Actor filenames differ from Random-P3."
        )

    if not np.array_equal(
        np.asarray(dataset.classes),
        full_classes,
    ):
        raise ValueError(
            "Class order differs from Full-P3."
        )

    if not np.array_equal(
        np.asarray(dataset.classes),
        random_classes,
    ):
        raise ValueError(
            "Class order differs from Random-P3."
        )

    print(
        "Integrity checks: PASSED"
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # OpenCLIP
    # ========================================================

    clip_model, _, clip_preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
        )
    )

    tokenizer = (
        open_clip.get_tokenizer(
            MODEL_NAME
        )
    )

    clip_model.eval()

    p3_text = create_p3_text_features(
        clip_model,
        tokenizer,
        dataset.classes,
        device,
    )

    # ========================================================
    # Frozen Keypoint R-CNN
    # ========================================================

    print()
    print(
        "Loading frozen COCO Keypoint R-CNN..."
    )

    pose_model = (
        keypointrcnn_resnet50_fpn(
            weights=(
                KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        )
        .to(device)
    )

    pose_model.eval()

    # ========================================================
    # Test inference
    # ========================================================

    actor_rows = []
    pose_rows = []

    fallback_count = 0

    print()
    print(
        "Evaluating frozen pose-interaction method..."
    )

    with torch.inference_mode():

        for index, (
            image_path,
            _
        ) in enumerate(
            dataset.samples
        ):

            with Image.open(
                image_path
            ) as loaded:

                actor_image = (
                    loaded.convert(
                        "RGB"
                    )
                )

                pose_input = (
                    to_tensor(
                        actor_image
                    )
                    .to(device)
                )

                detection = (
                    pose_model(
                        [pose_input]
                    )[0]
                )

                (
                    pose_crop,
                    _,
                    fallback,
                    _,
                ) = create_interaction_crop(
                    actor_image,
                    detection,
                )

                if fallback:
                    fallback_count += 1

                clip_batch = torch.stack([
                    clip_preprocess(
                        actor_image
                    ),
                    clip_preprocess(
                        pose_crop
                    ),
                ]).to(device)

            features = (
                clip_model.encode_image(
                    clip_batch
                )
            )

            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            similarities = (
                features
                @ p3_text.T
            )

            actor_rows.append(
                similarities[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            pose_rows.append(
                similarities[1]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            processed = (
                index + 1
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

    actor_scores = np.stack(
        actor_rows,
        axis=0,
    )

    pose_scores = np.stack(
        pose_rows,
        axis=0,
    )

    # ========================================================
    # Metrics
    # ========================================================

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

    both_correct = (
        pose_correct
        & random_correct
    )

    pose_only = (
        pose_correct
        & ~random_correct
    )

    random_only = (
        ~pose_correct
        & random_correct
    )

    both_wrong = (
        ~pose_correct
        & ~random_correct
    )

    disagreement = (
        pose_pred
        != random_pred
    )

    oracle = (
        pose_correct
        | random_correct
    )

    detection_success = (
        1.0
        - fallback_count
        / len(labels)
    )

    # ========================================================
    # Report
    # ========================================================

    lines = [
        "",
        (
            "FROZEN POSE-INTERACTION "
            "P3 TEST RESULT"
        ),
        "=" * 96,
        (
            f"Images: {len(labels)}"
        ),
        (
            f"Fallbacks to actor crop: "
            f"{fallback_count}"
        ),
        (
            f"Detection success: "
            f"{detection_success * 100:.2f}%"
        ),
        "",
        "TEST PERFORMANCE",
        "-" * 96,
        (
            f"{'Method':<30}"
            f"{'Accuracy':>14}"
            f"{'Macro-F1':>14}"
            f"{'Weighted-F1':>16}"
        ),
    ]

    for name, result in [
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
    ]:

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
            "PoseInteraction vs Full "
            "Macro-F1: "
            f"{(pose_result['macro_f1'] - full_result['macro_f1']) * 100:+.2f} pp"
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
            f"Both correct: "
            f"{both_correct.sum()} "
            f"({both_correct.mean() * 100:.2f}%)"
        ),
        (
            "Pose correct / Random wrong: "
            f"{pose_only.sum()} "
            f"({pose_only.mean() * 100:.2f}%)"
        ),
        (
            "Pose wrong / Random correct: "
            f"{random_only.sum()} "
            f"({random_only.mean() * 100:.2f}%)"
        ),
        (
            f"Both wrong: "
            f"{both_wrong.sum()} "
            f"({both_wrong.mean() * 100:.2f}%)"
        ),
        (
            f"Prediction disagreement: "
            f"{disagreement.sum()} "
            f"({disagreement.mean() * 100:.2f}%)"
        ),
        (
            "Oracle Random + Pose accuracy: "
            f"{oracle.mean() * 100:.2f}%"
        ),
    ])

    report = "\n".join(
        lines
    )

    print(
        report
    )

    OUTPUT_REPORT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        labels=labels,
        filenames=filenames,
        class_names=np.asarray(
            dataset.classes
        ),

        actor_p3_similarities=(
            actor_scores
        ),

        pose_interaction_p3_similarities=(
            pose_scores
        ),

        actor_p3_predictions=(
            actor_result[
                "predictions"
            ]
        ),

        pose_interaction_predictions=(
            pose_result[
                "predictions"
            ]
        ),

        actor_accuracy=np.asarray(
            actor_result[
                "accuracy"
            ]
        ),

        actor_macro_f1=np.asarray(
            actor_result[
                "macro_f1"
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

        fallback_count=np.asarray(
            fallback_count
        ),

        person_score_threshold=np.asarray(
            PERSON_SCORE_THRESHOLD
        ),

        crop_padding=np.asarray(
            CROP_PADDING
        ),

        text_strategy=np.asarray(
            "p3"
        ),
    )

    print()
    print(
        f"NPZ saved to: "
        f"{OUTPUT_NPZ}"
    )

    print(
        f"Report saved to: "
        f"{OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()
