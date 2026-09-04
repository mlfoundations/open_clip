from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_DIR = Path(
    r"C:\Projects\stanford40_split\train_clean"
)

VALIDATION_DIR = Path(
    r"C:\Projects\stanford40_split\validation"
)

OUTPUT_DIR = Path(
    r"C:\Projects\stanford40_results\resnet50_baseline_clean"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "best_resnet50_validation.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "selected_resnet50_validation.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "resnet50_validation_report.txt"
)


EXPECTED_TRAIN = 6090
EXPECTED_VALIDATION = 1521
EXPECTED_CLASSES = 40

MODEL_NAME = "resnet50"
PRETRAINING = "IMAGENET1K_V2"

BATCH_SIZE = 16
NUM_WORKERS = 0

STAGE1_EPOCHS = 2
STAGE2_EPOCHS = 6

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
):

    return {
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
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    labels_all = []
    predictions_all = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            labels_all.append(
                labels.cpu().numpy()
            )

            predictions_all.append(
                predictions.cpu().numpy()
            )

    labels_np = np.concatenate(
        labels_all
    )

    predictions_np = np.concatenate(
        predictions_all
    )

    result = calculate_metrics(
        labels_np,
        predictions_np,
    )

    result["loss"] = (
        total_loss
        / len(labels_np)
    )

    return result


# ============================================================
# Training epoch
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    stage,
):

    # Keep frozen backbone BatchNorm statistics fixed.
    model.eval()

    if stage == 1:

        model.fc.train()

    elif stage == 2:

        model.layer4.train()
        model.fc.train()

    else:

        raise ValueError(
            f"Unknown training stage: {stage}"
        )

    total_loss = 0.0

    labels_all = []
    predictions_all = []

    processed = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        labels_all.append(
            labels.detach()
            .cpu()
            .numpy()
        )

        predictions_all.append(
            predictions.detach()
            .cpu()
            .numpy()
        )

        processed += (
            images.size(0)
        )

        if (
            processed % 1000
            < BATCH_SIZE
            or processed
            == len(loader.dataset)
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{len(loader.dataset)}"
            )

    labels_np = np.concatenate(
        labels_all
    )

    predictions_np = np.concatenate(
        predictions_all
    )

    result = calculate_metrics(
        labels_np,
        predictions_np,
    )

    result["loss"] = (
        total_loss
        / len(labels_np)
    )

    return result


# ============================================================
# Trainable-layer configuration
# ============================================================

def configure_stage1(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    for parameter in model.fc.parameters():

        parameter.requires_grad = True


def configure_stage2(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    for parameter in model.layer4.parameters():

        parameter.requires_grad = True

    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print()
    print(
        "PHASE 3 - SUPERVISED RESNET-50 BASELINE"
    )
    print("=" * 94)

    print(
        "NON-VLM / VISION-ONLY BASELINE"
    )

    print(
        "IMPORTANT: Test split is NOT used."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    imagenet_mean = (
        0.485,
        0.456,
        0.406,
    )

    imagenet_std = (
        0.229,
        0.224,
        0.225,
    )

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.80, 1.00),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            imagenet_mean,
            imagenet_std,
        ),
    ])

    validation_transform = transforms.Compose([

        transforms.Resize(
            232
        ),

        transforms.CenterCrop(
            224
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            imagenet_mean,
            imagenet_std,
        ),
    ])

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )

    validation_dataset = ImageFolder(
        VALIDATION_DIR,
        transform=validation_transform,
    )

    if len(train_dataset) != EXPECTED_TRAIN:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_TRAIN} train images, "
            f"found {len(train_dataset)}."
        )

    if (
        len(validation_dataset)
        != EXPECTED_VALIDATION
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_VALIDATION} validation images, "
            f"found {len(validation_dataset)}."
        )

    if (
        len(train_dataset.classes)
        != EXPECTED_CLASSES
    ):

        raise ValueError(
            "Unexpected number of classes."
        )

    if (
        train_dataset.classes
        != validation_dataset.classes
    ):

        raise ValueError(
            "Train/validation class order differs."
        )

    print()
    print(
        "DATASET INTEGRITY: PASSED"
    )

    print(
        f"Training images:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Classes:           "
        f"{len(train_dataset.classes)}"
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Loading ImageNet-pretrained ResNet-50..."
    )

    weights = (
        ResNet50_Weights.IMAGENET1K_V2
    )

    model = resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        EXPECTED_CLASSES,
    )

    model = model.to(
        device
    )

    print(
        "Model: ResNet-50"
    )

    print(
        "Pretraining: ImageNet-1K V2"
    )

    print(
        "Output classes: 40"
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    history = []

    best_result = None
    best_epoch = None
    global_epoch = 0

    # ========================================================
    # Two frozen training stages
    # ========================================================

    stages = [
        {
            "stage": 1,
            "epochs": STAGE1_EPOCHS,
            "learning_rate": STAGE1_LR,
            "description": (
                "classifier head only"
            ),
        },

        {
            "stage": 2,
            "epochs": STAGE2_EPOCHS,
            "learning_rate": STAGE2_LR,
            "description": (
                "layer4 + classifier"
            ),
        },
    ]

    for stage_config in stages:

        stage = stage_config[
            "stage"
        ]

        epochs = stage_config[
            "epochs"
        ]

        learning_rate = stage_config[
            "learning_rate"
        ]

        print()
        print("=" * 94)

        print(
            f"STAGE {stage}: "
            f"{stage_config['description']}"
        )

        print(
            f"Epochs: {epochs}"
        )

        print(
            f"Learning rate: "
            f"{learning_rate}"
        )

        print("=" * 94)

        if stage == 1:

            configure_stage1(
                model
            )

        else:

            configure_stage2(
                model
            )

        trainable_parameters = [
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ]

        trainable_count = sum(
            parameter.numel()
            for parameter
            in trainable_parameters
        )

        print(
            f"Trainable parameters: "
            f"{trainable_count:,}"
        )

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        for stage_epoch in range(
            1,
            epochs + 1,
        ):

            global_epoch += 1

            print()
            print(
                f"GLOBAL EPOCH "
                f"{global_epoch:02d} "
                f"(Stage {stage}, "
                f"Epoch {stage_epoch}/{epochs})"
            )

            print(
                "-" * 94
            )

            train_result = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                stage,
            )

            validation_result = evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )

            print()
            print(
                "TRAIN"
            )

            print(
                f"Loss:        "
                f"{train_result['loss']:.4f}"
            )

            print(
                f"Accuracy:    "
                f"{train_result['accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro-F1:    "
                f"{train_result['macro_f1'] * 100:.2f}%"
            )

            print()
            print(
                "VALIDATION"
            )

            print(
                f"Loss:        "
                f"{validation_result['loss']:.4f}"
            )

            print(
                f"Accuracy:    "
                f"{validation_result['accuracy'] * 100:.2f}%"
            )

            print(
                f"Macro-F1:    "
                f"{validation_result['macro_f1'] * 100:.2f}%"
            )

            print(
                f"Weighted-F1: "
                f"{validation_result['weighted_f1'] * 100:.2f}%"
            )

            history.append({
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "learning_rate": learning_rate,

                "train_loss": (
                    train_result["loss"]
                ),

                "train_accuracy": (
                    train_result["accuracy"]
                ),

                "train_macro_f1": (
                    train_result["macro_f1"]
                ),

                "train_weighted_f1": (
                    train_result[
                        "weighted_f1"
                    ]
                ),

                "validation_loss": (
                    validation_result[
                        "loss"
                    ]
                ),

                "validation_accuracy": (
                    validation_result[
                        "accuracy"
                    ]
                ),

                "validation_macro_f1": (
                    validation_result[
                        "macro_f1"
                    ]
                ),

                "validation_weighted_f1": (
                    validation_result[
                        "weighted_f1"
                    ]
                ),
            })

            current_key = (
                validation_result[
                    "macro_f1"
                ],

                validation_result[
                    "accuracy"
                ],

                validation_result[
                    "weighted_f1"
                ],
            )

            if best_result is None:

                is_better = True

            else:

                best_key = (
                    best_result[
                        "macro_f1"
                    ],

                    best_result[
                        "accuracy"
                    ],

                    best_result[
                        "weighted_f1"
                    ],
                )

                is_better = (
                    current_key
                    > best_key
                )

            if is_better:

                best_result = (
                    validation_result.copy()
                )

                best_epoch = (
                    global_epoch
                )

                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),

                        "architecture": (
                            MODEL_NAME
                        ),

                        "pretraining": (
                            PRETRAINING
                        ),

                        "classes": (
                            train_dataset.classes
                        ),

                        "class_to_idx": (
                            train_dataset.class_to_idx
                        ),

                        "selected_epoch": (
                            global_epoch
                        ),

                        "selected_stage": (
                            stage
                        ),

                        "validation_accuracy": (
                            validation_result[
                                "accuracy"
                            ]
                        ),

                        "validation_macro_f1": (
                            validation_result[
                                "macro_f1"
                            ]
                        ),

                        "validation_weighted_f1": (
                            validation_result[
                                "weighted_f1"
                            ]
                        ),

                        "random_seed": (
                            RANDOM_SEED
                        ),
                    },
                    CHECKPOINT_PATH,
                )

                print()
                print(
                    ">>> BEST VALIDATION "
                    "CHECKPOINT SAVED"
                )

    # ========================================================
    # Save history
    # ========================================================

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    # ========================================================
    # Freeze validation selection
    # ========================================================

    summary = {
        "architecture": (
            "ResNet-50"
        ),

        "model_type": (
            "non-VLM supervised CNN"
        ),

        "pretraining": (
            "ImageNet-1K V2"
        ),

        "training_images": (
            EXPECTED_TRAIN
        ),

        "validation_images": (
            EXPECTED_VALIDATION
        ),

        "classes": (
            EXPECTED_CLASSES
        ),

        "stage1_epochs": (
            STAGE1_EPOCHS
        ),

        "stage2_epochs": (
            STAGE2_EPOCHS
        ),

        "stage1_learning_rate": (
            STAGE1_LR
        ),

        "stage2_learning_rate": (
            STAGE2_LR
        ),

        "selection_split": (
            "validation"
        ),

        "selection_metric": (
            "macro_f1_then_accuracy_then_weighted_f1"
        ),

        "test_used": False,

        "selected_epoch": (
            best_epoch
        ),

        "validation_accuracy": (
            best_result[
                "accuracy"
            ]
        ),

        "validation_macro_f1": (
            best_result[
                "macro_f1"
            ]
        ),

        "validation_weighted_f1": (
            best_result[
                "weighted_f1"
            ]
        ),

        "random_seed": (
            RANDOM_SEED
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final report
    # ========================================================

    lines = [
        "",
        (
            "SUPERVISED RESNET-50 "
            "VALIDATION RESULT"
        ),
        "=" * 94,

        (
            "Model type: "
            "non-VLM supervised CNN"
        ),

        (
            "Pretraining: "
            "ImageNet-1K V2"
        ),

        (
            f"Selected epoch: "
            f"{best_epoch}"
        ),

        (
            "Selection split: validation"
        ),

        (
            "Test split used: NO"
        ),

        "",
        "SELECTED VALIDATION PERFORMANCE",
        "-" * 94,

        (
            f"Accuracy:    "
            f"{best_result['accuracy'] * 100:.2f}%"
        ),

        (
            f"Macro-F1:    "
            f"{best_result['macro_f1'] * 100:.2f}%"
        ),

        (
            f"Weighted-F1: "
            f"{best_result['weighted_f1'] * 100:.2f}%"
        ),

        (
            f"Loss:        "
            f"{best_result['loss']:.4f}"
        ),
    ]

    report = "\n".join(
        lines
    )

    print(
        report
    )

    REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"History:    "
        f"{HISTORY_PATH}"
    )

    print(
        f"Summary:    "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Report:     "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()

