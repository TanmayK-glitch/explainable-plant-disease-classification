import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import DEFAULT_DATA_DIR, create_data_loaders
from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "confusion_matrix.png"

CLASS_NAMES = [
    "Pepper bell - Bacterial spot",
    "Pepper bell - Healthy",
    "Potato - Early blight",
    "Potato - Late blight",
    "Potato - Healthy",
    "Tomato - Bacterial spot",
    "Tomato - Early blight",
    "Tomato - Late blight",
    "Tomato - Leaf mold",
    "Tomato - Septoria leaf spot",
    "Tomato - Spider mites",
    "Tomato - Target spot",
    "Tomato - Yellow leaf curl virus",
    "Tomato - Mosaic virus",
    "Tomato - Healthy",
]


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, test_loader, device):
    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


def save_confusion_matrix(matrix, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    indices = np.arange(len(CLASS_NAMES))
    axis.set(
        title="Plant Disease Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=indices,
        yticks=indices,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate(checkpoint_path, data_dir, output_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    _, _, test_loader = create_data_loaders(
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        data_dir=data_dir,
    )

    true_labels, predicted_labels = collect_predictions(
        model,
        test_loader,
        device,
    )

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels,
    )
    save_confusion_matrix(matrix, output_path)

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(f"Device: {device}")
    print(f"Test samples: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print(f"Confusion matrix saved to: {Path(output_path).resolve()}")

    return true_labels, predicted_labels, matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the best plant disease model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"PlantVillage dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Confusion matrix image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
