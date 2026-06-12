import argparse
import csv
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import datasets, transforms

from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(checkpoint_path, num_classes, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=num_classes)
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


def preprocess_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
        rgb_image = np.asarray(
            image.resize((224, 224)),
            dtype=np.float32,
        ) / 255.0

    return input_tensor, rgb_image


def predict(model, input_tensor, device):
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor.to(device)), dim=1)[0]

    predicted_index = probabilities.argmax().item()
    return predicted_index, probabilities[predicted_index].item()


def is_healthy(class_name):
    return "healthy" in class_name.lower()


def select_examples(dataset, model, device, count, seed):
    indices = list(range(len(dataset.samples)))
    random.Random(seed).shuffle(indices)

    selected = {
        "correct": [],
        "healthy": [],
        "disease": [],
    }
    used_paths = set()

    for index in indices:
        image_path, true_index = dataset.samples[index]
        input_tensor, _ = preprocess_image(image_path)
        predicted_index, confidence = predict(model, input_tensor, device)

        if predicted_index != true_index:
            continue

        true_class = dataset.classes[true_index]
        example = {
            "image_path": Path(image_path),
            "true_class": true_class,
            "predicted_index": predicted_index,
            "predicted_class": dataset.classes[predicted_index],
            "confidence": confidence,
        }

        if len(selected["correct"]) < count:
            selected["correct"].append(example)
            used_paths.add(image_path)
            continue

        category = "healthy" if is_healthy(true_class) else "disease"
        if (
            len(selected[category]) < count
            and image_path not in used_paths
        ):
            selected[category].append(example)
            used_paths.add(image_path)

        if all(len(examples) >= count for examples in selected.values()):
            return selected

    missing = {
        category: count - len(examples)
        for category, examples in selected.items()
        if len(examples) < count
    }
    raise RuntimeError(f"Not enough correctly predicted examples: {missing}")


def safe_filename(value):
    value = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return value.strip("_")


def save_gradcam(cam, example, category, number, output_dir, device):
    input_tensor, rgb_image = preprocess_image(example["image_path"])
    targets = [ClassifierOutputTarget(example["predicted_index"])]

    grayscale_cam = cam(
        input_tensor=input_tensor.to(device),
        targets=targets,
    )[0]
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    filename = (
        f"{category}_{number:02d}_"
        f"{safe_filename(example['true_class'])}_gradcam.png"
    )
    output_path = output_dir / filename

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    figure.suptitle(
        f"True: {example['true_class']}\n"
        f"Predicted: {example['predicted_class']} "
        f"({example['confidence']:.2%})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_manifest(rows, output_dir):
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "category",
        "true_class",
        "predicted_class",
        "confidence",
        "source_image",
        "gradcam_image",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def generate_examples(
    checkpoint_path,
    data_dir,
    output_dir,
    count=5,
    seed=42,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, len(dataset.classes), device)
    selected = select_examples(dataset, model, device, count, seed)

    rows = []
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        for category, examples in selected.items():
            for number, example in enumerate(examples, start=1):
                output_path = save_gradcam(
                    cam,
                    example,
                    category,
                    number,
                    output_dir,
                    device,
                )
                rows.append({
                    "category": category,
                    "true_class": example["true_class"],
                    "predicted_class": example["predicted_class"],
                    "confidence": f"{example['confidence']:.6f}",
                    "source_image": str(example["image_path"]),
                    "gradcam_image": str(output_path),
                })
                print(f"Saved: {output_path.name}")

    manifest_path = write_manifest(rows, output_dir)
    print(f"\nGenerated {len(rows)} Grad-CAM images on {device}.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate correct, healthy, and disease Grad-CAM examples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Examples per category (default: 5, total: 15)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    generate_examples(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
    )
