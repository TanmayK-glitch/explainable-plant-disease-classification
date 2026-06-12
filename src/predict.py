import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
NUM_CLASSES = 15

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",                 # 0
    "Pepper__bell___healthy",                        # 1
    "Potato___Early_blight",                         # 2
    "Potato___Late_blight",                          # 3
    "Potato___healthy",                              # 4
    "Tomato_Bacterial_spot",                         # 5
    "Tomato_Early_blight",                           # 6
    "Tomato_Late_blight",                            # 7
    "Tomato_Leaf_Mold",                              # 8
    "Tomato_Septoria_leaf_spot",                     # 9
    "Tomato_Spider_mites_Two_spotted_spider_mite",  # 10
    "Tomato__Target_Spot",                           # 11
    "Tomato__Tomato_YellowLeaf__Curl_Virus",         # 12
    "Tomato__Tomato_mosaic_virus",                   # 13
    "Tomato_healthy",                                # 14
]


def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = transform(image).unsqueeze(0)

    rgb_image = image.resize((224, 224))
    rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

    return input_tensor, rgb_image


def predict(image_path, model, device, top_k=3):
    input_tensor, _ = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = max(1, min(top_k, len(CLASS_NAMES)))
    confidences, indices = torch.topk(probabilities, k=top_k)

    return [
        {
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index],
            "confidence": confidence,
        }
        for confidence, class_index in zip(
            confidences.cpu().tolist(),
            indices.cpu().tolist(),
        )
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the disease class of a plant leaf image."
    )
    parser.add_argument("image", type=Path, help="Path to the leaf image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Model checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to display (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, device = load_model(args.checkpoint)
    predictions = predict(args.image, model, device, args.top_k)

    best = predictions[0]
    print(f"Device: {device}")
    print(f"Prediction: {best['class_name']}")
    print(f"Confidence: {best['confidence']:.2%}")

    if len(predictions) > 1:
        print("\nTop predictions:")
        for rank, prediction in enumerate(predictions, start=1):
            print(
                f"{rank}. {prediction['class_name']}: "
                f"{prediction['confidence']:.2%}"
            )


if __name__ == "__main__":
    main()
