
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"

IMAGE_PATH = "/home/tanmay/Desktop/explainable-plant-disease-classification/data/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=15)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)

    rgb_image = image.resize((224, 224))
    rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

    return input_tensor, rgb_image


def generate_gradcam(image_path):
    model, device = load_model()

    input_tensor, rgb_image = preprocess_image(image_path)

    input_tensor = input_tensor.to(device)

    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_gradcam(IMAGE_PATH)