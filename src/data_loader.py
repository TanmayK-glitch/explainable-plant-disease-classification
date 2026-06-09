import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"

data = datasets.ImageFolder(root=DEFAULT_DATA_DIR)
    
def create_data_loaders(batch_size=32, train_ratio=0.8, val_ratio=0.1, seed=42, dir_data=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)

    base_dataset = datasets.ImageFolder(data_dir)
    dataset_size = len(base_dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset, val_subset = random_split(base_dataset, [train_size, test_size, val_size], generator=generator)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #Applying the trasnformations
    train_subset.dataset.transform = train_transform
    test_subset.dataset.transform = val_test_transform
    val_subset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    return train_loader, test_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")