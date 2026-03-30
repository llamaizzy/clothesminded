from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.data.transforms import _MEAN, _STD
from torchvision import transforms

def get_dataloaders(train_transform, test_transform, batch_size=64):
    train_dataset = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform)
    
    test_dataset = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def load_rotated_data(train: bool = True, batch_size: int = 64, shuffle: bool = False):
    path = ("experiments/data/rotated_train_set.pt" if train 
            else "experiments/data/rotated_test_set.pt")
    
    data = torch.load(path)
    images = data["images"]
    labels = data["labels"]
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_unrotated_data(images, labels, angles, batch_size: int = 64):
    """
    Predict angles with regressor, and unrotate images.
    """
    corrected_images = torch.stack([
        transforms.functional.rotate(img, -float(angle))
        for img, angle in zip(images, angles)])
    dataset = TensorDataset(corrected_images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    