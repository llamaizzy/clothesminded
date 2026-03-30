import torch
import numpy as np
from torchvision.datasets import FashionMNIST
from src.data.transforms import get_clean_transform
import torchvision.transforms.functional as TF

SEED = 42
SAVE_PATH = "experiments/data/rotated_test_set.pt"

def create_rotated_dataset(save_path: str = SAVE_PATH, train: bool = True):
    # load raw test set without any transform
    dataset = FashionMNIST(root='./data', train=train, download=True,
                                 transform=get_clean_transform())

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    images, labels, angles = [], [], []

    for img, label in dataset:
        angle = float(np.random.uniform(0, 360))    # generate random angle
        rotated = TF.rotate(img, angle)
        images.append(rotated)
        labels.append(label)
        angles.append(angle)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    angles = torch.tensor(angles)

    torch.save({"images": images, "labels": labels, "angles": angles}, save_path)
    print(f"Saved rotated test set to {save_path}")
    print(f"Images: {images.shape}, Labels: {labels.shape}, Angles: {angles.shape}")

if __name__ == "__main__":
    create_rotated_dataset("experiments/data/rotated_train_set.pt", train=True)
    create_rotated_dataset("experiments/data/rotated_test_set.pt", train=False)