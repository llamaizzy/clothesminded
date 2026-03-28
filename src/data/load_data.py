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

# for rotation invariant's convenience
def get_train_loader(train_transform, batch_size=64):
    train_dataset = FashionMNIST(
        root='./data', train=True, download=True,
        transform=train_transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# for rotation invariant's convenience
def get_test_loader(test_transform, batch_size=64):
    test_dataset = FashionMNIST(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    return DataLoader(test_dataset, batch_size=batch_size)

def load_rotated_data(train: bool = True,
                         target: str = "labels",
                         batch_size: int = 64,
                         shuffle: bool = False):
    
    path = ("experiments/data/rotated_train_set.pt" if train 
            else "experiments/data/rotated_test_set.pt")
    
    data = torch.load(path)
    images = data["images"]
    y = data[target].float() if target == "angles" else data[target]

    normalize = transforms.Normalize((_MEAN,), (_STD,))
    images = torch.stack([normalize(img) for img in images])

    dataset = TensorDataset(images, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
