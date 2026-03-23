from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
