from torchvision import transforms
import torch

_MEAN = 0.2860
_STD = 0.3530
def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

def get_clean_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),   # converts image to tensor and normalize to [0, 1]
        transforms.Normalize((0.2860,), (0.3530,))
    ])

def get_rotation_transform(degrees):
    return transforms.Compose([
        transforms.RandomRotation(degrees),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

