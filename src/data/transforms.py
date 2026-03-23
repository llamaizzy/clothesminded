from torchvision import transforms
import torch

def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_clean_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),   # converts image to tensor and normalize to [0, 1]
    ])

def get_rotation_transform(degrees):
    return transforms.Compose([
        transforms.RandomRotation(degrees),
        transforms.ToTensor(),
    ])

