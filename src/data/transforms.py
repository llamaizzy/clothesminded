from torchvision import transforms
import torch

_MEAN = 0.2860
_STD = 0.3530

def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((_MEAN ,), (_STD,))
    ])

def get_clean_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),   # converts image to tensor and normalize to [0, 1]
        transforms.Normalize((_MEAN,), (_STD,))
    ])

def get_rotate_transform(degrees):
    """Fixed rotation by exactly `degrees` (not random)."""
    return transforms.Compose([
        transforms.RandomRotation((degrees, degrees)),
        transforms.ToTensor(),
        transforms.Normalize((_MEAN,), (_STD))
    ])

def get_random_rotate_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=(0, 360)),  # random rotation every epoch
        transforms.ToTensor(),
        transforms.Normalize((_MEAN ,), (_STD,))
    ])
def get_blur_transform(kernel_size):
    """Gaussian blur with given kernel size (must be odd)."""
    return transforms.Compose([
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((_MEAN,), (_STD,))
    ])

def get_shift_transform(shift_x, shift_y):
    """Translate image by (shift_x, shift_y) pixels."""
    return transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.affine(img, angle=0, translate=[shift_x, shift_y], scale=1.0, shear=0)),transforms.ToTensor(),
        transforms.Normalize((_MEAN,), (_STD,))
    ])

def get_rotate_blur_transform(degrees, kernel_size):
    """Fixed rotation then Gaussian blur."""
    return transforms.Compose([
        transforms.RandomRotation((degrees, degrees)),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((_MEAN,), (_STD,))
    ])

def get_shift_rotate_blur_transform(shift, degrees, kernel_size):
    """Shift, then fixed rotation, then Gaussian blur."""
    shift_x, shift_y = shift
    return transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.affine(img, angle=0, translate=[shift_x, shift_y], scale=1.0, shear=0)),
        transforms.RandomRotation((degrees, degrees)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=1.0),
        transforms.Normalize((_MEAN,), (_STD,))
    ])

