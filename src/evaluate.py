import torch
from src.data.load_data import get_dataloaders
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from src.data.transforms import (
    get_train_transform,
    get_rotate_transform,
    get_blur_transform,
    get_shift_transform,
    get_rotate_blur_transform,
    get_shift_rotate_blur_transform,
)

# evaluate on normal test set
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1) # gets predicted class via highest score
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_true, all_preds)
    conf_matrix = confusion_matrix(all_true, all_preds)

    return all_preds, all_true, accuracy, conf_matrix

# evaluate on augmented test set
def evaluate_augmented(model, device, distortion, severity_list, batch_size=64):
    """
    distortion: one of "rotate", "blur", "shift", "rotate_blur", "shift_rotate_blur"
    severity_list:
        rotation            -> list of degrees, e.g. [0, 15, 30, 45, 90]
        blur              -> list of kernel sizes (odd ints), e.g. [3, 5, 7]
        shift             -> list of pixel offsets, e.g. [(0, 5), (-5, 0)]
        rotate_blur       -> list of (degree, kernel size)
        shift_rotate_blur -> list of (offset, degree, kernel size)
    """
    results = {}
    conf_matrix = None
    for p in severity_list:
        if distortion == "rotation":
            test_transform = get_rotate_transform(p)
        elif distortion == "blur":
            test_transform = get_blur_transform(kernel_size=p)
        elif distortion == "shift":
            x, y = p
            test_transform = get_shift_transform(shift_x=x, shift_y=y)
        elif distortion == "rotate_blur":
            test_transform = get_rotate_blur_transform(*p)
        elif distortion == "shift_rotate_blur":
            test_transform = get_shift_rotate_blur_transform(*p)
        else:
            raise ValueError(f"Unknown distortion: '{distortion}'. Choose rotate, blur, shift, rotate_blur, or shift_rotate_blur")

        _, test_loader = get_dataloaders(
            train_transform=get_train_transform(),
            test_transform=test_transform,
            batch_size=batch_size
        )

        _, _, accuracy, conf_matrix = evaluate(model, test_loader, device)
        results[p] = accuracy

    return results, conf_matrix

