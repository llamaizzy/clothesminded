import torch
from src.data.load_data import get_dataloaders
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# evaluate on normal test set
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1) # gets predicted class via highest score
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return all_preds, all_labels, accuracy, conf_matrix

# evaluate on augmented test set (robustness)
def evaluate_rotation_shift(model, degrees_list, device, batch_size=64):
    from src.data.transforms import get_rotation_transform

    results = {}

    for deg in degrees_list:
        _, test_loader = get_dataloaders(
            train_transform=get_rotation_transform(deg),
            test_transform=get_rotation_transform(deg),
            batch_size=batch_size
        )

        preds, labels, accuracy, conf_matrix = evaluate(model, test_loader, device)
        results[deg] = accuracy
    
    return results, conf_matrix