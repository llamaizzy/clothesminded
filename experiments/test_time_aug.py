########################################
# Solution 3. Test Time Augmentation
########################################
"""
Improves model performance during inference by applying transformations/corrections to the test data before passing into the model. 
Let model vote across different orientations -> predictions from these augmented images can then be aggregated to make
a more robust final prediction
"""
import json
import torch
import joblib
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score
from src.model.models import load_model
from src.data.load_data import load_rotated_data
from src.evaluate import evaluate

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
test_data = torch.load("experiments/data/rotated_test_set.pt")

angles = [0, -15, 15, 45, -45, 90, -90]
def evaluate_tta(model, dataloader, device, angles=angles):
    """
    Simple TTA: Rotate each image by fixed angles, get softmax 
    probabilities across all rotations for each image, sum them up, and 
    take argmax as final prediction (highest scoring class).
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # accumulate probabilities across all rotations
            total_probs = torch.zeros(x.shape[0], 10).to(device)
            for angle in angles:
                rotated = TF.rotate(x, angle)
                preds = model(rotated)
                total_probs += torch.softmax(preds, dim=1)
            pred = total_probs.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def evaluate_tta_improved(model, regressor, dataloader, device):
    """
    Improved TTA: for each rotation of the test image, use the regression 
    model to predict its angle and correct it back toward upright. Averages 
    probabilities across 7 outer rotations x 5 offset corrections = 35 
    forward passes per image, making predictions more robust to regressor error.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            total_probs = torch.zeros(x.shape[0], 10).to(device)
            for angle in angles:
                # 1. rotate batch by angle
                rotated = TF.rotate(x, angle)
                # 2. predict angles for whole batch with regressor
                x_flat = rotated.cpu().numpy().reshape(len(rotated), -1)
                pred_angles = regressor.predict(x_flat)

                # 3. for each image, try 5 corrections around predicted angle
                for i, pred_angle in enumerate(pred_angles):
                    img = rotated[i].unsqueeze(0)         # (1, C, H, W)
                    offset_probs = []

                    for offset in range(-2, 3):
                        unrotated_img = TF.rotate(img, -float((pred_angle + offset)))
                        scores = model(unrotated_img)
                        offset_probs.append(torch.softmax(scores, dim=1))
                    avg_prob = torch.stack(offset_probs).mean(dim=0) # average over 5 offsets
                    total_probs[i] += avg_prob.squeeze(0) # accumulate across all angles
            
            preds = total_probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

# compare baseline, simple tta, and tta with regressor accuracy
model = load_model(device, path="checkpoints/baseline_model.pth")
test_loader = load_rotated_data(train=False)
regressor = joblib.load("checkpoints/rotation_regression_model.joblib")

simple_tta_accuracy, _, _ = evaluate_tta(model, test_loader, device)
print(f"TTA Accuracy (simple): {simple_tta_accuracy:.4f}")
improved_tta_accuracy, predictions, _ = evaluate_tta_improved(model, regressor, test_loader, device)
print(f"TTA Accuracy (with regressor): {improved_tta_accuracy:.4f}")

# Save results to json
results = {
    "simple_tta": round(simple_tta_accuracy, 4),
    "improved_tta": round(improved_tta_accuracy, 4),
}

with open("experiments/results/tta_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to experiments/results/tta_results.json")