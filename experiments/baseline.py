#####################################
#       Baseline Model
#####################################
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.model.models import load_model, save_model
from src.model.train import train
from src.evaluate import evaluate, evaluate_augmented
from src.data.load_data import get_dataloaders, load_rotated_data
from src.data.transforms import get_clean_transform
import plotly.express as px

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train_new_model = False
model_path = "checkpoints/baseline_model.pth"
model = load_model(device, path=None if train_new_model else model_path)

# Load data
train_loader, test_loader = get_dataloaders(
    get_clean_transform(), get_clean_transform()
)

# Train classifier and create augmented results dataset
if train_new_model:
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(model, train_loader, opt, loss_fn, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    save_model(model, model_path)
    aug_data = []
    distortion_configs = [
        ("rotation", [45, 90, 200]),
        ("blur", [3, 5, 11]),
        ("shift", [(5, 0), (-5, 0), (0, 5), (0, -5)]),
        ("rotate_blur", [(45, 3), (90, 5)]),
        ("shift_rotate_blur", [((5, 0), 45, 3), ((-5, 0), 90, 5)])
    ]
    for distortion, severities in distortion_configs:
        results, _ = evaluate_augmented(model, device, distortion, severities)
        for severity, accuracy in results.items():
            aug_data.append({
                "augmentation": f"{distortion}_{severity}",
                "accuracy": accuracy,
                "type": distortion,
                "severity": severity,
            })
    aug_performance = pd.DataFrame(aug_data)
    print(aug_performance)
    aug_performance.to_csv("experiments/distortion_results.csv")

# Evaluate classifier on normal test data
train_accuracy, _, _, _ = evaluate(model, train_loader, device)
baseline_normal_accuracy, _, _, _= evaluate(model, test_loader, device)
print(f"Baseline Train Accuracy: {train_accuracy:.4f}")
print(f"Baseline Test Accuracy: {baseline_normal_accuracy:.4f}")

# Evaluate classifer on (manually) augmented test data
if os.path.exists("experiments/distortion_results.csv"):
    aug_performance = pd.read_csv(aug_fig_path)
    sorted_df = aug_performance.sort_values('accuracy', ascending=False)
    fig = px.bar(sorted_df, 
                x='augmentation', 
                y='accuracy', 
                color='type', title="Model accuracy by augmentation")

    fig.show()
    fig.write_image("experiments/plots/distortion_comparison.png")

# Evaluate classifier on rotated test data
rotated_data = load_rotated_data(train=False)
baseline_rotated_accuracy, _, _, _ = evaluate(model, rotated_data, device)
print(f"Baseline accuracy on rotated test: {baseline_rotated_accuracy:.4f}")

# Save accuracy to json
results = {
    "baseline_normal": round(baseline_normal_accuracy, 4),
    "baseline_rotated": round(baseline_rotated_accuracy, 4)}

with open("experiments/results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Results saved to experiments/results/baseline_results.json")