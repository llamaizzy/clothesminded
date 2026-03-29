#############################################
# Solution 1. Rotation-invariant classifier
#############################################
"""
Align training distribution with test distribution.
Generate augmented training data so our model is more robust to variations in orientation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.model.models import load_model, save_model
from src.model.train import train
from src.evaluate import evaluate
from src.data.load_data import load_rotated_data
import plotly.express as px

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train_new_model = False
model_path = "checkpoints/rotation_invariant_model.pth"
model = load_model(device, path=None if train_new_model else model_path)

rotated_train_loader = load_rotated_data(train=True, target="labels")
rotated_test_loader = load_rotated_data(train=False, target="labels")

# Train
if train_new_model:
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        loss = train(model, rotated_train_loader, opt, loss_fn, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    save_model(model, "checkpoints/rotation_invariant_model.pth")

# Evaluate
preds, true_labels, accuracy, conf_matrix = evaluate(model, rotated_train_loader, device)
print(f"Test Accuracy: {accuracy:.4f}")