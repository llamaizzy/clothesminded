#####################################
# 2. Rotation Correction
#####################################
"""
Align test distribution with training distribution.
Undo rotations in test set by building a regression model to predict rotation angle of an image
and using the predicted angle to rotate the image back into its original orientation before feeding it
into the classifier
"""
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.model.models import RotationRegressor, load_model, save_model
from src.model.train import train
from src.evaluate import evaluate
from src.data.load_data import load_rotated_data
from src.data.transforms import get_random_rotate_transform
import plotly.express as px

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train_new_model = True
model_path = "checkpoints/rotation_correction_model.pth"
model = load_model(device, RotationRegressor, path=None if train_new_model else model_path)

rotated_train_loader = load_rotated_data(train=True, target="labels")
rotated_test_loader = load_rotated_data(train=False, target="labels")

train_reg_loader = load_rotated_data(train=True, target="angles")
test_reg_loader = load_rotated_data(train=False, target="angles")

# Train regression
if train_new_model:
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(5):
        loss = train(model, train_reg_loader, opt, loss_fn, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    save_model(model, "checkpoints/rotation_correction_model.pth")

# Evaluate regression model
pred_angles, _ , accuracy, conf_matrix = evaluate(model, test_reg_loader, device)
print(f"Regression Test Accuracy: {accuracy:.4f}")

# Make new predictions using the original classifier model and check accuracy of images classified
model = load_model(device, path="checkpoints/baseline_model.pth")

