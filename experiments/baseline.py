import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from src.model.cnn_model import CNN, load_model, save_model
from src.model.train import train
from src.evaluate import evaluate, evaluate_rotation_shift
from src.data.load_data import get_dataloaders
from src.data.transforms import get_train_transform, get_clean_test_transform
import matplotlib.pyplot as plt
import plotly.express as px

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train_new_model = True
model_path = "checkpoints/model.pth"
model = load_model(device, path=None if train_new_model else model_path)

# -------- evaluate normal test set ------------
train_loader, test_loader = get_dataloaders(
    get_train_transform(), get_clean_test_transform()
)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if train_new_model:
    for epoch in range(20):
        loss = train(model, train_loader, opt, loss_fn, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    save_model(model, "checkpoints/model.pth")

predictions, labels, accuracy, conf_matrix = evaluate(model, test_loader, device)
print("Test Accuracy:", accuracy)
print("Confusion matrix:", conf_matrix)

# --------- evaluate rotated test set -----------
degrees = [0, 15, 30, 45, 60, 90]

results, rotated_conf_matrix= evaluate_rotation_shift(model, degrees, device)
print(results)
print("Confusion matrix with rotated test:", rotated_conf_matrix)


