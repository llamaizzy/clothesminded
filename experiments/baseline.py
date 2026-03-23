import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from src.model.cnn_model import CNN
from src.model.train import train
from src.evaluation.evaluate import evaluate, evaluate_rotation_shift
from src.data.load_data import get_dataloaders
from src.data.transforms import get_train_transform, get_clean_test_transform
import matplotlib.pyplot as plt
import plotly.express as px

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)

# -------- evaluate normal test set ------------
train_loader, test_loader = get_dataloaders(
    get_train_transform(), get_clean_test_transform()
)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    train(model, train_loader, opt, loss_fn, device)

predictions, labels, accuracy, conf_matrix = evaluate(model, test_loader, device)
print("Test Accuracy:", accuracy)
px.imshow(conf_matrix)

# --------- evaluate rotated test set -----------
# degrees = [0, 15, 30, 45, 60, 90]

# results = evaluate_rotation_shift(model, degrees, device)
# print(results)


