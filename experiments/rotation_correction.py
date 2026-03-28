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
from sklearn.metrics import mean_squared_error
import torch
import joblib
import numpy as np
from src.model.models import load_model
from src.evaluate import evaluate
from src.data.load_data import load_unrotated_data
from plotting import visualize_pred_true_angles, visualize_unrotation

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

regressor_path = "checkpoints/rotation_correction_model.pth"
train_new_reg = False

# -------- load data ------------
train_data = torch.load("experiments/data/rotated_train_set.pt")
X_train = train_data["images"].numpy().reshape(-1, 784)
y_train = train_data["angles"].numpy()

test_data = torch.load("experiments/data/rotated_test_set.pt")
X_test = test_data["images"].numpy().reshape(-1, 784)
y_test = test_data["angles"].numpy()
test_images = test_data["images"]
test_labels = test_data["labels"]

# -------- train or load regression model --------
if train_new_reg:
    regressor = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        verbose=True
    )
    regressor.fit(X_train, y_train)
    joblib.dump(regressor, regressor_path)
else:
    regressor = joblib.load(regressor_path)

# ---------- evaluate regressor ---------
y_pred_angles = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_angles)
rmse = np.sqrt(mse)
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")

# Show some predictions
print("\nSample predictions:")
for i in range(5):
    print(f"True rotation: {y_test[i]:.1f}°, Predicted: {y_pred_angles[i]:.1f}°")

# visualize relationship predicted vs true rotation angle
visualize_pred_true_angles(y_test, y_pred_angles)

# show unrotated vs original image
visualize_unrotation(test_images, true_angles=y_test, pred_angles=y_pred_angles, idx=0)

# ---------- unrotate and classify -------
classifier = load_model(device, path="checkpoints/baseline_model.pth")
unrotated_loader = load_unrotated_data(test_images, test_labels, y_pred_angles)
_, _, final_accuracy, conf_matrix = evaluate(classifier, unrotated_loader, device)
print(f"Test Accuracy after rotation correction: {final_accuracy:.4f}")