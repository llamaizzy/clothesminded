import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),  
            nn.ReLU(),
            nn.MaxPool2d(2),      

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),         # converts 2D feature to 1D vector
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)    # output layer: 10 classes
        )
    
    def forward(self, x):
        return self.net(x)

class RotationRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)      # single output: predicted angle
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (batch,)
    
def load_model(device, type, path=None):
    model = type().to(device)

    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded model from {path}")
    else:
        print("Initialized new model")

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to path{path}")
