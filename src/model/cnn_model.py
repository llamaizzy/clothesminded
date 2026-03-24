import torch
import torch.nn as nn

# class ConvBlock(nn.Module):
#     """Conv → BN → ReLU → Conv → BN → ReLU → MaxPool with residual shortcut."""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#         )
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         out = self.relu(self.block(x))
#         return out

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

def load_model(device, path=None):
    model = CNN().to(device)

    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded model from {path}")
    else:
        print("Initialized new model")

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to path{path}")
