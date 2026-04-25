import torch.nn as nn


class BaselineCNN(nn.Module):
    """Large CNN with no regularization — designed to overfit."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 32x32 -> 16x16 -> 8x8 -> 4x4 with 256 channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(),
            nn.Linear(4096, 1024), nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
