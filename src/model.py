import torch
import torch.nn as nn


class PotatoCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # 6 x [Conv2d(k=3) -> ReLU -> MaxPool2d(2)]
        # channels: 3 -> 32 -> 64 -> 64 -> 64 -> 64 -> 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Spatial size trace starting from 256x256:
        # Conv1: 254 -> Pool1: 127
        # Conv2: 125 -> Pool2: 62
        # Conv3: 60  -> Pool3: 30
        # Conv4: 28  -> Pool4: 14
        # Conv5: 12  -> Pool5: 6
        # Conv6: 4   -> Pool6: 2
        # Flattened features = 64 * 2 * 2 = 256
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    m = PotatoCNN()
    n_params = sum(p.numel() for p in m.parameters())
    print("params:", n_params)                       # must be 183,747
    print(m(torch.randn(2, 3, 256, 256)).shape)      # must be [2, 3]