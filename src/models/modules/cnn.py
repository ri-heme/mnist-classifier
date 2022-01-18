__all__ = ["CNN"]

from torch import nn
import torch

from src.models.modules.base import PredictionModule


class CNN(PredictionModule):
    def __init__(self, num_planes: int = 8, num_classes: int = 10, lr: float = 0.01, weight_decay: float = 0):
        super().__init__(num_classes)
        if not (8 <= num_planes <= 14):
            raise ValueError(f"Number of planes must be between 8-14. Got {num_planes}.")
        self.conv = nn.Sequential(nn.Conv2d(1, num_planes, 7, padding=3, stride=2, bias=False), nn.ReLU())
        self.stack = nn.Sequential(
            nn.Conv2d(num_planes, num_planes, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_planes),
            nn.ReLU(),
            nn.Conv2d(num_planes, num_planes, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_planes),
        )
        self.fc = nn.Sequential(nn.ReLU(), nn.AvgPool2d(num_planes), nn.Flatten(), nn.Linear(num_planes, num_classes))
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv(x)
        out = self.stack(residual)
        out += residual
        return self.fc(out)
