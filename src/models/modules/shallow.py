__all__ = ["ShallowNN"]

import torch
from torch import nn

from src.models.modules.base import PredictionModule


class ShallowNN(PredictionModule):
    def __init__(self, in_features: int = 784, hidden_features: int = 397, out_features: int = 10, lr: float = 1e-3):
        super().__init__(out_features)
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, out_features)
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))
