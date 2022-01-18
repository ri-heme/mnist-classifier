from typing import Dict, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from wandb.plot import confusion_matrix

from src.path import checkpoint_path


class CNN(LightningModule):
    def __init__(self, num_planes=8, num_classes=10, lr=0.01, weight_decay=0):
        super().__init__()
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
        self.training_accuracy = Accuracy(average="macro", num_classes=num_classes)
        self.validation_accuracy = Accuracy(average="macro", num_classes=num_classes)
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), weight_decay=self.hparams.weight_decay, lr=self.hparams.lr)

    def step(self, batch: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """Computes loss and updates the confusion matrix at every step.

        Parameters
        ----------
        batch : torch.Tensor
        mode : {'training', 'validation', 'test'}

        Returns
        -------
        outputs : Dict[str, torch.Tensor]
        """
        # Do forward pass
        x, y = batch
        logits = F.log_softmax(self(x), dim=1)

        # Calculate loss and predictions
        criterion = nn.NLLLoss()
        loss = criterion(logits, y.view(-1))
        preds = logits.exp().argmax(dim=1)

        # Update training/validation confusion matrix with step's values
        # Note: predictions & labels must be flattened: N-dimensional
        if mode in ("training", "validation"):
            getattr(self, f"{mode}_accuracy")(preds, y.view(-1))

        # Log loss at every step
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return dict(loss=loss, preds=preds, targets=y.view(-1))

    def epoch_end(self, mode: str) -> None:
        """Computes and logs confusion matrix at end of every epoch.

        Parameters
        ----------
        mode : {'training', 'validation'}
        """
        # Compute & log accuracy
        accuracy = getattr(self, f"{mode}_accuracy")
        self.log(f"{mode}_accuracy", accuracy.compute())
        accuracy.reset()

    def training_step(self, batch: torch.Tensor, _) -> Dict[str, torch.Tensor]:
        return self.step(batch, "training")

    def training_epoch_end(self, _) -> None:
        self.epoch_end("training")

    def validation_step(self, batch: torch.Tensor, _) -> Dict[str, torch.Tensor]:
        return self.step(batch, "validation")

    def validation_epoch_end(self, _) -> None:
        self.epoch_end("validation")

    def test_step(self, batch: torch.Tensor, _) -> Dict[str, torch.Tensor]:
        return self.step(batch, "test")

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        def cat_outputs(key: str) -> torch.Tensor:
            return torch.cat([step_output[key] for step_output in outputs])

        preds = cat_outputs("preds").numpy()
        targets = cat_outputs("targets").numpy()
        classes = list(map(int, range(10)))
        # SEE ALSO: wandb.sklearn.plot_confusion_matrix(targets, preds, classes)
        confmat = confusion_matrix(y_true=targets, preds=preds, class_names=classes)
        self.logger.experiment.log({"confusion_matrix": confmat})

    @classmethod
    def load_from_checkpoint(cls, experiment: str) -> LightningModule:
        path = checkpoint_path(experiment)
        return super().load_from_checkpoint(path)

    def forward(self, x):
        residual = self.conv(x)
        out = self.stack(residual)
        out += residual
        return self.fc(out)
