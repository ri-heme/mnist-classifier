import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, ConfusionMatrix

class CNN(LightningModule):
    def __init__(self, num_planes=8, num_classes=10, lr=0.01, weight_decay=0):
        super().__init__()
        if not (8 <= num_planes <= 14):
            raise ValueError(f"Number of planes must be between 8-14. Got {num_planes}.")
        self.conv = nn.Sequential(
            nn.Conv2d(1, num_planes, 7, padding=3, stride=2, bias=False),
            nn.ReLU()
        )
        self.stack = nn.Sequential(
            nn.Conv2d(num_planes, num_planes, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_planes),
            nn.ReLU(),
            nn.Conv2d(num_planes, num_planes, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_planes),     
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(num_planes),
            nn.Flatten(),
            nn.Linear(num_planes, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.training_accuracy = Accuracy(average="macro", num_classes=num_classes)
        self.validation_accuracy = Accuracy(average="macro", num_classes=num_classes)
        self.training_confmat = ConfusionMatrix(normalize="true", num_classes=num_classes)
        self.validation_confmat = ConfusionMatrix(normalize="true", num_classes=num_classes)
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr
        )

    def step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        """Computes loss and updates the confusion matrix at every step.

        Parameters
        ----------
        batch : torch.Tensor
        mode : {'training', 'validation'}

        Returns
        -------
        loss : torch.Tensor
        """
        # Do forward pass
        x, y = batch
        logits = self(x)

        # Calculate loss and predictions
        criterion = nn.NLLLoss()
        loss = criterion(logits, y.view(-1))
        preds = logits.exp().argmax(dim=1)

        # Update training/validation confusion matrix with step's values
        # Note: predictions & labels must be flattened: N-dimensional
        getattr(self, f"{mode}_accuracy")(preds, y.view(-1))
        getattr(self, f"{mode}_confmat")(preds, y.view(-1))

        # Log loss at every step
        self.log(
            f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

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
    
        # Compute & log confusion matrix
        confmat = getattr(self, f"{mode}_confmat")
        self.logger.log_image("confusion_matrix", [confmat.compute()])
        confmat.reset()

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "training")

    def training_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("training")

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "validation")

    def validation_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("validation")

    def forward(self, x):
        residual = self.conv(x)
        out = self.stack(residual)
        out += residual
        return self.fc(out)