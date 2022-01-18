__all__ = ["PredictionModule"]

from typing import Dict, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from wandb.plot import confusion_matrix

from src.path import checkpoint_path


class PredictionModule(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.training_accuracy = Accuracy(average="macro", num_classes=num_classes)
        self.validation_accuracy = Accuracy(average="macro", num_classes=num_classes)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr: float = self.hparams.get("lr")
        weight_decay: float = self.hparams.get("weight_decay")
        return torch.optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)

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
        if mode not in ("training", "validation", "test"):
            raise ValueError(f"Invalid mode ´{mode}´")

        # Do forward pass
        x, y = batch
        probs = F.log_softmax(self(x), dim=1)

        # Calculate loss and predictions
        criterion = nn.NLLLoss()
        loss = criterion(probs, y.view(-1))
        preds = probs.exp().argmax(dim=1)

        # Update training/validation confusion matrix with step's values
        # Note: predictions & labels must be flattened: N-dimensional
        log_on_epoch = True
        if mode in ("training", "validation"):
            getattr(self, f"{mode}_accuracy")(preds, y.view(-1))
        else:
            log_on_epoch = False

        # Log loss at every step
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=log_on_epoch, prog_bar=True)
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
    def load_from_checkpoint(cls, project: str, experiment: str) -> LightningModule:
        path = checkpoint_path(project, experiment)
        return super().load_from_checkpoint(path)
