import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from wandb.plot import confusion_matrix

from src.models.modules.base import unpack_outputs


def compute_preds(logits):
    return F.softmax(logits, dim=1).argmax(dim=1)


class Distiller(LightningModule):
    def __init__(self, teacher: LightningModule, student: LightningModule, temperature: float = 1.5, lr=0.01) -> None:
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()
        self.student = student
        self.save_hyperparameters()

    @property
    def temperature(self) -> float:
        return self.hparams.get("temperature", 1.0)

    def training_step(self, batch, _):
        x, y = batch

        teacher_logits, student_logits = self(x)

        # Compute loss from soft targets (predictions from cumbersome model)
        teacher_preds = compute_preds(teacher_logits / self.temperature)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        distillation_loss = F.nll_loss(student_probs, teacher_preds)
        self.log(f"training_distillation_loss", distillation_loss, on_step=True, on_epoch=True)

        # Compute loss from hard targets (ground truth)
        student_loss = F.cross_entropy(student_logits, y.view(-1))
        self.log(f"training_student_loss", student_loss, on_step=True, on_epoch=True)

        # Compute and log weighted loss
        alpha = 1 / (1 + self.temperature ** 2)
        beta = 1 - alpha
        loss = alpha * student_loss + beta * distillation_loss
        self.log(f"training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        # Compute predictions
        teacher_preds, student_preds = map(compute_preds, self(x))
        return dict(teacher_preds=teacher_preds, student_preds=student_preds, targets=y.view(-1))

    def test_epoch_end(self, outputs) -> None:
        # Unpack
        targets = unpack_outputs(outputs, "targets").numpy()
        classes = list(map(int, range(10)))

        # Compute confusion matrices
        for model in ["teacher", "student"]:
            preds = unpack_outputs(outputs, f"{model}_preds").numpy()
            confmat = confusion_matrix(y_true=targets, preds=preds, class_names=classes)
            self.logger.experiment.log({f"{model}_confusion_matrix": confmat})

    def forward(self, x):
        return self.teacher(x), self.student(x)
