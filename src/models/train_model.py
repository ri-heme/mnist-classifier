from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloaders import dataloaders
from src.models.cnn import CNN

def train():
    train_dataloader, test_dataloader = dataloaders()

    model = CNN()

    logger = WandbLogger(project="mnist-classifier", log_model=False)

    trainer = Trainer(max_epochs=10, logger=logger)
    trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    train()
