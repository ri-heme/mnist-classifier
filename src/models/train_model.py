from pathlib import Path

import hydra
import wandb
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloaders import dataloaders
from src.models.cnn import CNN

CONF_PATH = Path(find_dotenv(), "..", "conf").as_posix()
WEIGHTS_PATH = Path(find_dotenv(), "..", "models").as_posix()


@hydra.main(config_path=CONF_PATH, config_name="main")
def train(config: DictConfig):
    train_dataloader, test_dataloader = dataloaders(config.batch_size)

    # Init model
    model: CNN = hydra.utils.instantiate(config.model)

    # Init logger
    logger = WandbLogger(project="mnist-classifier", id=config.name, log_model=False)

    # Init trainer
    trainer: Trainer = hydra.utils.instantiate(config.training, logger=logger, weights_save_path=WEIGHTS_PATH)
    trainer.fit(model, train_dataloader)

    trainer.test(model, test_dataloader)

    # Close
    wandb.finish()


if __name__ == "__main__":
    train()
