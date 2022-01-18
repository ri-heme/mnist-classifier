import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloaders import dataloaders
from src.models.cnn import CNN
from src.path import CONF_PATH, WEIGHTS_PATH


@hydra.main(config_path=CONF_PATH.as_posix(), config_name="main")
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
