__all__ = ["checkpoint_path", "CONF_PATH", "WEIGHTS_PATH"]

from pathlib import Path

from dotenv import find_dotenv

ROOT_PATH = Path(find_dotenv()).parent

CONF_PATH = ROOT_PATH / "conf"
WEIGHTS_PATH = ROOT_PATH / "models"


def checkpoint_path(experiment: str, project: str = "mnist-classifier") -> Path:
    try:
        return next(WEIGHTS_PATH.joinpath(project, experiment).glob("**/*.ckpt"))
    except StopIteration:
        raise FileNotFoundError(f"Checkpoint for experiment ´{experiment}´ not found")
