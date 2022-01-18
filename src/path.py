__all__ = ["CONF_PATH", "WEIGHTS_PATH"]

from pathlib import Path
from dotenv import find_dotenv

ROOT_PATH = Path(find_dotenv()).parent

CONF_PATH = ROOT_PATH / "conf"
WEIGHTS_PATH = ROOT_PATH / "models"
