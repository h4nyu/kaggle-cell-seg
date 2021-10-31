import os
from omegaconf import OmegaConf
import enum


class CellType(str, enum.Enum):
    shsy5y = "shsy5y"
    astro = "astro"
    cort = "cort"


appearance_rates = {
    CellType.shsy5y: 155 / 606,
    CellType.astro: 131 / 60,
    CellType.cort: 320 / 60,
}


conf = OmegaConf.load("./config.yml")


ROOT_PATH = "/store"
TRAIN_FILE_NAME = "train.csv"

TRAIN_FILE_PATH = os.path.join(ROOT_PATH, TRAIN_FILE_NAME)

TRAIN_PATH = os.path.join(ROOT_PATH, "train")
