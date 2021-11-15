import os
from omegaconf import OmegaConf
import enum

conf = OmegaConf.load("./config.yml")


class CellType(str, enum.Enum):
    shsy5y = "shsy5y"
    astro = "astro"
    cort = "cort"


appearance_rates = {
    CellType.shsy5y: 155 / 606,
    CellType.astro: 131 / 60,
    CellType.cort: 320 / 60,
}

root_path = "/store"
train_file_path = os.path.join(root_path, "train.csv")
train_image_path = os.path.join(root_path, "train")
