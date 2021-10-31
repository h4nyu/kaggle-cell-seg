import os
from omegaconf import OmegaConf


conf = OmegaConf.load("./config.yml")


ROOT_PATH = "/store"
TRAIN_FILE_NAME = "train.csv"

TRAIN_FILE_PATH = os.path.join(ROOT_PATH, TRAIN_FILE_NAME)

TRAIN_PATH = os.path.join(ROOT_PATH, "train")
