import hydra
from torch import Tensor
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = getLogger(cfg.name)
    logger.info("aaaa")


if __name__ == "__main__":
    main()
