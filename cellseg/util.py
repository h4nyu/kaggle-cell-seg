import numpy as np
import torch
import random
import torch.nn as nn
from typing import Optional, TypeVar, Generic, Any
from omegaconf import OmegaConf
from pathlib import Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


T = TypeVar("T", bound=nn.Module)


class Checkpoint(Generic[T]):
    def __init__(self, root_path: str, default_score: float) -> None:
        self.root_path = Path(root_path)
        self.model_path = self.root_path.joinpath("checkpoint.pth")
        self.checkpoint_path = self.root_path.joinpath("checkpoint.yaml")
        self.default_score = default_score
        self.root_path.mkdir(exist_ok=True)

    def load_if_exists(self, model: T) -> tuple[T, float]:
        if self.model_path.exists() and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.model_path))
            conf = OmegaConf.load(self.checkpoint_path)
            score = conf.get("score", self.default_score)  # type: ignore
            return model, score
        else:
            return model, self.default_score

    def save(self, model: T, score: float) -> float:
        torch.save(model.state_dict(), self.model_path)  # type: ignore
        OmegaConf.save(config=dict(score=score), f=self.checkpoint_path)
        return score