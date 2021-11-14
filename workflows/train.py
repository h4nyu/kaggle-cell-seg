import os
import torch
from torch import Tensor
from torch import nn
from cellseg.solo import Solo
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, **kargs) -> None:
        super().__init__()
        self.net = Solo(**kargs)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
        return self.net(x)

# @hydra.main(config_path='config.yaml')
# k
