# import os
# import torch
# from torch import nn
# from cellseg.solo import Solo


# class LitAutoEncoder(pl.LightningModule):
#     def __init__(self) -> None:
#         super().__init__()
#         self.net = Solo()

#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
#         embedding = self.encoder(x)
#         return embedding

# @hydra.main(config_path='config.yaml')
# k
