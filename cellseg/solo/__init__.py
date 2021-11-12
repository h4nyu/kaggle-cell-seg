import torch.nn as nn
from torch import Tensor


class Solo(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()


    def forward(self, image_batch: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
        ...
