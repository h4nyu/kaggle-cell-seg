import torch.nn as nn
from torch import Tensor
from .heads import Head
from typing import Callable


Backbone = Callable[[Tensor], list[Tensor]]


class Solo(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        in_channels: int,
        out_channels: int,
        grid_size: int,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.category_head = Head(
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            fpn_length=4,
        )

        self.mask_head = Head(
            fpn_length=4,
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=grid_size ** 2,
        )
        self.backbone = backbone

    def forward(self, image_batch: Tensor) -> list[tuple[Tensor, Tensor]]:
        ...
        # features = self.backbone.extract_endpoints(image_batch)
        # for k,v in features.items():
        #     print(v.shape)
