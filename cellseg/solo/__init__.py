import torch.nn as nn
from torch import Tensor
from .heads import Head
from typing import Protocol


class Backbone(Protocol):
    feature_channels: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...


class Solo(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        in_channels: int,
        out_channels: int,
        grid_size: int,
        category_feat_range: tuple[int, int] = (3, 5),
        mask_feat_range: tuple[int, int] = (0, 3),
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.category_feat_range = category_feat_range
        self.mask_feat_range = mask_feat_range
        self.category_head = Head(
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            fpn_length=category_feat_range[1] - category_feat_range[0],
        )

        self.mask_head = Head(
            fpn_length=mask_feat_range[1] - mask_feat_range[0],
            in_channels=in_channels,
            out_channels=out_channels,
            num_classes=grid_size ** 2,
        )
        self.backbone = backbone

    def forward(self, image_batch: Tensor) -> None:
        features = self.backbone(image_batch)
        mask_feats = features[self.mask_feat_range[0] : self.mask_feat_range[1]]
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        # category_grid = self.category_head(category_feats)
        # for f in category_feats:
        #     print(f.shape)

        # category_grid = self.category_head(features[*])

        # features = self.backbone.extract_endpoints(image_batch)
        # for k,v in features.items():
        #     print(v.shape)
