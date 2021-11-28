import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable, Any
from cellseg.loss import FocalLoss, DiceLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import masks_to_boxes, box_convert
from .utils import grid, draw_save, ToPatches, MergePatchedMasks
from .backbones import FPNLike


class CenterMask(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        hidden_channels: int,
        mask_size: int,
        category_feat_range: tuple[int, int],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.category_feat_range = category_feat_range
        self.backbone = backbone
        self.category_head = Head(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.size_head = Head(
            hidden_channels=hidden_channels,
            num_classes=2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.offset_head = Head(
            hidden_channels=hidden_channels,
            num_classes=2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.sliency_head = Head(
            hidden_channels=hidden_channels,
            num_classes=1,
            channels=backbone.channels,
            reductions=backbone.reductions,
            use_cord=False,
        )

        self.mask_head = Head(
            hidden_channels=hidden_channels,
            num_classes=mask_size ** 2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=True,
        )

    def forward(
        self, image_batch: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.backbone(image_batch)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grids = self.category_head(category_feats)
        size_grids = self.size_head(category_feats)
        offset_grids = self.offset_head(category_feats)
        mask_grids = self.mask_head(category_feats)
        sliency_masks = self.sliency_head(features)
        return (category_grids, size_grids, offset_grids, mask_grids, sliency_masks)
