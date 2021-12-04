import torch
from torch import Tensor
from typing import Callable, Any, Optional
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBnAct, DefaultActivation
from .backbones import FPNLike
from .necks import NeckLike
from .heads import MaskHead
from .utils import grid_points
from .loss import DIoULoss, FocalLoss
from torchvision.ops import roi_align, box_convert, masks_to_boxes
from .assign import IoUAssign
from cellseg.utils import round_to
import math


class DecoupledHeadUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width_mult: float = 1.0,
        num_classes: int = 80,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ):
        super().__init__()
        hidden_channels = round_to(256 * width_mult)
        self.stem_conv = ConvBnAct(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
        )
        self.reg_conv = nn.Sequential(
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.reg_out_conv = nn.Conv2d(
            in_channels=hidden_channels, out_channels=4, kernel_size=1
        )
        self.obj_out_conv = nn.Conv2d(
            in_channels=hidden_channels, out_channels=1, kernel_size=1
        )
        self.cls_conv = nn.Sequential(
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.cls_out_conv = nn.Conv2d(hidden_channels, num_classes, 1)
        self._init_weights()

    def _init_weights(self, prior_prob: Any = 1e-2) -> None:
        for m in [self.cls_out_conv, self.obj_out_conv]:
            nn.init.constant_(m.bias, -math.log((1 - prior_prob) / prior_prob))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem_conv(x)
        h1 = self.reg_conv(h)
        y_reg = self.reg_out_conv(h1)
        y_obj = self.obj_out_conv(h1)
        y_cls = self.cls_out_conv(self.cls_conv(h))
        return torch.cat([y_reg, y_obj, y_cls], dim=1)


class YoloxHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int] = [256, 512, 1024],
        num_classes: int = 80,
        width_mult: float = 1.0,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.heads = nn.ModuleList(
            [
                DecoupledHeadUnit(
                    in_channels=in_chs,
                    num_classes=num_classes,
                    width_mult=width_mult,
                    act=act,
                )
                for in_chs in in_channels
            ]
        )

    def forward(
        self, inputs: list[Tensor]
    ) -> list[Tensor]:  # list[b, num_classes + 5, h, w]
        return [m(x) for m, x in zip(self.heads, inputs)]


class MaskYolo(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        neck: NeckLike,
        num_classes: int,
        mask_size: int,
        top_fpn_level: int,
        mask_feat_range: tuple[int, int] = (3, 5),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.mask_size = mask_size
        self.top_fpn_level = top_fpn_level
        self.num_classes = num_classes
        self.reductions = self.neck.reductions
        self.box_head = YoloxHead(
            in_channels=neck.out_channels, num_classes=num_classes
        )
        self.local_mask_head = MaskHead(
            in_channels=sum(neck.out_channels), out_channels=1
        )
        self.strides = self.neck.reductions
        self.mask_feat_range = mask_feat_range
        self.local_mask_stride = self.strides[self.mask_feat_range[0]]

    def box_branch(self, feats: list[Tensor]) -> list[Tensor]:
        return self.box_head(feats)

    def local_mask_branch(self, box_batch: Tensor, feats: list[Tensor]) -> list[Tensor]:
        first_level_size = feats[0].shape[2:]
        merged_feats = torch.cat(
            [
                feat if idx == 0 else F.interpolate(feat, size=first_level_size)
                for idx, feat in enumerate(feats)
            ],
            dim=1,
        )
        roi_feats = roi_align(
            merged_feats,
            list(box_batch.unsqueeze(1)),
            self.mask_size,
        )
        local_masks = self.local_mask_head(roi_feats)
        return local_masks

    def features(self, x: Tensor) -> list[Tensor]:
        feats = self.backbone(x)[: self.top_fpn_level]
        return self.neck(feats)

    # def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    #     feats = self.features(x)
    #     box_preds = self.box_branch(feats)
    #     box_batch = self.to_boxes(box_preds)
    #     return box_preds, feats


class ToBoxes:
    def __init__(
        self,
        strides: list[int],
    ) -> None:
        self.strides = strides

    def __call__(self, yolo_batch: Tensor) -> None:
        device = yolo_batch.device
        box_batch = box_convert(yolo_batch[..., :4], in_fmt="cxcywh", out_fmt="xyxy")
        obj_batch = yolo_batch[..., 4]
        cate_batch = yolo_batch[..., 5:]
        label_batch = cate_batch.argmax(-1)
        return obj_batch, box_batch, cate_batch, label_batch


class MergeLevel:
    def __init__(
        self,
        strides: list[int],
    ) -> None:
        self.strides = strides

    def __call__(self, box_levels: list[Tensor]) -> None:
        device = box_levels[0].device
        yolo_box_list = []
        for pred, stride in zip(box_levels, self.strides):
            batch_size, num_outputs, rows, cols = pred.shape
            grid = grid_points(rows, cols).to(device)
            strides = torch.full((len(grid),), stride).to(device)
            yolo_boxes = (
                pred.permute(0, 2, 3, 1)
                .reshape(batch_size, rows * cols, num_outputs)
                .float()
            )
            yolo_boxes = torch.cat(
                [
                    (yolo_boxes[..., 0:2] + grid) * stride,
                    yolo_boxes[..., 2:4].exp() * stride,
                    yolo_boxes[..., 4:],
                ],
                dim=-1,
            ).reshape(batch_size, rows * cols, num_outputs)
            yolo_box_list.append(yolo_boxes)
        yolo_batch = torch.cat(yolo_box_list, dim=1)
        return yolo_batch


class Criterion:
    def __init__(
        self,
        model: MaskYolo,
        obj_weight: float = 1.0,
        box_weight: float = 1.0,
        cate_weight: float = 1.0,
        local_mask_weight: float = 1.0,
    ) -> None:
        self.box_weight = box_weight
        self.cate_weight = cate_weight
        self.obj_weight = obj_weight
        self.local_mask_weight = local_mask_weight
        self.model = model
        self.strides = self.model.strides
        self.assign = IoUAssign(threshold=0.2)
        self.box_loss = DIoULoss()
        self.obj_loss = FocalLoss()
        self.to_boxes = ToBoxes(strides=self.strides)
        self.merge_level = MergeLevel(strides=self.strides)
        self.local_mask_loss = FocalLoss()
        self.cate_loss = F.binary_cross_entropy_with_logits

    def __call__(
        self,
        inputs: tuple[Tensor],
        targets: tuple[list[Tensor], list[Tensor]],
    ) -> None:
        (images,) = inputs  # list of [b, num_classes + 5, h, w]
        gt_mask_batch, gt_label_batch = targets
        device = images.device
        feats = self.model.features(images)
        box_levels = self.model.box_branch(feats)
        pred_yolo_batch = self.merge_level(box_levels)
        pred_obj_batch, pred_box_batch, pred_cate_batch, _ = self.to_boxes(
            pred_yolo_batch
        )
        gt_yolo_batch, gt_local_mask_batch, pos_idx = self.prepeare_gt(
            gt_mask_batch, gt_label_batch, pred_box_batch
        )
        gt_obj_batch, gt_boxes_batch, gt_cate_batch, _ = self.to_boxes(gt_yolo_batch)

        # 1-stage
        obj_loss = self.obj_loss(pred_obj_batch, gt_obj_batch)

        box_loss, cate_loss, local_mask_loss = (
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        if pos_idx.sum() > 0:
            box_loss += self.box_loss(pred_box_batch[pos_idx], gt_boxes_batch[pos_idx])
            cate_loss += self.cate_loss(
                pred_cate_batch[pos_idx], gt_cate_batch[pos_idx]
            )

            # 2-stage
            pred_local_masks = self.model.local_mask_branch(
                gt_boxes_batch[pos_idx], feats
            )
            local_mask_loss += self.local_mask_loss(
                pred_local_masks, gt_local_mask_batch
            )

        loss = (
            self.box_weight * box_loss
            + self.obj_weight * obj_loss
            + self.cate_weight * cate_loss
            + self.local_mask_weight * local_mask_loss
        )
        return loss, obj_loss, box_loss, cate_loss, local_mask_loss

    def prepeare_gt(
        self,
        gt_mask_batch: list[Tensor],
        gt_label_batch: list[Tensor],
        pred_box_batch: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
        device = pred_box_batch.device
        num_classes = self.model.num_classes
        gt_yolo_batch = torch.zeros(
            (pred_box_batch.shape[0], pred_box_batch.shape[1], num_classes + 5),
            dtype=pred_box_batch.dtype,
            device=device,
        )
        mask_size = self.model.mask_size
        mask_stride = self.model.local_mask_stride

        gt_local_mask_list = []
        for batch_idx, (gt_masks, gt_labels, pred_boxes) in enumerate(
            zip(gt_mask_batch, gt_label_batch, pred_box_batch)
        ):
            gt_boxes = masks_to_boxes(gt_masks)
            gt_cxcywh = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")

            gt_local_masks = roi_align(
                gt_masks.float().unsqueeze(1),
                list(gt_boxes.unsqueeze(1)),
                mask_size,
            )
            gt_local_mask_list.append(gt_local_masks)
            matched = self.assign(gt_boxes, pred_boxes)
            gt_yolo_batch[batch_idx, matched[:, 1], :4] = gt_cxcywh[matched[:, 0]]
            gt_yolo_batch[batch_idx, matched[:, 1], 4] = 1.0
            gt_yolo_batch[batch_idx, matched[:, 1], 5:] = F.one_hot(
                gt_labels[matched[:, 0]], num_classes
            ).to(gt_yolo_batch)
        pos_idx = gt_yolo_batch[..., 4] > 0
        gt_local_mask_batch = torch.cat(gt_local_mask_list)
        return gt_yolo_batch, gt_local_mask_batch, pos_idx
