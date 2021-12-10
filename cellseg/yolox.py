import torch
from torch import Tensor
from typing import Callable, Any, Optional
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBnAct, DefaultActivation
from .backbones import FPNLike
from .necks import NeckLike
from .heads import MaskHead
from .utils import grid_points, draw_save
from .loss import CIoULoss, FocalLoss
from torchvision.ops import roi_align, box_convert, masks_to_boxes, batched_nms
from .assign import ATSS, SimOTA
from cellseg.utils import round_to
import math

Batch = tuple[
    Tensor, list[Tensor], list[Tensor], list[Tensor]
]  # id, images, mask_batch, box_batch label_batch


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
        patch_size: int,
        box_limit: int = 1000,
        box_iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        mask_feat_range: tuple[int, int] = (0, 4),
        box_feat_range: tuple[int, int] = (3, 7),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.box_feat_range = box_feat_range
        self.mask_feat_range = mask_feat_range
        self.num_classes = num_classes
        self.box_limit = box_limit
        self.strides = self.neck.strides
        self.box_strides = self.strides[self.box_feat_range[0] : self.box_feat_range[1]]
        self.mask_stride = self.strides[self.mask_feat_range[0]]
        self.box_iou_threshold = box_iou_threshold
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        self.box_head = YoloxHead(
            in_channels=neck.out_channels[
                self.box_feat_range[0] : self.box_feat_range[1]
            ],
            num_classes=num_classes,
        )
        self.local_mask_head = MaskHead(
            in_channels=sum(
                neck.out_channels[self.mask_feat_range[0] : self.mask_feat_range[1]]
            ),
            out_channels=1,
        )

    def box_branch(self, feats: list[Tensor]) -> Tensor:
        device = feats[0].device
        box_levels = self.box_head(feats)
        yolo_box_list = []
        for pred, stride in zip(box_levels, self.box_strides):
            batch_size, num_outputs, rows, cols = pred.shape
            grid = grid_points(rows, cols).to(device)
            strides = torch.full((batch_size, len(grid), 1), stride).to(device)
            anchor_points = (grid + 0.5) * strides
            yolo_boxes = (
                pred.permute(0, 2, 3, 1)
                .reshape(batch_size, rows * cols, num_outputs)
                .float()
            )
            yolo_boxes = torch.cat(
                [
                    (yolo_boxes[..., 0:2] + 0.5 + grid) * strides,
                    yolo_boxes[..., 2:4].exp() * strides,
                    yolo_boxes[..., 4:],
                    strides,
                    anchor_points,
                ],
                dim=-1,
            )
            yolo_box_list.append(yolo_boxes)
        yolo_batch = torch.cat(yolo_box_list, dim=1)
        return yolo_batch

    def local_mask_branch(self, box_batch: list[Tensor], feats: list[Tensor]) -> Tensor:
        first_level_size = feats[0].shape[2:]
        merged_feat = torch.cat(
            [
                feat if idx == 0 else F.interpolate(feat, size=first_level_size)
                for idx, feat in enumerate(feats)
            ],
            dim=1,
        )
        roi_feats = roi_align(
            merged_feat,
            box_batch,
            self.mask_size,
        )
        local_masks = self.local_mask_head(roi_feats)
        return local_masks

    def feats(self, x: Tensor) -> list[Tensor]:
        feats = self.backbone(x)
        return self.neck(feats)

    def box_feats(self, x: list[Tensor]) -> list[Tensor]:
        return x[self.box_feat_range[0] : self.box_feat_range[1]]

    def mask_feats(self, x: list[Tensor]) -> list[Tensor]:
        return x[self.mask_feat_range[0] : self.mask_feat_range[1]]

    @torch.no_grad()
    def to_boxes(
        self, yolo_batch: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        score_batch, box_batch, lable_batch = [], [], []
        device = yolo_batch.device
        num_classes = self.num_classes
        batch = torch.zeros((*yolo_batch.shape[:2], 6)).to(device)
        batch[..., :4] = box_convert(
            yolo_batch[..., :4], in_fmt="cxcywh", out_fmt="xyxy"
        )
        batch[..., 4] = yolo_batch[..., 4].sigmoid()
        batch[..., 5] = yolo_batch[..., 5 : 5 + num_classes].argmax(-1)
        for r in batch:
            th_filter = r[..., 4] > self.score_threshold
            r = r[th_filter]
            boxes = r[..., :4]
            scores = r[..., 4]
            lables = r[..., 5].long()
            nms_index = batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=lables,
                iou_threshold=self.box_iou_threshold,
            )[: self.box_limit]
            box_batch.append(boxes[nms_index])
            score_batch.append(scores[nms_index])
            lable_batch.append(lables[nms_index])
        return score_batch, box_batch, lable_batch

    @torch.no_grad()
    def to_masks(
        self,
        all_local_masks: Tensor,
        score_batch: list[Tensor],
        box_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        mask_batch = []
        cur = 0
        device = all_local_masks.device
        all_local_masks = all_local_masks.sigmoid()
        out_box_batch, out_lable_batch, out_score_batch = [], [], []
        for boxes, scores, labels in zip(box_batch, score_batch, label_batch):
            # masks = torch.zeros(len(boxes), self.patch_size, self.patch_size).to(device)
            local_masks = all_local_masks[cur : cur + len(boxes)].squeeze(1)
            # mask_batch.append()
            cur += len(boxes)
            masks = torch.zeros(len(boxes), self.patch_size, self.patch_size).to(device)
            for i, (b, local_mask) in enumerate(
                zip(boxes.long().clip(min=0, max=self.patch_size - 1), local_masks)
            ):
                restored_mask = F.interpolate(
                    local_mask.view(1, 1, self.mask_size, self.mask_size),
                    size=(
                        b[3] - b[1] + 1,
                        b[2] - b[0] + 1,
                    ),
                )
                masks[i, b[1] : b[3] + 1, b[0] : b[2] + 1] = restored_mask.view(
                    1, *restored_mask.shape[2:]
                )
            masks = masks > self.mask_threshold
            empty_filter = masks.sum(dim=[1, 2]) > 0
            out_box_batch.append(boxes[empty_filter])
            mask_batch.append(masks[empty_filter])
            out_score_batch.append(scores[empty_filter])
            out_lable_batch.append(labels[empty_filter])
        return mask_batch, out_score_batch, out_box_batch, out_lable_batch

    def forward(
        self, image_batch: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        feats = self.feats(image_batch)
        box_feats = self.box_feats(feats)
        yolo_batch = self.box_branch(box_feats)
        score_batch, box_batch, label_batch = self.to_boxes(yolo_batch)
        mask_feats = self.mask_feats(feats)
        masks = self.local_mask_branch(box_batch, mask_feats)
        mask_batch, score_batch, box_batch, label_batch = self.to_masks(
            masks, score_batch, box_batch, label_batch
        )
        return score_batch, box_batch, label_batch, mask_batch


class Criterion:
    def __init__(
        self,
        model: MaskYolo,
        assign: SimOTA,
        obj_weight: float = 1.0,
        box_weight: float = 1.0,
        cate_weight: float = 1.0,
        local_mask_weight: float = 1.0,
        assign_topk: int = 9,
        assign_radius: float = 2.0,
        assign_center_wight: float = 1.0,
        local_mask_limit:int = 1,
    ) -> None:
        self.box_weight = box_weight
        self.cate_weight = cate_weight
        self.obj_weight = obj_weight
        self.local_mask_weight = local_mask_weight
        self.model = model
        self.strides = self.model.strides
        self.assign = assign

        self.box_loss = CIoULoss()
        self.obj_loss = F.binary_cross_entropy_with_logits
        self.local_mask_loss = F.binary_cross_entropy_with_logits
        self.cate_loss = F.binary_cross_entropy_with_logits
        self.local_mask_limit = local_mask_limit

    def __call__(
        self,
        inputs: tuple[Tensor],
        targets: tuple[list[Tensor], list[Tensor], list[Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (images,) = inputs  # list of [b, num_classes + 5, h, w]
        gt_mask_batch, gt_box_batch, gt_label_batch = targets
        device = images.device
        num_classes = self.model.num_classes
        feats = self.model.feats(images)
        box_feats = self.model.box_feats(feats)
        pred_yolo_batch = self.model.box_branch(box_feats)
        gt_yolo_batch, gt_local_mask_batch, pos_idx = self.prepeare_box_gt(
            gt_mask_batch, gt_box_batch, gt_label_batch, pred_yolo_batch
        )
        # filtered_gt_box_batch = gt_box_batch[:1]
        gt_local_mask_batch = self.prepeare_mask_gt(gt_mask_batch[:self.local_mask_limit], gt_box_batch[:self.local_mask_limit])

        # # 1-stage
        obj_loss = self.obj_loss(pred_yolo_batch[..., 4], gt_yolo_batch[..., 4])

        box_loss, cate_loss, local_mask_loss = (
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        matched_count = pos_idx.sum()
        if matched_count > 0:
            box_loss += self.box_loss(
                box_convert(
                    pred_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                box_convert(
                    gt_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
                ),
            )
            cate_loss += self.cate_loss(
                pred_yolo_batch[..., 5 : 5 + num_classes][pos_idx],
                gt_yolo_batch[..., 5 : 5 + num_classes][pos_idx],
            )

            # 2-stage
            pred_local_masks = self.model.local_mask_branch(
                gt_box_batch[:self.local_mask_limit],
                self.model.mask_feats(feats),
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

    @torch.no_grad()
    def prepeare_box_gt(
        self,
        gt_mask_batch: list[Tensor],
        gt_boxes_batch: list[Tensor],
        gt_label_batch: list[Tensor],
        pred_yolo_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = pred_yolo_batch.device
        num_classes = self.model.num_classes
        gt_yolo_batch = torch.zeros(
            pred_yolo_batch.shape,
            dtype=pred_yolo_batch.dtype,
            device=device,
        )
        mask_size = self.model.mask_size
        mask_stride = self.model.mask_stride

        gt_local_mask_list = []
        for batch_idx, (gt_masks, gt_boxes, gt_labels, pred_yolo) in enumerate(
            zip(gt_mask_batch, gt_boxes_batch, gt_label_batch, pred_yolo_batch)
        ):
            gt_cxcywh = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")
            matched = self.assign(
                gt_boxes=gt_boxes,
                pred_boxes=box_convert(
                    pred_yolo[..., :4], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                pred_scores=pred_yolo[..., 4].sigmoid(),
                strides=pred_yolo[..., 5 + num_classes],
                anchor_points=pred_yolo[..., 6 + num_classes : 6 + num_classes + 2],
            )
            gt_yolo_batch[batch_idx, matched[:, 1], :4] = gt_cxcywh[matched[:, 0]]
            gt_yolo_batch[batch_idx, matched[:, 1], 4] = 1.0
            gt_yolo_batch[batch_idx, matched[:, 1], 5 : 5 + num_classes] = F.one_hot(
                gt_labels[matched[:, 0]], num_classes
            ).to(gt_yolo_batch)
            gt_local_masks = roi_align(
                gt_masks.float().unsqueeze(1),
                list(gt_boxes.unsqueeze(1)),
                mask_size,
            )
            gt_local_mask_list.append(gt_local_masks)

        pos_idx = gt_yolo_batch[..., 4] == 1.0
        gt_local_mask_batch = torch.cat(gt_local_mask_list)
        return gt_yolo_batch, gt_local_mask_batch, pos_idx

    @torch.no_grad()
    def prepeare_mask_gt(
        self,
        gt_mask_batch: list[Tensor],
        gt_boxes_batch: list[Tensor],
    ) -> Tensor:
        device = gt_mask_batch[0].device
        mask_size = self.model.mask_size
        mask_stride = self.model.mask_stride
        gt_local_mask_list = []
        for gt_masks, gt_boxes in zip(gt_mask_batch, gt_boxes_batch):
            gt_local_masks = roi_align(
                gt_masks.float().unsqueeze(1),
                list(gt_boxes.unsqueeze(1)),
                mask_size,
            )
            gt_local_mask_list.append(gt_local_masks)
        gt_local_mask_batch = torch.cat(gt_local_mask_list)
        return gt_local_mask_batch


class TrainStep:
    def __init__(
        self,
        criterion: Criterion,
        optimizer: Any,
        use_amp: bool = True,
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler()

    def __call__(self, batch: Batch) -> dict[str, float]:
        self.criterion.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_box_batch, gt_label_batch = batch
            loss, obj_loss, box_loss, cate_loss, local_mask_loss = self.criterion(
                (images,),
                (
                    gt_mask_batch,
                    gt_box_batch,
                    gt_label_batch,
                ),
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return dict(
            loss=loss.item(),
            obj_loss=obj_loss.item(),
            box_loss=box_loss.item(),
            cate_loss=cate_loss.item(),
            local_mask_loss=local_mask_loss.item(),
        )


class ValidationStep:
    def __init__(
        self,
        criterion: Criterion,
    ) -> None:
        self.criterion = criterion
        self.model = criterion.model

    @torch.no_grad()
    def __call__(
        self,
        batch: Batch,
        on_end: Optional[Callable[[Any, Any], None]] = None,
    ) -> dict[str, float]:
        self.model.eval()
        images, gt_mask_batch, gt_box_batch, gt_label_batch = batch
        (
            pred_score_batch,
            pred_box_batch,
            pred_label_batch,
            pred_mask_batch,
        ) = self.model(images)
        if on_end is not None:
            on_end(
                pred_mask_batch,
                gt_mask_batch,
            )
        draw_save(
            "/app/test_outputs/gt.png",
            images[0],
            gt_mask_batch[0],
        )
        draw_save(
            "/app/test_outputs/pred.png",
            images[0],
            pred_mask_batch[0],
        )
        return dict()


class InferenceStep:
    def __init__(
        self,
        model: MaskYolo,
    ) -> None:
        self.model = model

    @torch.no_grad()
    def __call__(
        self, images: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        self.model.eval()
        (
            pred_score_batch,
            pred_box_batch,
            pred_label_batch,
            pred_mask_batch,
        ) = self.model(images)

        return pred_score_batch, pred_box_batch, pred_label_batch, pred_mask_batch
