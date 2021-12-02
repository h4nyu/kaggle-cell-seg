# import torch
# from torch import Tensor
# from typing import Callable, Any
# import torch.nn as nn
# import torch.nn.functional as F
# from .blocks import ConvBnAct, DefaultActivation
# from .backbones import FPNLike
# from .necks import NeckLike
# from .heads import MaskHead
# from .utils import grid_points
# from torchvision.ops import roi_align, box_convert
# from cellseg.utils import round_to
# import math


# class DecoupledHeadUnit(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         width_mult: float = 1.0,
#         num_classes: int = 80,
#         act: Callable[[Tensor], Tensor] = DefaultActivation,
#     ):
#         super().__init__()
#         hidden_channels = round_to(256 * width_mult)
#         self.stem_conv = ConvBnAct(
#             in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
#         )
#         self.reg_conv = nn.Sequential(
#             ConvBnAct(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=3,
#                 act=act,
#             ),
#             ConvBnAct(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=3,
#                 act=act,
#             ),
#         )
#         self.reg_out_conv = nn.Conv2d(
#             in_channels=hidden_channels, out_channels=4, kernel_size=1
#         )
#         self.obj_out_conv = nn.Conv2d(
#             in_channels=hidden_channels, out_channels=1, kernel_size=1
#         )
#         self.cls_conv = nn.Sequential(
#             ConvBnAct(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=3,
#                 act=act,
#             ),
#             ConvBnAct(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=3,
#                 act=act,
#             ),
#         )
#         self.cls_out_conv = nn.Conv2d(hidden_channels, num_classes, 1)
#         self._init_weights()

#     def _init_weights(self, prior_prob: Any = 1e-2) -> None:
#         for m in [self.cls_out_conv, self.obj_out_conv]:
#             nn.init.constant_(m.bias, -math.log((1 - prior_prob) / prior_prob))  # type: ignore

#     def forward(self, x: Tensor) -> Tensor:
#         h = self.stem_conv(x)
#         h1 = self.reg_conv(h)
#         y_reg = self.reg_out_conv(h1)
#         y_obj = self.obj_out_conv(h1)
#         y_cls = self.cls_out_conv(self.cls_conv(h))
#         return torch.cat([y_reg, y_obj, y_cls], dim=1)


# class YoloxHead(nn.Module):
#     def __init__(
#         self,
#         in_channels: list[int] = [256, 512, 1024],
#         num_classes: int = 80,
#         width_mult: float = 1.0,
#         act: Callable[[Tensor], Tensor] = DefaultActivation,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.heads = nn.ModuleList(
#             [
#                 DecoupledHeadUnit(
#                     in_channels=in_chs,
#                     num_classes=num_classes,
#                     width_mult=width_mult,
#                     act=act,
#                 )
#                 for in_chs in in_channels
#             ]
#         )

#     def forward(
#         self, inputs: list[Tensor]
#     ) -> list[Tensor]:  # list[b, num_classes + 5, h, w]
#         return [m(x) for m, x in zip(self.heads, inputs)]


# class MaskYolo(nn.Module):
#     def __init__(
#         self,
#         backbone: FPNLike,
#         neck: NeckLike,
#         num_classes: int,
#         mask_size: int,
#         top_fpn_level: int,
#     ) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.neck = neck
#         self.mask_size = mask_size
#         self.top_fpn_level = top_fpn_level
#         self.reductions = self.neck.reductions
#         self.box_head = YoloxHead(
#             in_channels=neck.out_channels, num_classes=num_classes
#         )
#         self.mask_head = MaskHead(in_channels=sum(neck.out_channels), out_channels=1)

#     def box_branch(self, feats: list[Tensor]) -> Tensor:
#         return self.box_head(feats)

#     def mask_branch(self, box_batch: list[Tensor], feats: list[Tensor]) -> list[Tensor]:
#         first_level_size = feats[0].shape[2:]
#         merged_feats = torch.cat(
#             [
#                 feat if idx == 0 else F.interpolate(feat, size=first_level_size)
#                 for idx, feat in enumerate(feats)
#             ],
#             dim=1,
#         )
#         roi_feats = roi_align(
#             merged_feats,
#             box_batch,
#             self.mask_size,
#         )

#         return self.box_head(feats)

#     def features(self, x: Tensor) -> list[Tensor]:
#         feats = self.backbone(x)[: self.top_fpn_level]
#         return self.neck(feats)

#     # def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
#     #     feats = self.features(x)
#     #     box_preds = self.box_branch(feats)
#     #     box_batch = self.to_boxes(box_preds)
#     #     return box_preds, feats


# class Criterion:
#     def __init__(
#         self,
#         model: MaskYolo,
#         loc_weight: float = 1.0,
#         cate_weight: float = 1.0,
#         obj_weight: float = 1.0,
#         local_mask_weight: float = 1.0,
#         global_mask_weight: float = 1.0,
#     ) -> None:
#         self.loc_weight = loc_weight
#         self.cate_weight = cate_weight
#         self.obj_weight = obj_weight
#         self.local_mask_weight = local_mask_weight
#         self.global_mask_weight = global_mask_weight
#         self.model = model

#     def __call__(
#         self,
#         inputs: tuple[Tensor],
#         targets: tuple[list[Tensor], list[Tensor]],
#     ) -> None:
#         images, = inputs  # list of [b, num_classes + 5, h, w]
#         gt_mask_batch, label_batch = targets
#         device = images.device
#         feats = self.model.features(images)
#         box_preds = self.model.box_branch(feats)
#         preds = self.prepeare_preds(box_preds)

#     # def grids(self) -> Tensor:

#     def prepeare_preds(self, box_preds:list[Tensor]) -> Tensor:

#         for box_pred, stride in zip(box_preds, self.model.reductions):
#             points = grid_points(rows, cols).to(device)
#             a = self.to_boxes(
#                 preds_batch_,
#                 grid_points,
#                 stride
#             )

#     # @torch.no_grad()
#     # def encode_gt(self, gt_box_batch:list[Tensor], gt_label_batch:list[Tensor], pred_box_batch:Tensor, strides:list[int]) -> Tensor:
#     #     device = pred_box_batch.device
#     #     batch_size, num_candidate, num_outputs = pred_box_batch.shape
#     #     num_classes = num_outputs - 5
#     #     coded = torch.zeros(pred_box_batch.shape, dtype=torch.float, device=device)
#     #     for gt_boxes, labels in zip(gt_boxes_batch, labels_batch):
#     #         gt_boxes_cxcywh = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")
#     #         preds = preds_batch[batch_idx]
#     #         assignment = self.get_assignment(gt_boxes, labels, preds, grid, strides)
#     #         mask = assignment >= 0
#     #         assigned_idx = assignment[mask]
#     #         coded[batch_idx, mask, :4] = gt_boxes_ctf[assigned_idx]  # loc
#     #         coded[batch_idx, mask, 4] = 1.0  # obj
#     #         coded[batch_idx, mask, 5:] = F.one_hot(labels[assigned_idx], num_classes).to(coded)  # cls
#     #     return coded

#     def to_boxes(self, pred_box_batch:Tensor, grid_points, stride:int) -> Tensor:
#         batch_size, num_outputs, rows, cols = pred.shape
#         pred = pred.permute(0, 2, 3, 1).reshape(batch_size, rows * cols, num_outputs).float()
#         return torch.cat(
#             [
#                 (pred[..., 0:2] + grid) * stride,  # cxcy
#                 pred[..., 2:4].exp() * stride,  # wh
#                 pred[..., 4:]
#             ],
#             dim=-1
#         ).reshape(batch_size, rows * cols, num_outputs)
