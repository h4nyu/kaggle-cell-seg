import torch
import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
import torch_optimizer as optim
from cellseg.yolox import (
    MaskYolo,
    Criterion,
    TrainStep,
    ValidationStep,
    InferenceStep,
)
from cellseg.metrics import MaskAP
from cellseg.backbones import EfficientNetFPN
from cellseg.utils import (
    seed_everything,
    Checkpoint,
    MeanReduceDict,
    ToDevice,
    draw_save,
)
from cellseg.data import (
    get_fold_indices,
    CellTrainDataset,
    CollateFn,
    Tranform,
)
from cellseg.necks import CSPNeck
from torch.utils.data import Subset, DataLoader
from pathlib import Path


@hydra.main(config_path="/app/config", config_name="mask_yolo")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[MaskYolo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=float("inf"),
    )
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=[cfg.hidden_channels for _ in backbone.out_channels],
        strides=backbone.strides,
    )
    model = MaskYolo(
        backbone=backbone,
        neck=neck,
        mask_size=cfg.mask_size,
        box_feat_range=cfg.box_feat_range,
        mask_feat_range=cfg.mask_feat_range,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
        score_threshold=cfg.score_threshold,
        mask_threshold=cfg.mask_threshold,
        box_iou_threshold=cfg.box_iou_threshold,
    )
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    inference_step = InferenceStep(
        model=model,
    )
    to_device = ToDevice(cfg.device)
    dataset = CellTrainDataset(
        **cfg.dataset, transform=Tranform(cfg.patch_size, use_patch=cfg.use_patch)
    )
    # dataset = CellTrainDataset(**cfg.dataset)
    collate_fn = CollateFn()
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
    )

    idx = 0
    mask_ap = MaskAP()
    for batch in loader:
        batch = to_device(**batch)
        images = batch["images"]
        gt_mask_batch = batch["mask_batch"]
        gt_box_batch = batch["box_batch"]
        gt_label_batch = batch["label_batch"]
        _, _, _, pred_mask_batch = inference_step(images)
        for image, masks, gt_masks in zip(images, pred_mask_batch, gt_mask_batch):
            pred_count = masks.shape[0]
            gt_count = gt_masks.shape[0]
            if gt_count == 0:
                continue
            score = mask_ap.accumulate(masks, gt_masks)
            logger.info(f"{idx=} {pred_count=} {gt_count=} {score=}")
            # logger.info(f"{idx=} {pred_count=} {gt_count=}")
            draw_save(
                os.path.join("/store", cfg.name, f"{idx}_pred.png"),
                image,
                masks,
            )
            draw_save(
                os.path.join("/store", cfg.name, f"{idx}_gt.png"),
                image,
                gt_masks,
            )
            idx += 1
    score = mask_ap.value
    logger.info(f"{score=}")


if __name__ == "__main__":
    main()
