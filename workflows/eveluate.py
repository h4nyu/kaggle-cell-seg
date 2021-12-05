import torch
import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
import torch_optimizer as optim
from cellseg.solo import (
    Solo,
    TrainStep,
    Criterion,
    ValidationStep,
    ToMasks,
    PatchInferenceStep,
    InferenceStep,
    BatchAdaptor,
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
    collate_fn,
    Tranform,
)
from cellseg.necks import CSPNeck
from torch.utils.data import Subset, DataLoader
from pathlib import Path


@hydra.main(config_path="/app/config", config_name="solo")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[Solo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=float("inf"),
    )
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=backbone.out_channels,
        strides=backbone.strides,
    )
    model = Solo(**cfg.model, backbone=backbone, neck=neck)
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    batch_adaptor = BatchAdaptor(
        num_classes=cfg.num_classes,
        grid_size=cfg.model.grid_size,
        patch_size=cfg.patch_size,
    )
    to_masks = ToMasks(**cfg.to_masks, patch_size=cfg.patch_size)
    inference_step = PatchInferenceStep(
        model=model,
        to_masks=to_masks,
        batch_adaptor=batch_adaptor,
        use_amp=cfg.use_amp,
        patch_size=cfg.patch_size,
    )
    to_device = ToDevice(cfg.device)
    dataset = CellTrainDataset(
        **cfg.dataset, transform=Tranform(cfg.patch_size, use_patch=cfg.use_patch)
    )
    # dataset = CellTrainDataset(**cfg.dataset)
    loader = DataLoader(
        Subset(dataset, indices=list(range(10))),
        collate_fn=collate_fn,
        batch_size=1,
    )

    idx = 0
    mask_ap = MaskAP()
    for batch in loader:
        batch = to_device(*batch)
        images, gt_mask_batch, _ = batch
        mask_batch, _ = inference_step(images)
        for image, masks, gt_masks in zip(images, mask_batch, gt_mask_batch):
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
