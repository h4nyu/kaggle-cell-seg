import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
import torch_optimizer as optim
from cellseg.solo import Solo, TrainStep, Criterion, ValidationStep, ToMasks
from cellseg.solo.adaptors import BatchAdaptor
from cellseg.metrics import MaskAP
from cellseg.backbones import EfficientNetFPN
from cellseg.util import seed_everything, Checkpoint, MeanReduceDict
from cellseg.data import (
    ToDevice,
    get_fold_indices,
    CellTrainDataset,
    collate_fn,
    Tranform,
)
from torch.utils.data import Subset, DataLoader


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[Solo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=0,
    )
    model = Solo(**cfg.model, backbone=backbone)
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    criterion = Criterion(**cfg.criterion)
    batch_adaptor = BatchAdaptor(
        num_classes=cfg.num_classes,
        grid_size=cfg.model.grid_size,
        original_size=cfg.original_size,
    )
    train_step = TrainStep(
        optimizer=optim.AdaBound(model.parameters(), **cfg.optimizer),
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
        use_amp=cfg.use_amp,
    )
    to_masks = ToMasks(**cfg.to_masks)
    validation_step = ValidationStep(
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
        to_masks=to_masks,
        use_amp=cfg.use_amp,
    )
    dataset = CellTrainDataset(
        **cfg.dataset,
        transform=Tranform(
            original_size=cfg.original_size,
        ),
    )
    train_indecies, validation_indecies = get_fold_indices(dataset, **cfg.fold)
    train_len = len(train_indecies) // cfg.train_loader.batch_size
    validation_len = len(validation_indecies) // cfg.validation_loader.batch_size
    train_loader = DataLoader(
        Subset(dataset, train_indecies), collate_fn=collate_fn, **cfg.train_loader
    )
    val_loader = DataLoader(
        Subset(dataset, validation_indecies),
        collate_fn=collate_fn,
        **cfg.validation_loader,
    )
    to_device = ToDevice(cfg.device)

    for epoch in range(cfg.num_epochs):
        train_reduer = MeanReduceDict(keys=cfg.log_keys)
        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(*batch)
            train_log = train_step(batch)
            train_reduer.accumulate(train_log)

        logger.info(f"{epoch=} {train_reduer.value=} ")
        val_reduer = MeanReduceDict(keys=cfg.log_keys)
        mask_ap = MaskAP(**cfg.mask_ap)
        for batch in val_loader:
            batch = to_device(*batch)
            validation_log = validation_step(batch, on_end=mask_ap.accumulate_batch)
            val_reduer.accumulate(validation_log)

        if score < mask_ap.value:
            score = checkpoint.save(model, mask_ap.value)
            logger.info(f"new {score=} updated model!!")
        logger.info(f"{epoch=} score={mask_ap.value} {val_reduer.value=}")


if __name__ == "__main__":
    main()
