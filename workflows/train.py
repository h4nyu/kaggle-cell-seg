import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger, FileHandler

# import torch_optimizer as optim
import torch.optim as optim
from cellseg.solo import (
    Solo,
    TrainStep,
    Criterion,
    ValidationStep,
    ToMasks,
    BatchAdaptor,
)
from cellseg.metrics import MaskAP
from cellseg.backbones import EfficientNetFPN
from cellseg.utils import seed_everything, Checkpoint, MeanReduceDict, ToDevice
from cellseg.data import (
    get_fold_indices,
    CellTrainDataset,
    collate_fn,
    TrainTranform,
    Tranform,
)
from torch.utils.data import Subset, DataLoader
from pathlib import Path


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    Path(os.path.join(cfg.data.root_path, f"{cfg.name}")).mkdir(exist_ok=True)
    logger.addHandler(FileHandler(os.path.join(cfg.root_path, cfg.name, "train.log")))
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[Solo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=float("inf"),
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
    to_masks = ToMasks(**cfg.to_masks)
    train_step = TrainStep(
        optimizer=optim.Adam(model.parameters(), **cfg.optimizer),
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
        use_amp=cfg.use_amp,
        to_masks=to_masks,
    )
    validation_step = ValidationStep(
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
        to_masks=to_masks,
        use_amp=cfg.use_amp,
    )
    train_dataset = CellTrainDataset(
        **cfg.dataset,
        transform=TrainTranform(
            original_size=cfg.original_size,
        ),
    )
    val_dataset = CellTrainDataset(
        **cfg.dataset,
        transform=Tranform(
            original_size=cfg.original_size,
        ),
    )
    train_indecies, validation_indecies = get_fold_indices(train_dataset, **cfg.fold)
    train_loader = DataLoader(
        Subset(train_dataset, train_indecies), collate_fn=collate_fn, **cfg.train_loader
    )
    val_loader = DataLoader(
        Subset(val_dataset, validation_indecies),
        collate_fn=collate_fn,
        **cfg.validation_loader,
    )
    to_device = ToDevice(cfg.device)

    for _ in range(cfg.num_epochs):
        train_reduer = MeanReduceDict(keys=cfg.log_keys)
        for batch in train_loader:
            batch = to_device(*batch)
            train_log = train_step(batch)
            train_reduer.accumulate(train_log)
            logger.info(f"train batch {train_log} ")
        logger.info(f"train {train_reduer.value} ")
        val_reduer = MeanReduceDict(keys=cfg.log_keys)
        mask_ap = MaskAP(**cfg.mask_ap)
        for batch in val_loader:
            batch = to_device(*batch)
            validation_log = validation_step(batch)
            val_reduer.accumulate(validation_log)
        if score > val_reduer.value["loss"]:
            score = checkpoint.save(model, val_reduer.value["loss"])
            logger.info(f"save checkpoint")
        logger.info(f"eval {score=} {val_reduer.value} {mask_ap.value=}")


if __name__ == "__main__":
    main()
