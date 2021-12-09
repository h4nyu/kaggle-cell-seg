import pytest
import torch
from torch import Tensor
from cellseg.yolox import MaskYolo, Criterion
from torchvision.ops import masks_to_boxes
from cellseg.backbones import EfficientNetFPN
from cellseg.necks import CSPNeck
from cellseg.utils import draw_save, seed_everything
from cellseg.assign import SimOTA
from torchvision.ops import box_convert
from cellseg.data import CellTrainDataset, Tranform, TrainItem
from pathlib import Path
from hydra import compose, initialize

initialize(config_path="../config")
cfg = compose(config_name="mask_yolo")
has_data = Path(cfg.data.train_file_path).exists()
seed_everything(cfg.seed)


@pytest.fixture
def mask_yolo() -> MaskYolo:
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 2
    mask_size = 16
    box_feat_range = (2, 7)
    mask_feat_range = (0, 3)
    patch_size = 128
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=backbone.out_channels,
        strides=backbone.strides,
    )
    return MaskYolo(
        backbone=backbone,
        neck=neck,
        mask_size=mask_size,
        num_classes=num_classes,
        box_feat_range=box_feat_range,
        mask_feat_range=mask_feat_range,
        patch_size=patch_size,
        box_iou_threshold=0.1,
        score_threshold=0.0,
        mask_threshold=0.0,
    )


@pytest.fixture
def assign() -> SimOTA:
    return SimOTA(topk=10)


@pytest.fixture
def sample() -> TrainItem:
    transform = Tranform(size=cfg.patch_size)
    dataset = CellTrainDataset(
        transform=transform,
        **cfg.dataset,
    )
    sample = dataset[1]
    assert sample is not None
    return sample


@pytest.fixture
def targets() -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    masks0 = torch.zeros(1, 128, 128, dtype=torch.bool)
    masks0[0, 10:20, 30:40] = True
    masks1 = torch.zeros(2, 128, 128, dtype=torch.bool)
    masks1[0, 10:20, 30:40] = True
    masks1[1, 20:30, 30:40] = True
    mask_batch = [masks0, masks1]
    box_batch = [masks_to_boxes(m) for m in mask_batch]
    label_batch = [torch.zeros(len(m)).long() for m in mask_batch]
    return mask_batch, box_batch, label_batch


def test_mask_yolo(mask_yolo: MaskYolo) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    mask_batch = mask_yolo(images)


def test_mask_yolo_box_branch(mask_yolo: MaskYolo) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    feats = mask_yolo.feats(images)
    box_feats = mask_yolo.box_feats(feats)
    yolo_batch = mask_yolo.box_branch(box_feats)
    assert yolo_batch.shape[0] == 2
    assert yolo_batch.shape[2] == 5 + mask_yolo.num_classes + 3


def test_mask_yolo_local_mask_branch(mask_yolo: MaskYolo) -> None:
    out_channels = mask_yolo.backbone.out_channels
    strides = mask_yolo.strides
    image_size = 256
    image_batch = torch.rand(2, 3, image_size, image_size)
    box_batch = [
        torch.tensor(
            [
                [10, 20, 30, 40],
                [10, 20, 30, 40],
            ],
        ).float(),
        torch.tensor(
            [
                [10, 20, 30, 40],
            ],
        ).float(),
    ]
    feats = mask_yolo.feats(image_batch)
    mask_feats = mask_yolo.mask_feats(feats)
    masks = mask_yolo.local_mask_branch(box_batch, mask_feats)


def test_criterion(
    mask_yolo: MaskYolo,
    assign: SimOTA,
    targets: tuple[list[Tensor], list[Tensor], list[Tensor]],
) -> None:
    criterion = Criterion(model=mask_yolo, assign=assign)
    images = torch.rand(2, 3, 128, 128)
    criterion(inputs=(images,), targets=targets)


def test_forward(
    mask_yolo: MaskYolo, targets: tuple[list[Tensor], list[Tensor], list[Tensor]]
) -> None:
    images = torch.rand(1, 3, 256, 256)
    mask_yolo.score_threshold = 0.0
    score_batch, lable_batch, box_batch, mask_batch = mask_yolo(images)
    draw_save(
        "/app/test_outputs/yolox-forward.png",
        images[0],
        mask_batch[0][:10],
    )


def test_to_boxes(
    mask_yolo: MaskYolo,
) -> None:
    yolo_batch = torch.zeros((1, 1, 7))
    yolo_batch[0, 0, :2] = torch.tensor([15.0, 30.0])
    yolo_batch[0, 0, 2:4] = torch.tensor([10.0, 20.0])
    yolo_batch[0, 0, 4] = 0.9
    yolo_batch[0, 0, 5:] = torch.tensor([10.0, 20.0])
    score_batch, box_batch, lable_batch = mask_yolo.to_boxes(yolo_batch)
    assert len(score_batch) == len(box_batch) == len(lable_batch)
    assert score_batch[0][0] == 0.9
    assert box_batch[0][0].tolist() == [10.0, 20.0, 20.0, 40.0]
    assert lable_batch[0][0] == 1


@pytest.mark.skipif(not has_data, reason="no data volume")
def test_assign(sample: TrainItem, mask_yolo: MaskYolo, assign: SimOTA) -> None:
    limit = 10
    gt_box_batch = [sample["boxes"][:1]]
    gt_mask_batch = [sample["masks"]]
    gt_label_batch = [sample["labels"]]
    images = sample["image"].unsqueeze(0)
    feats = mask_yolo.feats(images)
    box_feats = mask_yolo.box_feats(feats)
    pred_yolo_batch = mask_yolo.box_branch(box_feats)
    num_classes = mask_yolo.num_classes
    criterion = Criterion(model=mask_yolo, assign=assign)

    gt_yolo_batch, gt_local_mask_batch, pos_idx = criterion.prepeare_gt(
        gt_mask_batch, gt_box_batch, gt_label_batch, pred_yolo_batch
    )
    pos_idx = pos_idx
    draw_save(
        "/app/test_outputs/yolox-assign-anchor.png",
        images[0],
        boxes=box_convert(
            pred_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
        ),
    )
    draw_save(
        "/app/test_outputs/yolox-assign-gt.png",
        images[0],
        boxes=box_convert(
            gt_yolo_batch[..., :4][pos_idx], in_fmt="cxcywh", out_fmt="xyxy"
        ),
    )
