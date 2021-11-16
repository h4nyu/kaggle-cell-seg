from cellseg.dataset import CellTrainDataset, Tranform
from hydra import compose, initialize
from cellseg.data import draw_save

initialize(config_path="../config")
cfg = compose(config_name="config")


def test_cell_dataset() -> None:
    transform = Tranform(original_size=cfg.original_size)
    dataset = CellTrainDataset(
        transform=transform,
    )
    assert len(dataset) == 606
    sample = dataset[0]
    assert sample is not None
    image = sample["image"]
    masks = sample["masks"]
    labels = sample["labels"]
    draw_save("/store/test-dataset.png", image, masks)
    assert image.shape == (3, cfg.original_size, cfg.original_size)
    assert image.shape[1:] == masks.shape[1:]
    assert labels.shape[0] == masks.shape[0]
