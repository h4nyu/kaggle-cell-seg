from diagrams import Diagram
from diagrams import Cluster
from diagrams.custom import Custom

Tensor1D = lambda x: Custom(x, "./img/1d.png")
Tensor2D = lambda x: Custom(x, "./img/2d.png")
Tensor3D = lambda x: Custom(x, "./img/3d.png")
Tensor3D = lambda x: Custom(x, "./img/3d.png")
InsntanceMasks = lambda x: Custom(x, "./img/instance-masks.png")
CategoryGrid = lambda x: Custom(x, "./img/cagetory-grid.png")
FPN = lambda x: Custom(x, "./img/fpn.png")

Scalar = lambda x: Custom(x, "./img/scalar.png")
with Diagram("Train", show=False):
    gt_boxes = Tensor1D("gt_boxes")
    gt_masks = Tensor3D("gt_masks")
    gt_centers = Tensor1D("gt_centers")
    gt_labels = Tensor1D("gt_labels")
    loss = Scalar("loss")

    image = Tensor3D("image")
    gt_grid = CategoryGrid("gt_grid")

    gt_all_masks = InsntanceMasks("gt_all_masks")

    category_loss = Scalar("category_loss")
    mask_loss = Scalar("mask_loss")
    model = FPN("model")

    pred_grid = CategoryGrid("pred_grid")
    pred_all_masks = InsntanceMasks("pred_all_masks")

    gt_labels >> gt_grid
    gt_masks >> gt_boxes >> gt_centers >> gt_grid
    [gt_centers, gt_masks] >> gt_all_masks
    image >> model >> pred_grid
    model >> pred_all_masks
    [pred_all_masks, gt_all_masks] >> mask_loss
    [pred_grid, gt_grid] >> category_loss
    [mask_loss, category_loss] >> loss
