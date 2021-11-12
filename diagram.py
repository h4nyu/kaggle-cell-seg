from diagrams import Diagram, Cluster, Edge
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
    gt_masks = Tensor3D("gt_masks")
    gt_centerness = Tensor1D("gt_centerness")
    gt_labels = Tensor1D("gt_labels")
    loss = Scalar("loss")

    mask_index = Tensor1D("mask_index")

    image = Tensor3D("image")
    gt_grid = CategoryGrid("gt_grid")

    category_loss = Scalar("category_loss")
    mask_loss = Scalar("mask_loss")
    model = FPN("model")

    pred_grid = CategoryGrid("pred_grid")
    pred_all_masks = InsntanceMasks("pred_all_masks")

    gt_labels >> gt_grid
    gt_masks >> gt_centerness >> gt_grid
    image >> model >> pred_grid
    model >> pred_all_masks
    gt_centerness >> mask_index
    [pred_all_masks, gt_masks, mask_index] >> mask_loss
    [pred_grid, gt_grid] >> category_loss
    [mask_loss, category_loss] >> loss
