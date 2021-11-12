from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

Tensor1D = lambda x: Custom(x, "./img/1d.png")
Tensor2D = lambda x: Custom(x, "./img/2d.png")
Tensor3D = lambda x: Custom(x, "./img/3d.png")
Tensor3D = lambda x: Custom(x, "./img/3d.png")
Head = lambda x: Custom(x, "./img/head.png")
InsntanceMasks = lambda x: Custom(x, "./img/instance-masks.png")
CategoryGrid = lambda x: Custom(x, "./img/cagetory-grid.png")
FPN = lambda x: Custom(x, "./img/fpn.png")

Scalar = lambda x: Custom(x, "./img/scalar.png")
with Diagram("train", show=False):
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


with Diagram("model", show=False):
    image = Tensor3D("image")
    fpn = FPN("FPN")
    all_masks = InsntanceMasks("all_masks")
    cagetory_grid = CategoryGrid("grid")
    cagetory_head = Head("cagetory_head")
    mask_head = Head("mask_head")


    image >> fpn >> cagetory_head >> cagetory_grid
    fpn >> mask_head >> all_masks
