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

with Diagram("inference", show=False):
    model = FPN("model")
    image = Tensor3D("image")
    mask_index = Tensor1D("mask_index")

    cagetory_grid = CategoryGrid("pred_grid")
    all_masks = InsntanceMasks("pred_all_masks")
    final_masks = Tensor3D("final_masks")

    image >> model >> [cagetory_grid, all_masks]
    cagetory_grid >> mask_index >> final_masks
    all_masks >> final_masks


with Diagram("model", show=False):
    image = Tensor3D("image 512")
    all_masks = InsntanceMasks("all_masks 512")
    cagetory_grid = CategoryGrid("cagetory_grid 64")
    cagetory_head = Head("cagetory_head")
    mask_head = Head("mask_head")
    p1 = Tensor3D("P1 512")
    p2 = Tensor3D("P2 256")
    p3 = Tensor3D("P3 128")
    p4 = Tensor3D("P4 64")
    p5 = Tensor3D("P5 32")
    p6 = Tensor3D("P6 16")
    p7 = Tensor3D("P7 8")

    image >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7
    [p1, p2, p3, p4, p5] >> mask_head >> all_masks
    [p4, p5, p6, p7] >> cagetory_head >> cagetory_grid

    # image >>  >> cagetory_head >> cagetory_grid
    # fpn >> mask_head >> all_masks
