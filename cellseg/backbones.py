import torch.nn as nn
from torch import Tensor
from efficientnet_pytorch import EfficientNet


class EfficientNetFPN(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b1")

    def forward(self, images: Tensor) -> list[Tensor]:  # P1 - P6, P7 is dropped
        features = self.net.extract_endpoints(images)
        return [images, *features.values()][:6]
