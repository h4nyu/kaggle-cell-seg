import torch.nn as nn
import torch


class CoordConv(nn.Module):
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x_range = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
        y_range = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.expand([feat.shape[0], 1, -1, -1])
        x = x.expand([feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([feat, x, y], 1)
        return coord_feat
