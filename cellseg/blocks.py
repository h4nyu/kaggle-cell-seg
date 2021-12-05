import torch
import torch.nn as nn
from torch import Tensor
from fvcore.nn import weight_init
import torch.nn.functional as F
from typing import Callable, Optional, Any

DefaultActivation = nn.Mish(inplace=True)
BN_PARAMS = dict(eps=1e-3, momentum=3e-2)


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act: Optional[Callable[[Tensor], Tensor]] = nn.Mish(inplace=True),
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm = nn.BatchNorm2d(
            num_features=out_channels,
        )
        self.act = act

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.conv(feat)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        act: Optional[Callable[[Tensor], Tensor]] = nn.Mish(inplace=True),
    ):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 1, act=act),
            ConvBnAct(in_channels, in_channels, 3, act=act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x) + x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.cat(
            [x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1
        )  # [b, 2, 1, 1]
        return x * self.conv(h).sigmoid()

    def _init_weights(self) -> None:
        weight_init.c2_msra_fill(self.conv)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, kernel_sizes: list[int] = [5, 9, 13]) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x] + [pool(x) for pool in self.pools], dim=1)


class ReversedCSP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        act: Callable = DefaultActivation,
        eps: float = 1e-3,
        momentum: float = 3e-2,
    ):
        super().__init__()
        self.in_conv = ConvBnAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )
        main_conv = [
            nn.Sequential(
                ConvBnAct(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    act=act,
                ),
                ConvBnAct(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    act=act,
                ),
            )
            for _ in range(depth)
        ]
        self.main_conv = nn.Sequential(*main_conv)
        self.bypass_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(2 * out_channels, eps=eps, momentum=momentum)
        self.act = act
        self.out_conv = ConvBnAct(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        h1 = self.main_conv(x)
        h2 = self.bypass_conv(x)
        h = self.act(self.bn(torch.cat([h1, h2], dim=1)))
        return self.out_conv(h)


class CSPUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        depth: int,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ) -> None:
        super().__init__()
        self.in_conv1 = ConvBnAct(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )
        self.in_conv2 = ConvBnAct(
            in_channels=in_channels[1],
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )
        self.rcsp = ReversedCSP(2 * out_channels, out_channels, depth, act=act)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tensor:
        h1 = F.interpolate(self.in_conv1(x1), scale_factor=2, mode="nearest")
        h2 = self.in_conv2(x2)
        h = torch.cat([h2, h1], dim=1)
        return self.rcsp(h)


class CSPPlainBlock(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        depth: int,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ) -> None:
        super().__init__()
        self.in_conv1 = ConvBnAct(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )
        self.in_conv2 = ConvBnAct(
            in_channels=in_channels[1],
            out_channels=out_channels,
            kernel_size=1,
            act=act,
        )
        self.rcsp = ReversedCSP(2 * out_channels, out_channels, depth, act=act)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tensor:
        h1 = self.in_conv1(x1)
        h2 = self.in_conv2(x2)
        h = torch.cat([h2, h1], dim=1)
        return self.rcsp(h)


class CSPDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        depth: int,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ):
        super().__init__()
        self.in_conv = ConvBnAct(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            act=act,
        )
        self.rcsp = ReversedCSP(
            out_channels + in_channels[1], out_channels, depth, act=act
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.in_conv(x1)
        h = torch.cat([x1, x2], dim=1)
        return self.rcsp(h)


class CSPDarkBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.in_conv = ConvBnAct(in_channels, out_channels, 3, stride=2, act=act)
        self.csp = CSPBlock(out_channels, out_channels, depth, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.csp(self.in_conv(x))


class CSPSPPBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, act: Callable = DefaultActivation
    ):
        super().__init__()
        self.main_conv = nn.Sequential(
            ConvBnAct(in_channels, out_channels, 1, act=act),
            ConvBnAct(out_channels, out_channels, 3, act=act),
            ConvBnAct(out_channels, out_channels, 1, act=act),
            SpatialPyramidPooling(),
            ConvBnAct(4 * out_channels, out_channels, 1, act=act),
            ConvBnAct(out_channels, out_channels, 3, act=act),
        )
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(2 * out_channels, **BN_PARAMS)
        self.act = act
        self.out_conv = ConvBnAct(2 * out_channels, out_channels, 1, act=act)

    def forward(self, x: Tensor) -> None:
        h1 = self.main_conv(x)
        h2 = self.bypass_conv(x)
        h = self.act(self.bn(torch.cat([h1, h2], dim=1)))
        return self.out_conv(h)


class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        main_conv: list[Any] = [ConvBnAct(in_channels, out_channels // 2, 1, act=act)]
        main_conv += [ResBlock(out_channels // 2, act=act) for _ in range(depth)]
        main_conv += [nn.Conv2d(out_channels // 2, out_channels // 2, 1, bias=False)]
        self.main_conv = nn.Sequential(*main_conv)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels // 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, **BN_PARAMS)
        self.act = act
        self.out_conv = ConvBnAct(out_channels, out_channels, 1, act=act)

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.main_conv(x)
        h2 = self.bypass_conv(x)
        h = self.act(self.bn(torch.cat([h1, h2], dim=1)))
        return self.out_conv(h)


class DarkBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        assert depth == 1
        self.in_conv = ConvBnAct(in_channels, out_channels, 3, 2, act=act)
        self.res = ResBlock(out_channels, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(self.in_conv(x))
