"""Zero-initialized multiscale residual correction for atmospheric tendencies."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _SphericalConv2d(nn.Module):
    """A local convolution with periodic longitude and reflected latitude."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        values = F.pad(values, (1, 1, 0, 0), mode="circular")
        values = F.pad(values, (0, 0, 1, 1), mode="reflect")
        return self.convolution(values)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.first = _SphericalConv2d(channels, channels)
        self.first_norm = nn.GroupNorm(groups, channels)
        self.second = _SphericalConv2d(channels, channels)
        self.second_norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.activation(self.first_norm(self.first(values)))
        values = self.second_norm(self.second(values))
        return self.activation(values + residual)


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = _SphericalConv2d(
            in_channels, out_channels, stride=2
        )
        self.normalization = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(values)))


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = _SphericalConv2d(in_channels, out_channels)
        self.normalization = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.GELU()

    def forward(self, values: Tensor, skip: Tensor) -> Tensor:
        values = F.interpolate(
            values,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        values = torch.cat((values, skip), dim=1)
        return self.activation(self.normalization(self.convolution(values)))


class ResidualErrorCorrector(nn.Module):
    """Learn the residual error left by a frozen atmospheric backbone.

    The network is a compact two-level U-Net. Full-resolution skip connections
    preserve local detail, while quarter-resolution residual blocks provide
    multiscale capacity. The final projection is exactly zero-initialized, so
    an untrained corrector leaves the source forecast unchanged.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if min(input_channels, output_channels, hidden_channels, depth) < 1:
            raise ValueError("Corrector channel counts and depth must be positive")
        if hidden_channels % 8 != 0:
            raise ValueError("Corrector hidden_channels must be divisible by 8")

        width = hidden_channels
        self.input_projection = nn.Conv2d(input_channels, width, kernel_size=1)
        self.input_activation = nn.GELU()
        self.down_half = _Downsample(width, 2 * width)
        self.half_encoder = _ResidualBlock(2 * width)
        self.down_quarter = _Downsample(2 * width, 4 * width)
        self.bottleneck = nn.ModuleList(
            [_ResidualBlock(4 * width) for _ in range(depth)]
        )
        self.up_half = _Upsample(6 * width, 2 * width)
        self.half_decoder = _ResidualBlock(2 * width)
        self.up_full = _Upsample(3 * width, width)
        self.full_decoder = _ResidualBlock(width)
        self.output_projection = nn.Conv2d(width, output_channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, values: Tensor) -> Tensor:
        full = self.input_activation(self.input_projection(values))
        half = self.half_encoder(self.down_half(full))
        quarter = self.down_quarter(half)
        for block in self.bottleneck:
            quarter = block(quarter)
        half_decoded = self.half_decoder(self.up_half(quarter, half))
        full_decoded = self.full_decoder(self.up_full(half_decoded, full))
        return self.output_projection(full_decoded)
