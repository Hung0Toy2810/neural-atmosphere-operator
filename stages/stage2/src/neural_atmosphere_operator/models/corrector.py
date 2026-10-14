"""Geometry-aware spherical correction of standardized forecast tendencies."""

from __future__ import annotations

import inspect
import math
from importlib import import_module
from typing import Any, cast

import torch
from torch import Tensor, nn

from .transforms import stabilize_sht_constants


_sfno_module = import_module("torch_harmonics.examples.models.sfno")
_MODERN_SFNO_API = hasattr(_sfno_module, "SphericalFourierNeuralOperator")
_SFNO_CLASS: Any = getattr(
    _sfno_module,
    (
        "SphericalFourierNeuralOperator"
        if _MODERN_SFNO_API
        else "SphericalFourierNeuralOperatorNet"
    ),
)


def _build_corrector_sfno(
    *,
    img_size: tuple[int, int],
    input_channels: int,
    output_channels: int,
    embed_dim: int,
    num_layers: int,
) -> nn.Module:
    """Build a full-resolution SFNO with no hidden planar resampling."""
    options: dict[str, Any] = {
        "img_size": img_size,
        "operator_type": "driscoll-healy",
        "grid": "equiangular",
        "grid_internal": "equiangular",
        "scale_factor": 1,
        "in_chans": input_channels,
        "out_chans": output_channels,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "activation_function": "gelu",
        "use_mlp": True,
        "mlp_ratio": 2.0,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "normalization_layer": "instance_norm",
        "hard_thresholding_fraction": 1.0,
        "use_complex_kernels": True,
    }
    constructor_parameters = inspect.signature(_SFNO_CLASS).parameters
    accepts_extra_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    if "pos_embed" in constructor_parameters or accepts_extra_options:
        options["pos_embed"] = False
    if _MODERN_SFNO_API and (
        "residual_prediction" in constructor_parameters or accepts_extra_options
    ):
        options["residual_prediction"] = False
    elif "big_skip" in constructor_parameters or accepts_extra_options:
        options["big_skip"] = False
    else:
        raise RuntimeError(
            "Installed SFNO API cannot disable its external residual connection"
        )
    if "bias" in constructor_parameters:
        options["bias"] = False
    supported = (
        options
        if accepts_extra_options
        else {
            name: value
            for name, value in options.items()
            if name in constructor_parameters
        }
    )
    network = _SFNO_CLASS(**supported)
    stabilize_sht_constants(network)
    return network


class SphericalResidualErrorCorrector(nn.Module):
    """Learn the standardized tendency error left by a frozen SFNO.

    The learnable operator is an SFNO at the native latitude-longitude
    resolution. It therefore avoids planar pole padding, longitude seams,
    strided pooling, and bilinear upsampling. Two fixed latitude-coordinate
    channels are added to the physical covariates. The last 1x1 projection is
    exactly zero at construction, so the composite forecast initially equals
    the Stage-1 baseline.
    """

    geometry_channels = 2

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        img_size: tuple[int, int],
        embed_dim: int = 84,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if min(input_channels, output_channels, embed_dim, num_layers) < 1:
            raise ValueError("Corrector dimensions must be positive")
        height, width = img_size
        if height < 3 or width < 4:
            raise ValueError("Corrector grid is too small for spherical transforms")

        latitudes = torch.linspace(math.pi / 2.0, -math.pi / 2.0, height)
        latitude_features = torch.stack(
            (torch.sin(latitudes), torch.cos(latitudes))
        )[:, :, None].expand(-1, -1, width)
        self.register_buffer("latitude_features", latitude_features.contiguous())
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.network = _build_corrector_sfno(
            img_size=img_size,
            input_channels=input_channels + self.geometry_channels,
            output_channels=output_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        output_projection = self.output_projection
        nn.init.zeros_(output_projection.weight)
        if output_projection.bias is not None:
            nn.init.zeros_(output_projection.bias)

    @property
    def output_projection(self) -> nn.Conv2d:
        projection = cast(Any, self.network).decoder[0]
        if not isinstance(projection, nn.Conv2d):
            raise RuntimeError("SFNO decoder does not expose the expected 1x1 projection")
        return projection

    def forward(self, values: Tensor) -> Tensor:
        expected = (
            self.input_channels,
            cast(Tensor, self.latitude_features).shape[-2],
            cast(Tensor, self.latitude_features).shape[-1],
        )
        if values.ndim != 4 or values.shape[1:] != expected:
            raise ValueError(
                f"Corrector input must have shape [B, {expected[0]}, "
                f"{expected[1]}, {expected[2]}]"
            )
        geometry = cast(Tensor, self.latitude_features).to(
            device=values.device, dtype=values.dtype
        )
        geometry = geometry.unsqueeze(0).expand(values.shape[0], -1, -1, -1)
        output = cast(Tensor, self.network(torch.cat((values, geometry), dim=1)))
        expected_output = (values.shape[0], self.output_channels, *expected[-2:])
        if output.shape != expected_output:
            raise RuntimeError(
                f"Corrector returned shape {tuple(output.shape)}, expected {expected_output}"
            )
        return output


# Deliberately keep no compatibility alias for the rejected planar U-Net. A
# stale import should fail loudly instead of silently restoring its geometry.
