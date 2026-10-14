"""Spherical Fourier Neural Operator for global atmospheric forecasting."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, cast

import torch
from torch import Tensor, nn

from configs.model_config import AtmosphereModelConfig

from .conditioning import AnalyticForcingEncoder
from .corrector import SphericalResidualErrorCorrector
from .transforms import stabilize_sht_constants

# Follow NeuralOceanOperator's dependency boundary: the reference spherical
# architectures are examples in torch-harmonics rather than stable core API.
# Keeping the import and API adaptation here prevents version-specific details
# from leaking into the rest of the project.
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


def _build_sfno(
    config: AtmosphereModelConfig, *, input_channels: int | None = None
) -> nn.Module:
    """Build the official torch-harmonics SFNO for the installed API.

    ``torch-harmonics`` renamed the reference class and some residual/position
    options between releases. Only supported constructor arguments are passed,
    while the scientific architecture remains the same. The large internal
    residual is deliberately disabled because :class:`AtmosphereNeuralOperator`
    performs the physically interpretable current-state plus tendency update.
    """

    options: dict[str, Any] = {
        "img_size": config.img_size,
        "operator_type": config.operator_type,
        "grid": config.grid,
        "grid_internal": config.grid_internal,
        "scale_factor": config.scale_factor,
        "in_chans": input_channels or config.in_channels,
        "out_chans": config.out_channels,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "activation_function": config.activation_function,
        "use_mlp": config.use_mlp,
        "mlp_ratio": config.mlp_ratio,
        "drop_rate": config.drop_rate,
        "drop_path_rate": config.drop_path_rate,
        "normalization_layer": config.normalization_layer,
        "hard_thresholding_fraction": config.hard_thresholding_fraction,
        "use_complex_kernels": config.use_complex_kernels,
    }

    if config.pos_embed == "none":
        position_embedding: bool | str = False
    elif _MODERN_SFNO_API:
        position_embedding = "learnable lat"
    else:
        # torch-harmonics 0.7 accepts named modes although its annotation in
        # some releases incorrectly declares this parameter as bool.
        position_embedding = cast(Any, "lat")

    constructor_parameters = inspect.signature(_SFNO_CLASS).parameters
    accepts_extra_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    required_options = {
        "img_size",
        "scale_factor",
        "in_chans",
        "out_chans",
        "embed_dim",
        "num_layers",
        "activation_function",
        "use_mlp",
        "mlp_ratio",
        "normalization_layer",
        "hard_thresholding_fraction",
    }
    unsupported_required = required_options - constructor_parameters.keys()
    if unsupported_required and not accepts_extra_options:
        raise RuntimeError(
            "Installed torch-harmonics SFNO API cannot represent the requested "
            "architecture; missing constructor options: "
            + ", ".join(sorted(unsupported_required))
        )
    if "pos_embed" in constructor_parameters or accepts_extra_options:
        options["pos_embed"] = position_embedding
    if _MODERN_SFNO_API and (
        "residual_prediction" in constructor_parameters or accepts_extra_options
    ):
        options["residual_prediction"] = False
    elif "big_skip" in constructor_parameters or accepts_extra_options:
        options["big_skip"] = False
    if not accepts_extra_options and not {
        "residual_prediction",
        "big_skip",
    }.intersection(constructor_parameters):
        raise RuntimeError(
            "Installed SFNO API does not expose a residual control; external "
            "current-state addition could otherwise be applied twice"
        )
    if "bias" in constructor_parameters:
        options["bias"] = False

    supported_options = (
        options
        if accepts_extra_options
        else {
            name: value
            for name, value in options.items()
            if name in constructor_parameters
        }
    )
    model = _SFNO_CLASS(**supported_options)
    if config.stabilize_sht_constants:
        stabilize_sht_constants(model)
    return model


class AtmosphereNeuralOperator(nn.Module):
    """Predict the next normalized atmospheric state with an SFNO tendency.

    The backbone is NVIDIA's reference Spherical Fourier Neural Operator from
    ``torch-harmonics``. Unlike planar AFNO, its spectral mixing uses spherical
    harmonic transforms and therefore respects the geometry of the global
    latitude-longitude grid.

    As in ``NeuralOceanOperator``, the backbone predicts a normalized tendency
    and this wrapper adds it to the most recent state. The operation remains
    differentiable through an autoregressive rollout.
    """

    def __init__(self, config: AtmosphereModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or AtmosphereModelConfig()
        # An enhanced run keeps the Stage-1 operator structurally identical:
        # it sees only the latest 71-channel state. History and forcing are
        # covariates of the separate residual-error model below.
        backbone_input_channels = (
            self.config.out_channels
            if self.config.corrector_hidden_channels
            else self.config.in_channels + self.config.forcing_channels
        )
        self.backbone = _build_sfno(
            self.config,
            input_channels=backbone_input_channels,
        )
        self.forcing_encoder = (
            AnalyticForcingEncoder(self.config.img_size)
            if self.config.forcing_channels
            else None
        )
        history_states = self.config.in_channels // self.config.out_channels
        corrector_input_channels = (
            2 * self.config.out_channels
            + (self.config.out_channels if history_states > 1 else 0)
            + self.config.forcing_channels
        )
        self.corrector = (
            SphericalResidualErrorCorrector(
                corrector_input_channels,
                self.config.out_channels,
                img_size=self.config.img_size,
                embed_dim=self.config.corrector_hidden_channels,
                num_layers=self.config.corrector_depth,
            )
            if self.config.corrector_hidden_channels
            else None
        )
        scale = (
            torch.ones(self.config.out_channels, dtype=torch.float32)
            if self.config.tendency_scale is None
            else torch.tensor(self.config.tendency_scale, dtype=torch.float32)
        )
        self.register_buffer("tendency_scale", scale.view(1, -1, 1, 1))

    def set_static_forcing(self, fields: Tensor) -> None:
        """Install the audited two-channel static forcing bundle."""
        if self.forcing_encoder is None:
            raise RuntimeError("Static forcing is disabled by the model config")
        self.forcing_encoder.set_static_fields(fields)

    def standardized_tendency_components(
        self,
        x: Tensor,
        forcing_time_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return baseline and learned-correction standardized tendencies."""
        self._validate_input(x)
        forcing = None
        if self.forcing_encoder is not None:
            if forcing_time_features is None:
                raise ValueError("forcing_time_features are required by this model")
            forcing = self.forcing_encoder(forcing_time_features)
            forcing = forcing.to(device=x.device, dtype=x.dtype)
        elif forcing_time_features is not None:
            raise ValueError("forcing_time_features were provided to a disabled model")

        current_state = x[:, -self.config.out_channels :]
        backbone_input = current_state if self.corrector is not None else x
        if forcing is not None and self.corrector is None:
            backbone_input = torch.cat((backbone_input, forcing), dim=1)
        baseline = self.backbone(backbone_input)
        expected_output_shape = (
            x.shape[0],
            self.config.out_channels,
            *self.config.img_size,
        )
        if baseline.shape != expected_output_shape:
            raise RuntimeError(
                "SFNO backbone returned shape "
                f"{tuple(baseline.shape)}, expected {expected_output_shape}"
            )

        if self.corrector is None:
            return baseline, torch.zeros_like(baseline)
        features = [current_state]
        if self.config.in_channels > self.config.out_channels:
            previous_state = x[
                :, -2 * self.config.out_channels : -self.config.out_channels
            ]
            # x is normalized by sigma(x). Divide the normalized difference
            # by sigma(delta x)/sigma(x), yielding delta x / sigma(delta x):
            # the same units as both standardized tendency predictions.
            scale = cast(Tensor, self.tendency_scale).to(
                device=current_state.device, dtype=current_state.dtype
            )
            features.append((current_state - previous_state) / scale)
        # Treat the baseline forecast as a fixed covariate for the downstream
        # error model instead of creating a second gradient path into SFNO.
        features.append(baseline.detach())
        if forcing is not None:
            features.append(forcing)
        return baseline, self.corrector(torch.cat(features, dim=1))

    def forward(
        self, x: Tensor, forcing_time_features: Tensor | None = None
    ) -> Tensor:
        """Return the next state for an input shaped ``[B, C_in, H, W]``."""
        baseline, correction = self.standardized_tendency_components(
            x, forcing_time_features
        )
        standardized_tendency = (
            baseline if self.corrector is None else baseline + correction
        )

        if not self.config.use_residual_connection:
            return standardized_tendency

        # AtmosphereZarrDataset concatenates history oldest-to-newest, making
        # the final out_channels the current normalized atmospheric state.
        current_state = x[:, -self.config.out_channels :]
        scale = cast(Tensor, self.tendency_scale).to(
            device=standardized_tendency.device,
            dtype=standardized_tendency.dtype,
        )
        return current_state + standardized_tendency * scale

    def rollout(
        self,
        x: Tensor,
        steps: int = 1,
        forcing_time_features: Tensor | None = None,
    ) -> list[Tensor]:
        """Perform a differentiable autoregressive rollout for ``steps`` leads."""
        self._validate_input(x)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if self.config.in_channels % self.config.out_channels != 0:
            raise ValueError(
                "Autoregressive rollout requires in_channels to be a multiple "
                "of out_channels"
            )

        from neural_atmosphere_operator.pipeline.forecast import forecast_states

        return list(
            forecast_states(
                self,
                x,
                steps,
                forcing_time_features=forcing_time_features,
            )
        )

    def _validate_input(self, x: Tensor) -> None:
        expected_shape = (self.config.in_channels, *self.config.img_size)
        if x.ndim != 4 or x.shape[1:] != expected_shape:
            raise ValueError(
                f"x must have shape [B, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}]"
            )
