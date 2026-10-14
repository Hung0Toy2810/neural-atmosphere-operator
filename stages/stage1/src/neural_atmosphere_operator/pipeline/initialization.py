"""Audited warm-starts for architecture-expansion experiments."""

from __future__ import annotations

from typing import Any

import torch

from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.runtime import model_config_from_dict


_ENCODER_WEIGHT = "backbone.encoder.0.weight"


def initialize_enhanced_from_baseline(
    model: AtmosphereNeuralOperator,
    checkpoint: dict[str, Any],
) -> dict[str, int]:
    """Expand a one-state baseline without changing its initial prediction.

    All compatible SFNO tensors are copied exactly. The old 71-channel input
    projection is placed on the newest-state slice, while older history and
    forcing slices are zero. The target model's zero-output corrector remains
    untouched. Optimizer/scheduler state is intentionally not transferred
    because the parameter topology has changed.
    """
    source = model_config_from_dict(checkpoint["model_config"])
    target = model.config
    _validate_expansion(source, target)

    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    if _ENCODER_WEIGHT not in source_state or _ENCODER_WEIGHT not in target_state:
        raise ValueError("Checkpoint does not expose the expected SFNO input encoder")

    copied = 0
    for name, source_value in source_state.items():
        if name == _ENCODER_WEIGHT:
            continue
        if not (name.startswith("backbone.") or name == "tendency_scale"):
            raise ValueError(f"Baseline contains unsupported enhanced tensor: {name}")
        if name not in target_state or target_state[name].shape != source_value.shape:
            raise ValueError(f"Cannot transfer baseline tensor {name}")
        target_state[name].copy_(source_value)
        copied += source_value.numel()

    source_encoder = source_state[_ENCODER_WEIGHT]
    target_encoder = target_state[_ENCODER_WEIGHT]
    expected_target_inputs = target.in_channels + target.forcing_channels
    if target_encoder.shape[1] != expected_target_inputs:
        raise ValueError("Target SFNO encoder does not match model conditioning")
    if source_encoder.shape[1] != source.in_channels:
        raise ValueError("Source SFNO encoder does not match checkpoint config")
    current_start = target.in_channels - target.out_channels
    current_end = current_start + source.in_channels
    with torch.no_grad():
        target_encoder.zero_()
        target_encoder[:, current_start:current_end].copy_(source_encoder)
        model.load_state_dict(target_state, strict=True)

    return {
        "copied_backbone_values": copied + source_encoder.numel(),
        "new_state_input_values": (
            target_encoder[:, :current_start].numel()
            + target_encoder[:, current_end : target.in_channels].numel()
        ),
        "new_forcing_input_values": target_encoder[
            :, target.in_channels :
        ].numel(),
    }


def _validate_expansion(
    source: AtmosphereModelConfig,
    target: AtmosphereModelConfig,
) -> None:
    if source.forcing_mode != "none" or source.corrector_hidden_channels != 0:
        raise ValueError("Expansion source must be an unconditioned baseline")
    if source.in_channels != source.out_channels:
        raise ValueError("Expansion source must use exactly one prognostic state")
    if target.in_channels <= source.in_channels:
        raise ValueError("Target must add at least one history state")
    if target.out_channels != source.out_channels:
        raise ValueError("Source and target output channels must match")
    comparable = (
        "img_size",
        "scale_factor",
        "embed_dim",
        "num_layers",
        "activation_function",
        "normalization_layer",
        "use_mlp",
        "mlp_ratio",
        "drop_rate",
        "drop_path_rate",
        "hard_thresholding_fraction",
        "use_complex_kernels",
        "stabilize_sht_constants",
        "operator_type",
        "grid",
        "grid_internal",
        "pos_embed",
        "use_residual_connection",
        "tendency_scale",
        "initialization",
    )
    mismatches = [
        name for name in comparable if getattr(source, name) != getattr(target, name)
    ]
    if mismatches:
        raise ValueError(
            "Baseline and enhanced SFNO differ in: " + ", ".join(mismatches)
        )
    if target.forcing_mode == "none" or target.corrector_hidden_channels == 0:
        raise ValueError("Target must enable both conditioning and corrector")
