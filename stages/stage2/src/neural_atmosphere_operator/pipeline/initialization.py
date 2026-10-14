"""Audited warm-starts for architecture-expansion experiments."""

from __future__ import annotations

import json
from typing import Any

import torch

from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.runtime import model_config_from_dict


def _same_contract_value(left: Any, right: Any) -> bool:
    """Compare JSON-compatible provenance independent of tuple/list storage."""
    try:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return left == right


def validate_expansion_training_contract(
    checkpoint: dict[str, Any], current_training: dict[str, Any]
) -> None:
    """Require the baseline and expanded run to use the same scientific data.

    History changes the window counts by design, so sample-count fields are
    excluded. Payload inventories, physical coordinates, units, time ranges,
    statistics artifacts, loss definition, and forecast cadence must match.
    """
    source_training = checkpoint.get("training_config")
    if not isinstance(source_training, dict):
        raise ValueError("Baseline checkpoint lacks a training contract")
    exact_fields = (
        "loss_name",
        "channel_weighting",
        "time_step",
        "rollout_steps",
        "forecast_contract",
    )
    mismatches = [
        field
        for field in exact_fields
        if not _same_contract_value(
            source_training.get(field), current_training.get(field)
        )
    ]
    source_data = source_training.get("data_signature", {})
    current_data = current_training.get("data_signature", {})
    if not isinstance(source_data, dict) or not isinstance(current_data, dict):
        raise ValueError("Baseline or target lacks data provenance")

    statistic_names = ("means", "stds", "climatology", "time_diff_stds")
    for name in statistic_names:
        source = source_data.get(name)
        current = current_data.get(name)
        if (
            not isinstance(source, dict)
            or not isinstance(current, dict)
            or source.get("sha256") != current.get("sha256")
        ):
            mismatches.append(f"data_signature.{name}.sha256")

    dataset_fields = (
        "paths",
        "payload_inventory_sha256",
        "timesteps",
        "first_time",
        "last_time",
        "cadence_hours",
        "spatial_shape",
        "channels",
        "channel_units",
        "latitude",
        "longitude",
    )
    for split in ("train", "valid"):
        source = source_data.get(split)
        current = current_data.get(split)
        if not isinstance(source, dict) or not isinstance(current, dict):
            mismatches.append(f"data_signature.{split}")
            continue
        mismatches.extend(
            f"data_signature.{split}.{field}"
            for field in dataset_fields
            if not _same_contract_value(source.get(field), current.get(field))
        )
    if mismatches:
        raise ValueError(
            "Baseline scientific training contract differs in: "
            + ", ".join(mismatches)
        )


def initialize_enhanced_from_baseline(
    model: AtmosphereNeuralOperator,
    checkpoint: dict[str, Any],
) -> dict[str, int]:
    """Attach a zero-output corrector to an unchanged Stage-1 backbone.

    The SFNO tensors and tendency scale are copied exactly. History and
    forcing never enter the frozen backbone, so no input-projection expansion
    or approximate checkpoint conversion is needed. Optimizer/scheduler state
    is intentionally not transferred because the trainable topology changed.
    """
    source = model_config_from_dict(checkpoint["model_config"])
    target = model.config
    _validate_expansion(source, target)

    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    copied = 0
    for name, source_value in source_state.items():
        if not (name.startswith("backbone.") or name == "tendency_scale"):
            raise ValueError(f"Baseline contains unsupported enhanced tensor: {name}")
        if name not in target_state or target_state[name].shape != source_value.shape:
            raise ValueError(f"Cannot transfer baseline tensor {name}")
        target_state[name].copy_(source_value)
        copied += source_value.numel()

    with torch.no_grad():
        model.load_state_dict(target_state, strict=True)

    return {
        "copied_backbone_values": copied,
        "backbone_input_channels": source.in_channels,
        "corrector_covariate_channels": (
            3 * target.out_channels + target.forcing_channels + 2
        ),
    }


def _validate_expansion(
    source: AtmosphereModelConfig,
    target: AtmosphereModelConfig,
) -> None:
    if source.forcing_mode != "none" or source.corrector_hidden_channels != 0:
        raise ValueError("Expansion source must be an unconditioned baseline")
    if source.in_channels != source.out_channels:
        raise ValueError("Expansion source must use exactly one prognostic state")
    if target.in_channels != 2 * source.in_channels:
        raise ValueError("Target must add exactly the t-6 history state")
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
