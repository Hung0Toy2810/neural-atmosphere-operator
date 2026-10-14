"""Controlled progressive unfreezing for residual-corrector training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import nn

from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator


def _backbone_tail_modules(
    model: AtmosphereNeuralOperator, block_count: int
) -> tuple[nn.Module, ...]:
    if block_count < 0:
        raise ValueError("backbone tail block count cannot be negative")
    blocks = getattr(model.backbone, "blocks", None)
    decoder = getattr(model.backbone, "decoder", None)
    if not isinstance(blocks, nn.ModuleList) or not isinstance(decoder, nn.Module):
        raise TypeError("SFNO backbone does not expose blocks and decoder")
    if block_count > len(blocks):
        raise ValueError(
            f"cannot unfreeze {block_count} blocks from a {len(blocks)}-block SFNO"
        )
    selected_blocks = tuple(blocks[-block_count:]) if block_count else ()
    return (decoder, *selected_blocks) if block_count else ()


def _parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def configure_progressive_finetuning(
    model: AtmosphereNeuralOperator,
    *,
    backbone_unfreeze_blocks: int,
    corrector_learning_rate: float,
    backbone_learning_rate: float,
    corrector_weight_decay: float,
    backbone_weight_decay: float,
) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    """Freeze the backbone and construct stable optimizer parameter groups.

    The future backbone-tail group exists from update zero but remains frozen
    initially. This keeps optimizer and scheduler group topology unchanged
    across phase transitions and therefore makes epoch-boundary resume exact.
    """

    if model.corrector is None:
        raise ValueError("progressive fine-tuning requires a corrector")
    if corrector_learning_rate <= 0 or backbone_learning_rate <= 0:
        raise ValueError("fine-tuning learning rates must be positive")
    if corrector_weight_decay < 0 or backbone_weight_decay < 0:
        raise ValueError("weight decay cannot be negative")

    model.backbone.requires_grad_(False)
    model.corrector.requires_grad_(True)
    corrector_parameters = list(model.corrector.parameters())
    groups: list[dict[str, Any]] = [
        {
            "name": "corrector",
            "params": corrector_parameters,
            "lr": corrector_learning_rate,
            "weight_decay": corrector_weight_decay,
        }
    ]
    optimized_parameters = list(corrector_parameters)
    tail_parameters = _parameters(
        _backbone_tail_modules(model, backbone_unfreeze_blocks)
    )
    if tail_parameters:
        groups.append(
            {
                "name": "backbone_tail",
                "params": tail_parameters,
                "lr": backbone_learning_rate,
                # Do not shrink a pretrained operator toward zero merely
                # because it was selectively unfrozen. Its small LR and
                # narrow tail selection are the drift controls.
                "weight_decay": backbone_weight_decay,
            }
        )
        optimized_parameters.extend(tail_parameters)
    return groups, optimized_parameters


def apply_progressive_finetuning_phase(
    model: AtmosphereNeuralOperator,
    *,
    epoch: int,
    corrector_only_epochs: int,
    backbone_unfreeze_blocks: int,
) -> str:
    """Apply the train/eval and gradient state for one zero-based epoch."""

    if epoch < 0 or corrector_only_epochs < 0:
        raise ValueError("epoch counts cannot be negative")
    if model.corrector is None:
        raise ValueError("progressive fine-tuning requires a corrector")

    model.train()
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.corrector.requires_grad_(True)
    model.corrector.train()

    tail_modules = _backbone_tail_modules(model, backbone_unfreeze_blocks)
    tail_active = bool(tail_modules) and epoch >= corrector_only_epochs
    if tail_active:
        for module in tail_modules:
            module.requires_grad_(True)
            module.train()
        return "corrector_plus_backbone_tail"
    return "corrector_only"


def optimizer_learning_rate(optimizer: Any, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return 0.0


def real_scalar_parameter_count(parameters: Iterable[nn.Parameter]) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in parameters
    )
