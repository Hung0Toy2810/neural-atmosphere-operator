"""Update-level warmup and cosine learning-rate scheduling."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_multiplier(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float,
    min_factor: float,
) -> float:
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if not 0 < warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0 <= min_factor <= 1:
        raise ValueError("min_factor must be in [0, 1]")

    bounded = min(max(update, 0), total_updates)
    if warmup_updates and bounded < warmup_updates:
        progress = bounded / warmup_updates
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress
    progress = (bounded - warmup_updates) / (total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def replanned_cosine_multiplier(
    update: int,
    *,
    original_total_updates: int,
    new_total_updates: int,
    warmup_updates: int,
    replan_update: int,
    warmup_start_factor: float,
    min_factor: float,
) -> float:
    """Continue an existing cosine schedule and land at a new end update.

    The multiplier is identical to the original schedule through
    ``replan_update``.  From that point it compresses the remaining phase of
    the original cosine into the shorter horizon.  This preserves the exact
    boundary value while making the decay faster from the very next update;
    unlike restarting a cosine segment, it does not flatten the initial
    post-replan slope.
    """
    if not warmup_updates <= replan_update < new_total_updates:
        raise ValueError(
            "replan_update must be after warmup and before new_total_updates"
        )
    if new_total_updates > original_total_updates:
        raise ValueError("a scheduler replan may only shorten the decay horizon")
    if update <= replan_update:
        return warmup_cosine_multiplier(
            update,
            total_updates=original_total_updates,
            warmup_updates=warmup_updates,
            warmup_start_factor=warmup_start_factor,
            min_factor=min_factor,
        )
    bounded = min(update, new_total_updates)
    progress = (bounded - replan_update) / (new_total_updates - replan_update)
    original_phase = math.pi * (
        (replan_update - warmup_updates)
        / (original_total_updates - warmup_updates)
    )
    phase = original_phase + (math.pi - original_phase) * progress
    cosine = 0.5 * (1.0 + math.cos(phase))
    return min_factor + (1.0 - min_factor) * cosine


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float,
    min_learning_rate: float,
    base_learning_rate: float,
    original_total_updates: int | None = None,
    replan_update: int | None = None,
) -> LambdaLR:
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if not 0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be in [0, base_learning_rate]")
    min_factor = min_learning_rate / base_learning_rate
    if (original_total_updates is None) != (replan_update is None):
        raise ValueError(
            "original_total_updates and replan_update must be provided together"
        )
    if original_total_updates is None:
        multiplier = lambda update: warmup_cosine_multiplier(
            update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            warmup_start_factor=warmup_start_factor,
            min_factor=min_factor,
        )
    else:
        assert replan_update is not None
        multiplier = lambda update: replanned_cosine_multiplier(
            update,
            original_total_updates=original_total_updates,
            new_total_updates=total_updates,
            warmup_updates=warmup_updates,
            replan_update=replan_update,
            warmup_start_factor=warmup_start_factor,
            min_factor=min_factor,
        )
    return LambdaLR(
        optimizer,
        lr_lambda=multiplier,
    )
