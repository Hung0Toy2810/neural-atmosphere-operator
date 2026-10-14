"""Deterministic astronomical and static conditioning for weather models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn


FORCING_CHANNEL_NAMES = (
    "normalized_surface_geopotential",
    "normalized_land_sea_mask",
    "cosine_solar_zenith_angle",
    "sine_local_time",
    "cosine_local_time",
    "sine_year_progress",
    "cosine_year_progress",
)


def time_forcing_phases(timestamp: np.datetime64) -> np.ndarray:
    """Return NOAA fractional-year, UTC-day and annual phases.

    Only three scalars are moved by the data loader. Spatial forcing fields are
    constructed on the accelerator, avoiding several dense CPU feature maps
    per sample and preventing a new input pipeline bottleneck.
    """
    instant = np.datetime64(timestamp, "ns")
    year_start = instant.astype("datetime64[Y]").astype("datetime64[ns]")
    next_year = (instant.astype("datetime64[Y]") + 1).astype("datetime64[ns]")
    day_start = instant.astype("datetime64[D]").astype("datetime64[ns]")
    day_ns = float(np.timedelta64(1, "D") / np.timedelta64(1, "ns"))
    elapsed_days = float((instant - year_start) / np.timedelta64(1, "ns")) / day_ns
    days_in_year = float((next_year - year_start) / np.timedelta64(1, "D"))
    day_fraction = float((instant - day_start) / np.timedelta64(1, "ns")) / day_ns
    gamma = 2.0 * math.pi * (elapsed_days - 0.5) / days_in_year
    utc_phase = 2.0 * math.pi * day_fraction
    annual_phase = 2.0 * math.pi * elapsed_days / days_in_year
    return np.asarray((gamma, utc_phase, annual_phase), dtype=np.float32)


def load_static_forcing(path: str | Path, shape: tuple[int, int]) -> Tensor:
    """Load and verify the audited normalized orography and land-mask arrays."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Static forcing bundle not found: {source}")
    metadata_path = source.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Static forcing metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid static forcing metadata: {metadata_path}") from error
    expected_metadata = {
        "version": 1,
        "shape": list(shape),
        "normalization": "spherical_latitude_cell_area_zscore",
        "regridding_version": 2,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Static forcing metadata {key!r} must be {expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if metadata.get("bundle_sha256") != digest.hexdigest():
        raise ValueError("Static forcing bundle SHA-256 does not match its metadata")
    with np.load(source, allow_pickle=False) as bundle:
        required = {
            "normalized_surface_geopotential",
            "normalized_land_sea_mask",
            "latitude",
            "longitude",
        }
        missing = required - set(bundle.files)
        if missing:
            raise ValueError("Static forcing bundle is missing: " + ", ".join(missing))
        latitudes = np.asarray(bundle["latitude"], dtype=np.float64)
        longitudes = np.asarray(bundle["longitude"], dtype=np.float64)
        # Preserve the semantic channel order rather than alphabetical order.
        values = np.stack(
            [
                np.asarray(bundle["normalized_surface_geopotential"]),
                np.asarray(bundle["normalized_land_sea_mask"]),
            ]
        ).astype(np.float32, copy=False)
    if values.shape != (2, *shape):
        raise ValueError(
            f"Static forcing shape must be {(2, *shape)}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Static forcing values must be finite")
    expected_latitudes = np.linspace(90.0, -90.0, shape[0], dtype=np.float64)
    expected_longitudes = np.arange(shape[1], dtype=np.float64) * (360.0 / shape[1])
    if latitudes.shape != (shape[0],) or not np.allclose(
        latitudes, expected_latitudes, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Static forcing latitude must run north-to-south from 90 to -90")
    if longitudes.shape != (shape[1],) or not np.allclose(
        longitudes, expected_longitudes, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Static forcing longitude must run eastward from 0 degrees")
    return torch.from_numpy(np.ascontiguousarray(values))


class AnalyticForcingEncoder(nn.Module):
    """Create seven full-grid conditioning fields from three time scalars."""

    channel_names = FORCING_CHANNEL_NAMES

    def __init__(self, img_size: tuple[int, int]) -> None:
        super().__init__()
        height, width = img_size
        latitudes = torch.linspace(90.0, -90.0, height).deg2rad()
        longitudes = torch.arange(width, dtype=torch.float32) * (2.0 * math.pi / width)
        self.register_buffer("latitudes_rad", latitudes.view(1, height, 1))
        self.register_buffer("longitudes_rad", longitudes.view(1, 1, width))
        self.register_buffer("static_fields", torch.zeros(2, height, width))

    def set_static_fields(self, fields: Tensor) -> None:
        static_fields = cast(Tensor, self.static_fields)
        if fields.shape != static_fields.shape or not torch.isfinite(fields).all():
            raise ValueError(
                f"Static forcing must be finite with shape {tuple(static_fields.shape)}"
            )
        static_fields.copy_(
            fields.to(device=static_fields.device, dtype=static_fields.dtype)
        )

    def forward(self, phases: Tensor) -> Tensor:
        if phases.ndim != 2 or phases.shape[1] != 3:
            raise ValueError("forcing phases must have shape [B, 3]")
        static_fields = cast(Tensor, self.static_fields)
        phases = phases.to(device=static_fields.device, dtype=torch.float32)
        gamma, utc_phase, annual_phase = phases.unbind(dim=1)
        # NOAA's compact solar-position approximation. Equation of time is in
        # minutes and declination is in radians.
        equation_minutes = 229.18 * (
            0.000075
            + 0.001868 * torch.cos(gamma)
            - 0.032077 * torch.sin(gamma)
            - 0.014615 * torch.cos(2.0 * gamma)
            - 0.040849 * torch.sin(2.0 * gamma)
        )
        declination = (
            0.006918
            - 0.399912 * torch.cos(gamma)
            + 0.070257 * torch.sin(gamma)
            - 0.006758 * torch.cos(2.0 * gamma)
            + 0.000907 * torch.sin(2.0 * gamma)
            - 0.002697 * torch.cos(3.0 * gamma)
            + 0.00148 * torch.sin(3.0 * gamma)
        )
        longitude = cast(Tensor, self.longitudes_rad)
        latitude = cast(Tensor, self.latitudes_rad)
        local_phase = utc_phase[:, None, None] + longitude
        hour_angle = (
            local_phase
            - math.pi
            + equation_minutes[:, None, None] * (math.pi / 720.0)
        )
        cosine_zenith = (
            torch.sin(latitude) * torch.sin(declination)[:, None, None]
            + torch.cos(latitude)
            * torch.cos(declination)[:, None, None]
            * torch.cos(hour_angle)
        ).clamp(-1.0, 1.0)
        batch = phases.shape[0]
        height, width = static_fields.shape[-2:]
        static = static_fields.unsqueeze(0).expand(batch, -1, -1, -1)
        sine_year = torch.sin(annual_phase)[:, None, None].expand(batch, height, width)
        cosine_year = torch.cos(annual_phase)[:, None, None].expand(batch, height, width)
        return torch.cat(
            (
                static,
                cosine_zenith[:, None],
                torch.sin(local_phase).expand(batch, height, width)[:, None],
                torch.cos(local_phase).expand(batch, height, width)[:, None],
                sine_year[:, None],
                cosine_year[:, None],
            ),
            dim=1,
        )
