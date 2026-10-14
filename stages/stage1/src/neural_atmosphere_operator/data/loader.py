"""Worker-safe Zarr and array data pipeline for atmospheric weather forecasting."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .normalization import AtmosphereNormalizer
from configs.download_data_config import channel_names as configured_channel_names
from neural_atmosphere_operator.models.conditioning import time_forcing_phases

LOGGER = logging.getLogger(__name__)

class AtmosphereSample(TypedDict):
    """Standard training sample contract for atmospheric forecasting models."""

    input: Tensor  # [C_in, H, W] where C_in = (history + 1) * C
    target: Tensor  # [rollout_steps, C_out, H, W]
    time_index: Tensor  # scalar index of base time t
    target_time_indices: Tensor  # [rollout_steps], used for lazy verification
    forcing_time_features: NotRequired[Tensor]  # [rollout_steps, 3]


@dataclass(frozen=True, slots=True)
class AtmosphereDatasetConfig:
    """Configuration for :class:`AtmosphereZarrDataset`."""

    data_path: str | Path | tuple[str | Path, ...]
    means_path: str | Path | np.ndarray | Tensor | None = None
    stds_path: str | Path | np.ndarray | Tensor | None = None
    start_time: str | None = None
    end_time: str | None = None
    history: int = 0
    rollout_steps: int = 1
    time_step: int = 1
    normalize: bool = True
    add_noise: bool = False
    noise_std: float = 0.0
    spatial_crop: tuple[int, int] | None = None
    max_samples: int | None = None
    sample_selection: str = "first"
    require_targets: bool = True
    lazy_targets: bool = False
    drop_file_cache: bool = False
    time_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.sample_selection not in {"first", "evenly_spaced"}:
            raise ValueError("Unsupported sample_selection")
        if self.history < 0:
            raise ValueError("history cannot be negative")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.time_step < 1:
            raise ValueError("time_step must be at least 1")
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must not be later than end_time")
        if self.add_noise and self.noise_std <= 0:
            raise ValueError("noise_std must be positive when add_noise=True")

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return configured_channel_names()


class AtmosphereZarrDataset(Dataset[AtmosphereSample]):
    """Worker-safe Zarr dataset for global atmospheric forecasting."""

    data_paths: tuple[Path, ...]
    data_path: Path

    def __init__(
        self, config: AtmosphereDatasetConfig, *, training: bool = True
    ) -> None:
        self.config = config
        self.training = training
        configured_paths = (
            config.data_path
            if isinstance(config.data_path, tuple)
            else (config.data_path,)
        )
        if not configured_paths:
            raise ValueError("At least one dataset path is required")
        self.data_paths = tuple(
            Path(path).expanduser().resolve() for path in configured_paths
        )
        # Retain the single-path attribute for existing callers. Multi-shard
        # provenance is represented by data_paths and dataset_signature.
        self.data_path = self.data_paths[0]
        missing = [path for path in self.data_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Dataset path not found: {missing[0]}")

        self.normalizer = AtmosphereNormalizer(
            means=config.means_path,
            stds=config.stds_path,
            normalize=config.normalize,
        )
        if config.normalize:
            assert self.normalizer.means is not None
            if self.normalizer.means.size != config.channel_count:
                raise ValueError(
                    "Normalization statistics must contain exactly "
                    f"{config.channel_count} channels"
                )

        # Inspect dimensions and coordinates from metadata lazily
        self._inspect_dataset()

        # Cache for worker-local zarr/xarray handles
        self._worker_ds: xr.Dataset | None = None
        self._worker_sources: tuple[xr.Dataset, ...] = ()

    def _open_dataset(self) -> tuple[xr.Dataset, tuple[xr.Dataset, ...]]:
        sources = tuple(
            xr.open_zarr(str(path), consolidated=True) for path in self.data_paths
        )
        if len(sources) == 1:
            combined = sources[0]
        else:
            try:
                combined = xr.concat(
                    sources,
                    dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="equals",
                    join="exact",
                    combine_attrs="override",
                )
            except Exception:
                for source in sources:
                    source.close()
                raise
        return self._select_time_range(combined), sources

    @staticmethod
    def _close_dataset(
        dataset: xr.Dataset, sources: tuple[xr.Dataset, ...]
    ) -> None:
        dataset.close()
        for source in sources:
            if source is not dataset:
                source.close()

    def _inspect_dataset(self) -> None:
        ds, sources = self._open_dataset()
        try:
            if "state" not in ds:
                raise ValueError(
                    "Dataset must contain the canonical flattened 71-channel state"
                )
            if set(ds.state.dims) != {"time", "channel", "latitude", "longitude"}:
                raise ValueError(
                    "state dimensions must be time/channel/latitude/longitude"
                )
            stored_channels = tuple(str(value) for value in ds.channel.values)
            if stored_channels != self.config.channel_names:
                raise ValueError("Stored state channels do not match the 71-channel contract")
            self.channel_units = self._string_channel_coordinate(
                ds, "channel_units", stored_channels, fallback=""
            )
            self.channel_long_names = self._string_channel_coordinate(
                ds, "channel_long_name", stored_channels, fallback=""
            )
            self.channel_source_variables = self._string_channel_coordinate(
                ds, "channel_source_variable", stored_channels, fallback=""
            )
            if "channel_pressure_level_hpa" in ds.coords:
                pressure_levels = np.asarray(
                    ds["channel_pressure_level_hpa"].values, dtype=np.float32
                )
                if pressure_levels.shape != (len(stored_channels),):
                    raise ValueError("channel_pressure_level_hpa must align with channel")
                self.channel_pressure_levels_hpa = tuple(
                    float(value) for value in pressure_levels
                )
            else:
                self.channel_pressure_levels_hpa = (float("nan"),) * len(
                    stored_channels
                )
            source = str(ds.attrs.get("source", ""))
            if (
                source.startswith("gs://weatherbench2/")
                and int(ds.attrs.get("regridding_version", -1)) != 2
            ):
                raise ValueError(
                    "WeatherBench2 state store uses an unaudited regridding "
                    "version; regenerate it with the current downloader"
                )
            for coordinate in ("time", "latitude", "longitude"):
                if coordinate not in ds.coords:
                    raise ValueError(
                        f"Dataset is missing required coordinate {coordinate!r}"
                    )
            self.total_timesteps = len(ds.time)
            time_values = np.asarray(ds.time.values)
            if self.total_timesteps < 2:
                raise ValueError("Dataset must contain at least two timesteps")
            all_latitudes = np.asarray(ds.latitude.values)
            all_longitudes = np.asarray(ds.longitude.values)
            self.first_time = str(time_values[0])
            self.last_time = str(time_values[-1])

            if not np.issubdtype(time_values.dtype, np.datetime64):
                raise ValueError("ERA5 time coordinate must use datetime64 values")
            time_nanoseconds = time_values.astype("datetime64[ns]").astype(np.int64)
            time_differences = np.diff(time_nanoseconds)
            if not np.all(time_differences > 0):
                raise ValueError("Time coordinates must be strictly increasing")
            if not np.all(time_differences == time_differences[0]):
                raise ValueError("Time coordinates must have a regular cadence")
            self.cadence_hours = float(time_differences[0] / 3_600_000_000_000)
            self.cadence_nanoseconds = int(time_differences[0])
            if not np.isfinite(self.cadence_hours) or self.cadence_hours <= 0:
                raise ValueError("Time cadence must be positive")
            self.times = time_values.astype("datetime64[ns]").copy()
            # Keep a compact timestamp index for direct shard reads. Reading
            # payload through xr.concat creates a Dask concatenation graph for
            # every sample and is much slower than indexing the owning store.
            self._source_times = tuple(
                np.asarray(source.time.values).astype("datetime64[ns]")
                for source in sources
            )
            self._source_last_times = np.asarray(
                [values[-1] for values in self._source_times],
                dtype="datetime64[ns]",
            )

            if not np.all(np.isfinite(all_latitudes)) or not np.all(
                np.isfinite(all_longitudes)
            ):
                raise ValueError("Latitude and longitude coordinates must be finite")
            if len(all_latitudes) < 3 or len(all_longitudes) < 4:
                raise ValueError("Grid is too small for spherical forecasting")
            latitude_steps = np.diff(all_latitudes.astype(np.float64))
            longitude_steps = np.diff(all_longitudes.astype(np.float64))
            if not (np.all(latitude_steps > 0) or np.all(latitude_steps < 0)):
                raise ValueError("Latitude coordinates must be strictly monotonic")
            if not np.all(longitude_steps > 0):
                raise ValueError("Longitude coordinates must be strictly increasing")
            if not np.allclose(
                np.abs(latitude_steps),
                np.abs(latitude_steps[0]),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("SFNO requires regularly spaced latitudes")
            if not np.allclose(
                longitude_steps,
                longitude_steps[0],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("SFNO requires regularly spaced longitudes")

        finally:
            self._close_dataset(ds, sources)

        full_h, full_w = len(all_latitudes), len(all_longitudes)
        if self.config.spatial_crop is not None:
            crop_h, crop_w = self.config.spatial_crop
            if crop_h < 1 or crop_w < 1:
                raise ValueError("spatial_crop dimensions must be positive")
            if crop_h > full_h or crop_w > full_w:
                raise ValueError(
                    f"Crop {self.config.spatial_crop} exceeds dataset shape "
                    f"{(full_h, full_w)}"
                )
            self.spatial_shape = (crop_h, crop_w)
        else:
            self.spatial_shape = (full_h, full_w)
        self.latitudes = np.asarray(
            all_latitudes[: self.spatial_shape[0]], dtype=np.float32
        ).copy()
        self.longitudes = np.asarray(
            all_longitudes[: self.spatial_shape[1]], dtype=np.float32
        ).copy()

        # Compute valid sample count
        context = self.config.history * self.config.time_step
        horizon = (
            self.config.rollout_steps * self.config.time_step
            if self.config.require_targets
            else 0
        )
        valid_count = max(0, self.total_timesteps - context - horizon)

        if valid_count == 0:
            raise ValueError(
                f"Dataset with {self.total_timesteps} steps is too short for history={self.config.history}, "
                f"rollout_steps={self.config.rollout_steps}, time_step={self.config.time_step}"
            )

        if self.config.max_samples is not None:
            self.sample_count = min(valid_count, self.config.max_samples)
        else:
            self.sample_count = valid_count
        self.sample_indices = (
            np.linspace(0, valid_count - 1, self.sample_count, dtype=np.int64)
            if self.config.sample_selection == "evenly_spaced"
            else np.arange(self.sample_count, dtype=np.int64)
        )

    @staticmethod
    def _string_channel_coordinate(
        ds: xr.Dataset,
        name: str,
        channels: tuple[str, ...],
        *,
        fallback: str,
    ) -> tuple[str, ...]:
        """Read and validate a string coordinate aligned with ``channel``."""
        if name not in ds.coords:
            return (fallback,) * len(channels)
        values = tuple(str(value) for value in ds[name].values)
        if len(values) != len(channels):
            raise ValueError(f"{name} must align with the channel dimension")
        return values

    def _get_dataset(self) -> xr.Dataset:
        if self._worker_ds is None:
            self._worker_ds, self._worker_sources = self._open_dataset()
        return self._worker_ds

    def _select_time_range(self, ds: xr.Dataset) -> xr.Dataset:
        if self.config.start_time is None and self.config.end_time is None:
            return ds
        return ds.sel(time=slice(self.config.start_time, self.config.end_time))

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the exact flattened channel order used by every sample."""
        return self.config.channel_names

    def close(self) -> None:
        """Close this process' lazy xarray/Zarr handle."""
        worker_dataset = getattr(self, "_worker_ds", None)
        if worker_dataset is not None:
            self._close_dataset(worker_dataset, self._worker_sources)
            self._worker_ds = None
            self._worker_sources = ()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.sample_count

    def _read_timestep_channels(self, ds: xr.Dataset, t_idx: int) -> np.ndarray:
        """Extract the canonical state at timestep ``t_idx`` as ``[C,H,W]``."""
        h, w = self.spatial_shape
        payload_ds = ds
        payload_index = t_idx
        source_index = 0
        if self._worker_sources:
            timestamp = self.times[t_idx]
            source_index = int(
                np.searchsorted(self._source_last_times, timestamp, side="left")
            )
            if source_index >= len(self._worker_sources):
                raise IndexError(f"Timestamp {timestamp} is outside the source shards")
            source_times = self._source_times[source_index]
            payload_index = int(np.searchsorted(source_times, timestamp, side="left"))
            if (
                payload_index >= len(source_times)
                or source_times[payload_index] != timestamp
            ):
                raise IndexError(f"Timestamp {timestamp} is missing from its source shard")
            payload_ds = self._worker_sources[source_index]
        values = np.asarray(
            payload_ds["state"]
            .transpose("time", "channel", "latitude", "longitude")
            .isel(
                time=payload_index,
                latitude=slice(0, h),
                longitude=slice(0, w),
            ),
            dtype=np.float32,
        )
        if self.config.drop_file_cache:
            chunk_path = (
                self.data_paths[source_index]
                / "state"
                / f"{payload_index}.0.0.0"
            )
            if chunk_path.is_file():
                descriptor = os.open(chunk_path, os.O_RDONLY)
                try:
                    posix_fadvise = getattr(os, "posix_fadvise", None)
                    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
                    if posix_fadvise is not None and dontneed is not None:
                        posix_fadvise(descriptor, 0, 0, dontneed)
                finally:
                    os.close(descriptor)
        return values

    def __getitem__(self, index: int) -> AtmosphereSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds for dataset of length {len(self)}"
            )

        ds = self._get_dataset()
        step = self.config.time_step
        base_t = self.config.history * step + int(self.sample_indices[index])

        # Extract history inputs [t - history * step, ..., t]
        input_timesteps: list[np.ndarray] = []
        for h_step in range(self.config.history, -1, -1):
            t = base_t - h_step * step
            data = self._read_timestep_channels(ds, t)
            data = self.normalizer.normalize(data)
            input_timesteps.append(data)

        # Concatenate history along channel dimension: [(history + 1) * C, H, W]
        model_input = np.concatenate(input_timesteps, axis=0)

        # Add Gaussian noise during training if requested
        if self.training and self.config.add_noise and self.config.noise_std > 0:
            noise = np.random.normal(
                0.0, self.config.noise_std, size=model_input.shape
            ).astype(np.float32)
            model_input = model_input + noise

        # Extract rollout targets [t + 1 * step, ..., t + rollout * step]
        target_timesteps: list[np.ndarray] = []
        for r_step in range(
            1,
            self.config.rollout_steps + 1
            if self.config.require_targets and not self.config.lazy_targets
            else 1,
        ):
            t = base_t + r_step * step
            target_data = self._read_timestep_channels(ds, t)
            target_data = self.normalizer.normalize(target_data)
            target_timesteps.append(target_data)

        # Target tensor of shape [rollout_steps, C, H, W]
        model_target = (
            np.stack(target_timesteps, axis=0)
            if target_timesteps
            else np.empty(
                (0, self.config.channel_count, *self.spatial_shape), dtype=np.float32
            )
        )

        sample: AtmosphereSample = {
            "input": torch.from_numpy(
                np.ascontiguousarray(model_input, dtype=np.float32)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(model_target, dtype=np.float32)
            ),
            "time_index": torch.tensor(base_t, dtype=torch.long),
            "target_time_indices": torch.tensor(
                [
                    base_t + lead * step
                    for lead in range(1, self.config.rollout_steps + 1)
                ]
                if self.config.require_targets
                else [],
                dtype=torch.long,
            ),
        }
        # ``getattr`` preserves worker compatibility when a long-running
        # parent created this slotted config before time conditioning existed,
        # then a spawned validation worker imports the newer loader module.
        if getattr(self.config, "time_conditioning", False):
            sample["forcing_time_features"] = torch.from_numpy(
                np.stack(
                    [
                        time_forcing_phases(
                            self.times[base_t]
                            + np.timedelta64(
                                lead * step * self.cadence_nanoseconds, "ns"
                            )
                        )
                        for lead in range(1, self.config.rollout_steps + 1)
                    ],
                    axis=0,
                )
            )
        return sample


def create_data_loader(
    dataset: Dataset[AtmosphereSample],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    distributed: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader[AtmosphereSample]:
    """Create a worker-safe loader with optional asynchronous prefetching."""
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        # Forking a process after PyTorch/CUDA has initialized background
        # threads can deadlock on inherited locks. Spawn gives each Zarr worker
        # a clean interpreter and keeps CUDA state in the parent process.
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        # PyTorch only accepts prefetching/persistence with worker processes.
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
    )
