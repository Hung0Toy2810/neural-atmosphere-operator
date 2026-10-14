"""Compute train-only channel z-score statistics from an ERA5 Zarr store."""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import numcodecs
import numcodecs.blosc
import torch
import xarray as xr

from configs.pipeline_config import DataPaths
from configs.download_data_config import channel_names
from neural_atmosphere_operator.data.loader import (
    AtmosphereDatasetConfig,
    AtmosphereZarrDataset,
    create_data_loader,
)
from neural_atmosphere_operator.data.normalization import latitude_cell_weights
from neural_atmosphere_operator.pipeline.runtime import statistics_signature


def spherical_mean_and_std(
    field: xr.DataArray,
    latitude_weights: xr.DataArray,
) -> tuple[float, float]:
    """Compute population statistics with exact latitude-cell area weights."""
    # The public helper only needs state moments; reuse a bounded two-pass
    # reduction even for a single time slice.
    dimensions = ("time", "latitude", "longitude")
    mean = (
        field.astype("float64")
        .weighted(latitude_weights)
        .mean(dim=dimensions, skipna=False)
        .compute()
    )
    variance = (
        ((field.astype("float64") - mean) ** 2)
        .weighted(latitude_weights)
        .mean(dim=dimensions, skipna=False)
        .compute()
    )
    return float(mean), float(np.sqrt(variance))


def positional_time_difference(field: xr.DataArray, time_step: int) -> xr.DataArray:
    """Subtract samples by position without xarray timestamp re-alignment."""
    count = field.sizes["time"] - time_step
    coordinate = np.arange(count, dtype=np.int64)
    later = field.isel(time=slice(time_step, None)).assign_coords(time=coordinate)
    earlier = field.isel(time=slice(None, -time_step)).assign_coords(time=coordinate)
    return later - earlier


def spherical_channel_statistics(
    state: xr.DataArray,
    latitude_weights: xr.DataArray,
    time_step: int,
) -> xr.Dataset:
    """Stream joint-channel float64 Chan moments with O(stride * C * H * W) RAM.

    Each time slice is materialized separately. No global Dask reduction keeps
    chunks alive across the entire corpus. Delta buffers preserve positional
    stride; climatology and both moments share each decompression.
    """
    required = {"time", "channel", "latitude", "longitude"}
    if set(state.dims) != required:
        raise ValueError("State must have time/channel/latitude/longitude dimensions")
    state = state.transpose("time", "channel", "latitude", "longitude")
    if not 1 <= time_step < state.sizes["time"]:
        raise ValueError("time_step requires at least one temporal difference")
    weights = np.asarray(latitude_weights.values, dtype=np.float64)
    weights = weights / weights.sum() / state.sizes["longitude"]
    weight = weights[None, :, None]
    channels = state.sizes["channel"]
    means = [np.zeros(channels), np.zeros(channels)]
    m2 = [np.zeros(channels), np.zeros(channels)]
    counts = [0, 0]
    climate = np.zeros(tuple(state.shape[1:]), dtype=np.float64)
    previous = deque()
    payload_hash = hashlib.sha256()
    started = time.perf_counter()

    def update(values, branch):
        mean = np.sum(values * weight, axis=(1, 2))
        variance = np.sum((values - mean[:, None, None]) ** 2 * weight, axis=(1, 2))
        counts[branch] += 1
        difference = mean - means[branch]
        means[branch] += difference / counts[branch]
        m2[branch] += variance + difference**2 * (counts[branch] - 1) / counts[branch]

    for index in range(state.sizes["time"]):
        values = np.asarray(state.isel(time=index).values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Training data contains non-finite values at time index {index}"
            )
        payload_hash.update(values.astype("<f4").tobytes())
        update(values, 0)
        climate += (values - climate) / (index + 1)
        if len(previous) == time_step:
            update(values - previous.popleft(), 1)
        previous.append(values)
        completed = index + 1
        if completed % 200 == 0 or completed == state.sizes["time"]:
            elapsed = time.perf_counter() - started
            print(
                f"statistics_progress={completed}/{state.sizes['time']} "
                f"samples_per_second={completed / max(elapsed, 1e-9):.3f}",
                flush=True,
            )
    coords = {name: state[name] for name in ("channel", "latitude", "longitude")}
    return xr.Dataset(
        {
            "mean": ("channel", means[0]),
            "variance": ("channel", m2[0] / counts[0]),
            "time_diff_mean": ("channel", means[1]),
            "time_diff_variance": ("channel", m2[1] / counts[1]),
            "time_mean": (("channel", "latitude", "longitude"), climate),
        },
        coords=coords,
        attrs={"training_state_float32_sha256": payload_hash.hexdigest()},
    )


def spherical_channel_statistics_cuda_loader(
    train_paths: tuple[Path, ...],
    train_start: str,
    train_end: str,
    latitude_weights: xr.DataArray,
    time_step: int,
    *,
    batch_size: int,
    num_workers: int,
) -> xr.Dataset:
    """Feed ordered decoded states from parallel workers into CUDA reductions."""
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(
            data_path=train_paths,
            start_time=train_start,
            end_time=train_end,
            normalize=False,
            rollout_steps=1,
            time_step=time_step,
            require_targets=False,
            drop_file_cache=True,
        ),
        training=False,
    )
    loader = create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    )
    total = len(dataset)
    device = torch.device("cuda")
    weights = torch.as_tensor(
        np.asarray(latitude_weights.values), device=device, dtype=torch.float64
    )
    weights = weights / weights.sum() / dataset.spatial_shape[1]
    weight = weights[None, :, None]
    channels = dataset.config.channel_count
    means = [torch.zeros(channels, device=device, dtype=torch.float64) for _ in range(2)]
    m2 = [torch.zeros_like(means[0]), torch.zeros_like(means[0])]
    counts = [0, 0]
    climate = torch.zeros(
        (channels, *dataset.spatial_shape), device=device, dtype=torch.float64
    )
    previous: deque[torch.Tensor] = deque()
    payload_hash = hashlib.sha256()
    started = time.perf_counter()

    def update(values: torch.Tensor, branch: int) -> None:
        mean = torch.sum(values * weight, dim=(1, 2))
        variance = torch.sum(
            (values - mean[:, None, None]).square() * weight, dim=(1, 2)
        )
        counts[branch] += 1
        difference = mean - means[branch]
        means[branch].add_(difference / counts[branch])
        m2[branch].add_(
            variance
            + difference.square() * (counts[branch] - 1) / counts[branch]
        )

    completed = 0
    try:
        for batch in loader:
            for values_cpu in batch["input"]:
                values_np = values_cpu.numpy()
                if not np.isfinite(values_np).all():
                    raise ValueError(
                        "Training data contains non-finite values at time index "
                        f"{completed}"
                    )
                payload_hash.update(
                    values_np.astype("<f4", copy=False).tobytes()
                )
                values = values_cpu.to(
                    device=device, dtype=torch.float64, non_blocking=True
                )
                update(values, 0)
                climate.add_((values - climate) / (completed + 1))
                if len(previous) == time_step:
                    update(values - previous.popleft(), 1)
                previous.append(values)
                completed += 1
                if completed % 200 == 0 or completed == total:
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    print(
                        f"statistics_progress={completed}/{total} "
                        f"samples_per_second={completed / max(elapsed, 1e-9):.3f}",
                        flush=True,
                    )
    finally:
        dataset.close()
    torch.cuda.synchronize(device)
    coords = {
        "channel": list(dataset.channel_names),
        "latitude": dataset.latitudes,
        "longitude": dataset.longitudes,
    }
    return xr.Dataset(
        {
            "mean": ("channel", means[0].cpu().numpy()),
            "variance": ("channel", (m2[0] / counts[0]).cpu().numpy()),
            "time_diff_mean": ("channel", means[1].cpu().numpy()),
            "time_diff_variance": (
                "channel",
                (m2[1] / counts[1]).cpu().numpy(),
            ),
            "time_mean": (
                ("channel", "latitude", "longitude"),
                climate.cpu().numpy(),
            ),
        },
        coords=coords,
        attrs={"training_state_float32_sha256": payload_hash.hexdigest()},
    )
    total = sum(state.sizes["time"] for state in states)
    if not 1 <= time_step < total:
        raise ValueError("time_step requires at least one temporal difference")
    device = torch.device("cuda")
    weights = torch.as_tensor(
        np.asarray(latitude_weights.values), device=device, dtype=torch.float64
    )
    weights = weights / weights.sum() / states[0].sizes["longitude"]
    weight = weights[None, :, None]
    channels = states[0].sizes["channel"]
    means = [torch.zeros(channels, device=device, dtype=torch.float64) for _ in range(2)]
    m2 = [torch.zeros_like(means[0]), torch.zeros_like(means[0])]
    counts = [0, 0]
    climate = torch.zeros(
        tuple(states[0].shape[1:]), device=device, dtype=torch.float64
    )
    previous: deque[torch.Tensor] = deque()
    payload_hash = hashlib.sha256()
    started = time.perf_counter()

    def update(values: torch.Tensor, branch: int) -> None:
        mean = torch.sum(values * weight, dim=(1, 2))
        variance = torch.sum(
            (values - mean[:, None, None]).square() * weight, dim=(1, 2)
        )
        counts[branch] += 1
        difference = mean - means[branch]
        means[branch].add_(difference / counts[branch])
        m2[branch].add_(
            variance
            + difference.square() * (counts[branch] - 1) / counts[branch]
        )

    completed = 0
    for state in states:
        for local_index in range(state.sizes["time"]):
            values_np = np.asarray(state.isel(time=local_index).values, dtype=np.float32)
            if not np.isfinite(values_np).all():
                raise ValueError(
                    f"Training data contains non-finite values at time index {completed}"
                )
            payload_hash.update(values_np.astype("<f4", copy=False).tobytes())
            values = torch.as_tensor(values_np, device=device).to(torch.float64)
            update(values, 0)
            climate.add_((values - climate) / (completed + 1))
            if len(previous) == time_step:
                update(values - previous.popleft(), 1)
            previous.append(values)
            completed += 1
            if completed % 200 == 0 or completed == total:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                print(
                    f"statistics_progress={completed}/{total} "
                    f"samples_per_second={completed / max(elapsed, 1e-9):.3f}",
                    flush=True,
                )
    torch.cuda.synchronize(device)
    coords = {
        name: states[0][name] for name in ("channel", "latitude", "longitude")
    }
    return xr.Dataset(
        {
            "mean": ("channel", means[0].cpu().numpy()),
            "variance": ("channel", (m2[0] / counts[0]).cpu().numpy()),
            "time_diff_mean": ("channel", means[1].cpu().numpy()),
            "time_diff_variance": (
                "channel",
                (m2[1] / counts[1]).cpu().numpy(),
            ),
            "time_mean": (
                ("channel", "latitude", "longitude"),
                climate.cpu().numpy(),
            ),
        },
        coords=coords,
        attrs={"training_state_float32_sha256": payload_hash.hexdigest()},
    )


def spherical_channel_statistics_cuda_batched(
    train_paths: tuple[Path, ...],
    train_start: str,
    train_end: str,
    latitude_weights: xr.DataArray,
    time_step: int,
    *,
    batch_size: int,
    num_workers: int,
    reduction_batch_size: int,
    decoder_threads: int,
) -> xr.Dataset:
    """Use direct parallel Blosc decode and batch-wide CUDA FP64 reductions."""
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(
            data_path=train_paths,
            start_time=train_start,
            end_time=train_end,
            normalize=False,
            rollout_steps=1,
            time_step=time_step,
            require_targets=False,
            drop_file_cache=True,
        ),
        training=False,
    )
    del batch_size  # Retained as a backward-compatible CLI field.
    numcodecs.blosc.set_nthreads(decoder_threads)
    chunk_specs: list[tuple[Path, int]] = []
    for timestamp in dataset.times:
        source_index = int(
            np.searchsorted(dataset._source_last_times, timestamp, side="left")
        )
        source_times = dataset._source_times[source_index]
        payload_index = int(np.searchsorted(source_times, timestamp, side="left"))
        chunk_specs.append((dataset.data_paths[source_index], payload_index))
    codecs = []
    chunk_shapes = []
    dtypes = []
    for path in dataset.data_paths:
        metadata = json.loads((path / "state" / ".zarray").read_text())
        if metadata["chunks"][0] != 1 or metadata.get(
            "dimension_separator", "."
        ) != ".":
            raise ValueError("Direct CUDA statistics require canonical time=1 chunks")
        codecs.append(numcodecs.get_codec(metadata["compressor"]))
        chunk_shapes.append(tuple(metadata["chunks"]))
        dtypes.append(np.dtype(metadata["dtype"]))
    cpu_weights = np.asarray(latitude_weights.values, dtype=np.float64)
    cpu_weights = cpu_weights / cpu_weights.sum() / dataset.spatial_shape[1]
    cpu_weight = cpu_weights[None, :, None]

    def decode(
        spec: tuple[Path, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root, payload_index = spec
        source_index = dataset.data_paths.index(root)
        path = root / "state" / f"{payload_index}.0.0.0"
        payload = path.read_bytes()
        decoded = codecs[source_index].decode(payload)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise = getattr(os, "posix_fadvise", None)
            dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
            if posix_fadvise is not None and dontneed is not None:
                posix_fadvise(descriptor, 0, 0, dontneed)
        finally:
            os.close(descriptor)
        values = np.frombuffer(decoded, dtype=dtypes[source_index]).reshape(
            chunk_shapes[source_index]
        )[0]
        if not np.isfinite(values).all():
            raise ValueError(f"Training data contains non-finite values in {path}")
        values64 = values.astype(np.float64)
        sample_mean = np.sum(values64 * cpu_weight, axis=(1, 2))
        sample_variance = np.sum(
            np.square(values64 - sample_mean[:, None, None]) * cpu_weight,
            axis=(1, 2),
        )
        return values, sample_mean, sample_variance
    device = torch.device("cuda")
    weights = torch.as_tensor(
        np.asarray(latitude_weights.values), device=device, dtype=torch.float64
    )
    weights = weights / weights.sum() / dataset.spatial_shape[1]
    weight = weights[None, None, :, None]
    channels = dataset.config.channel_count
    means = [torch.zeros(channels, device=device, dtype=torch.float64) for _ in range(2)]
    m2 = [torch.zeros_like(means[0]), torch.zeros_like(means[0])]
    counts = [0, 0]
    cpu_mean = np.zeros(channels, dtype=np.float64)
    cpu_m2 = np.zeros(channels, dtype=np.float64)
    cpu_count = 0
    climate = torch.zeros(
        (channels, *dataset.spatial_shape), device=device, dtype=torch.float64
    )
    previous: torch.Tensor | None = None
    payload_hash = hashlib.sha256()
    total = len(dataset)
    completed = 0
    hashed = 0
    last_progress_bucket = 0
    started = time.perf_counter()

    def merge(values: torch.Tensor, branch: int) -> None:
        sample_means = torch.sum(values * weight, dim=(2, 3))
        sample_variances = torch.sum(
            (values - sample_means[:, :, None, None]).square() * weight,
            dim=(2, 3),
        )
        group_count = values.shape[0]
        group_mean = sample_means.mean(dim=0)
        group_m2 = torch.sum(
            sample_variances + (sample_means - group_mean).square(), dim=0
        )
        previous_count = counts[branch]
        combined_count = previous_count + group_count
        if previous_count == 0:
            means[branch].copy_(group_mean)
            m2[branch].copy_(group_m2)
        else:
            difference = group_mean - means[branch]
            m2[branch].add_(
                group_m2
                + difference.square()
                * previous_count
                * group_count
                / combined_count
            )
            means[branch].add_(difference * group_count / combined_count)
        counts[branch] = combined_count

    def reduce_batch(values_cpu: torch.Tensor) -> None:
        nonlocal completed, last_progress_bucket, previous
        values = values_cpu.to(device=device, dtype=torch.float64, non_blocking=True)
        old_count = counts[0]
        merge(values, 0)
        group_count = values.shape[0]
        group_climate = values.mean(dim=0)
        climate.add_(
            (group_climate - climate) * group_count / (old_count + group_count)
        )
        joined = values if previous is None else torch.cat((previous, values))
        if joined.shape[0] > time_step:
            merge(joined[time_step:] - joined[:-time_step], 1)
        previous = joined[-time_step:].clone()
        completed += group_count
        progress_bucket = completed // 200
        if progress_bucket > last_progress_bucket or completed == total:
            last_progress_bucket = progress_bucket
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            print(
                f"statistics_progress={completed}/{total} "
                f"samples_per_second={completed / max(elapsed, 1e-9):.3f}",
                flush=True,
            )

    pending: list[np.ndarray] = []
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            specifications = iter(chunk_specs)
            futures: deque[
                Future[tuple[np.ndarray, np.ndarray, np.ndarray]]
            ] = deque()
            for _ in range(num_workers * 2):
                try:
                    futures.append(executor.submit(decode, next(specifications)))
                except StopIteration:
                    break
            while futures:
                values_np, sample_cpu_mean, sample_cpu_variance = (
                    futures.popleft().result()
                )
                try:
                    futures.append(executor.submit(decode, next(specifications)))
                except StopIteration:
                    pass
                payload_hash.update(values_np.astype("<f4", copy=False).tobytes())
                hashed += 1
                cpu_count += 1
                cpu_shift = sample_cpu_mean - cpu_mean
                cpu_mean += cpu_shift / cpu_count
                cpu_m2 += (
                    sample_cpu_variance
                    + np.square(cpu_shift) * (cpu_count - 1) / cpu_count
                )
                pending.append(values_np)
                if len(pending) == reduction_batch_size:
                    reduce_batch(torch.from_numpy(np.stack(pending, axis=0)))
                    pending.clear()
        if pending:
            reduce_batch(torch.from_numpy(np.stack(pending, axis=0)))
    finally:
        dataset.close()
    torch.cuda.synchronize(device)
    gpu_mean = means[0].cpu().numpy()
    gpu_variance = (m2[0] / counts[0]).cpu().numpy()
    cpu_variance = cpu_m2 / cpu_count
    mean_error = float(
        np.max(np.abs(gpu_mean - cpu_mean) / np.maximum(np.abs(cpu_mean), 1e-12))
    )
    variance_error = float(
        np.max(
            np.abs(gpu_variance - cpu_variance)
            / np.maximum(np.abs(cpu_variance), 1e-24)
        )
    )
    if mean_error > 1e-8 or variance_error > 1e-8:
        raise RuntimeError(
            "Independent CPU/CUDA statistics cross-check failed: "
            f"mean={mean_error}, variance={variance_error}"
        )
    coords = {
        "channel": list(dataset.channel_names),
        "latitude": dataset.latitudes,
        "longitude": dataset.longitudes,
    }
    return xr.Dataset(
        {
            "mean": ("channel", gpu_mean),
            "variance": ("channel", gpu_variance),
            "time_diff_mean": ("channel", means[1].cpu().numpy()),
            "time_diff_variance": (
                "channel",
                (m2[1] / counts[1]).cpu().numpy(),
            ),
            "time_mean": (
                ("channel", "latitude", "longitude"),
                climate.cpu().numpy(),
            ),
        },
        coords=coords,
        attrs={
            "training_state_float32_sha256": payload_hash.hexdigest(),
            "cpu_cuda_mean_max_relative_error": mean_error,
            "cpu_cuda_variance_max_relative_error": variance_error,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DataPaths()
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        default=list(defaults.dataset_paths("train")),
        help="One or more chronologically ordered training Zarr stores",
    )
    parser.add_argument("--train-start", default=defaults.split_time_range("train")[0])
    parser.add_argument("--train-end", default=defaults.split_time_range("train")[1])
    parser.add_argument("--output-dir", type=Path, default=defaults.means.parent)
    parser.add_argument(
        "--time-step",
        type=int,
        default=1,
        help="Temporal stride used by the forecast target and delta statistics",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Reduction device; CUDA keeps full-corpus statistics bounded but faster",
    )
    parser.add_argument("--cuda-workers", type=int, default=8)
    parser.add_argument("--cuda-batch-size", type=int, default=4)
    parser.add_argument("--cuda-reduction-batch-size", type=int, default=16)
    parser.add_argument("--decoder-threads", type=int, default=2)
    args = parser.parse_args()
    if args.time_step < 1:
        raise ValueError("time-step must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA statistics requested but CUDA is unavailable")
    if min(
        args.cuda_workers,
        args.cuda_batch_size,
        args.cuda_reduction_batch_size,
        args.decoder_threads,
    ) < 1:
        raise ValueError("CUDA workers and batch sizes must be positive")
    train_paths = tuple(path.expanduser().resolve() for path in args.train_data)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Statistics bundles are immutable; choose a new output directory: {output_dir}"
        )
    sources = tuple(
        xr.open_zarr(str(path), consolidated=True) for path in train_paths
    )
    if len(sources) == 1:
        combined = sources[0]
    else:
        combined = xr.concat(
            sources,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
            join="exact",
            combine_attrs="override",
        )
    ds = combined.sel(time=slice(args.train_start, args.train_end))
    means: list[float] = []
    stds: list[float] = []
    time_diff_means: list[float] = []
    time_diff_stds: list[float] = []
    time_means: list[np.ndarray] = []
    names: list[str] = []
    units: list[str] = []
    if ds.sizes["time"] <= args.time_step:
        ds.close()
        raise ValueError("Training split is empty or too short for time-step")
    first_time = str(ds.time.values[0])
    last_time = str(ds.time.values[-1])
    timesteps = int(ds.sizes["time"])
    if timesteps <= args.time_step:
        ds.close()
        raise ValueError("Training split is too short for the requested time-step")
    time_values = np.asarray(ds.time.values)
    if not np.issubdtype(time_values.dtype, np.datetime64):
        ds.close()
        raise ValueError("ERA5 time coordinate must use datetime64 values")
    time_nanoseconds = time_values.astype("datetime64[ns]").astype(np.int64)
    time_differences = np.diff(time_nanoseconds)
    if not np.all(time_differences > 0) or not np.all(
        time_differences == time_differences[0]
    ):
        ds.close()
        raise ValueError("Training timestamps must be strictly increasing and regular")
    cadence_hours = float(time_differences[0] / 3_600_000_000_000)
    latitudes = np.asarray(ds.latitude.values, dtype=np.float64)
    longitudes = np.asarray(ds.longitude.values, dtype=np.float64)
    area_weights = np.asarray(latitude_cell_weights(latitudes), dtype=np.float64)
    latitude_weights = xr.DataArray(
        area_weights,
        dims=("latitude",),
        coords={"latitude": ds.latitude},
    )
    try:
        if "state" not in ds:
            raise ValueError(
                "Dataset must contain the canonical flattened 71-channel state"
            )
        names = [str(name) for name in ds.channel.values]
        if tuple(names) != channel_names():
            raise ValueError("Stored state channels do not match the 71-channel contract")
        if "channel_units" not in ds.coords:
            raise ValueError(
                "Flattened state is missing per-channel units; regenerate with the current downloader"
            )
        units = [str(unit) for unit in ds.channel_units.values]
        state = ds["state"]
        if len(units) != len(names) or any(not unit.strip() for unit in units):
            raise ValueError("channel_units must be non-empty and align with channel")
        if args.device == "cuda":
            computed = spherical_channel_statistics_cuda_batched(
                train_paths,
                args.train_start,
                args.train_end,
                latitude_weights,
                args.time_step,
                batch_size=args.cuda_batch_size,
                num_workers=args.cuda_workers,
                reduction_batch_size=args.cuda_reduction_batch_size,
                decoder_threads=args.decoder_threads,
            )
        else:
            computed = spherical_channel_statistics(
                state, latitude_weights, args.time_step
            )
        means = np.asarray(computed["mean"]).tolist()
        stds = np.sqrt(np.asarray(computed["variance"])).tolist()
        time_diff_means = np.asarray(computed["time_diff_mean"]).tolist()
        time_diff_stds = np.sqrt(np.asarray(computed["time_diff_variance"])).tolist()
        time_means = list(np.asarray(computed["time_mean"]))
        payload_sha256 = computed.attrs["training_state_float32_sha256"]
        for index, name in enumerate(names):
            print(
                f"{name}: mean={means[index]:.7g} std={stds[index]:.7g} "
                f"delta_mean={time_diff_means[index]:.7g} delta_std={time_diff_stds[index]:.7g}"
            )
    finally:
        ds.close()
        for source in sources:
            if source is not ds:
                source.close()
    means_array = np.asarray(means, dtype=np.float32)
    stds_array = np.asarray(stds, dtype=np.float32)
    time_diff_means_array = np.asarray(time_diff_means, dtype=np.float32)
    time_diff_stds_array = np.asarray(time_diff_stds, dtype=np.float32)
    time_means_array = np.stack(time_means, axis=0).astype(np.float32)
    arrays = (
        means_array,
        stds_array,
        time_diff_means_array,
        time_diff_stds_array,
        time_means_array,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Training data contains non-finite values")
    if np.any(stds_array <= 0) or np.any(time_diff_stds_array <= 0):
        raise ValueError("Every training channel must have positive variance")
    destination = output_dir
    if destination.exists():
        raise FileExistsError(
            f"Statistics bundles are immutable; choose a new output directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=destination.name + ".part-", dir=destination.parent)
    )
    try:
        np.save(staging_dir / "means.npy", means_array)
        np.save(staging_dir / "stds.npy", stds_array)
        np.save(staging_dir / "time_means.npy", time_means_array)
        np.save(
            staging_dir / f"time_diff_means_dt{args.time_step}.npy",
            time_diff_means_array,
        )
        np.save(
            staging_dir / f"time_diff_stds_dt{args.time_step}.npy",
            time_diff_stds_array,
        )
        (staging_dir / "channels.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )
        (staging_dir / "stats.json").write_text(
            json.dumps(
                {
                "source": [str(path) for path in train_paths],
                "training_state_float32_sha256": payload_sha256,
                "latitude": latitudes.astype(np.float32).tolist(),
                "longitude": longitudes.astype(np.float32).tolist(),
                "artifacts": {
                    name: statistics_signature(staging_dir / filename)["sha256"]
                    for name, filename in {
                        "means": "means.npy",
                        "stds": "stds.npy",
                        "climatology": "time_means.npy",
                        "time_diff_stds": f"time_diff_stds_dt{args.time_step}.npy",
                        "time_diff_means": f"time_diff_means_dt{args.time_step}.npy",
                    }.items()
                },
                "first_time": first_time,
                "last_time": last_time,
                "timesteps": timesteps,
                "channels": names,
                "channel_units": units,
                "spatial_weighting": "spherical_latitude_cell_area",
                "statistics_version": 4,
                "computation": "streaming_joint_channel_float64_chan",
                "climatology": "training_long_term_mean_by_grid_cell",
                "time_difference_step": args.time_step,
                "cadence_hours": cadence_hours,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(staging_dir, destination)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(f"Statistics written to {destination}")


if __name__ == "__main__":
    main()
