"""Reproducible SFNO-style evaluation on one independently downloaded ERA5 month.

The script reports latitude-area-weighted RMSE, anomaly correlation (ACC), and
bias in physical units for every channel and requested lead.  ACC subtracts the
training-only long-term spatial climatology, matching the SFNO convention.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = SUBMISSION_ROOT / "stages" / "stage2"
sys.path[:0] = [str(STAGE2_ROOT), str(STAGE2_ROOT / "src")]

from neural_atmosphere_operator.data.loader import (  # noqa: E402
    AtmosphereDatasetConfig,
    AtmosphereZarrDataset,
)
from neural_atmosphere_operator.models.conditioning import (  # noqa: E402
    load_static_forcing,
)
from neural_atmosphere_operator.pipeline.checkpoint import load_checkpoint  # noqa: E402
from neural_atmosphere_operator.pipeline.dependencies import (  # noqa: E402
    checkpoint_runtime_mismatches,
    validate_supported_runtime,
)
from neural_atmosphere_operator.pipeline.metrics import (  # noqa: E402
    LeadMetricAccumulator,
    channel_stds_tensor,
    normalized_climatology_tensor,
)
from neural_atmosphere_operator.pipeline.runtime import (  # noqa: E402
    build_model,
    choose_device,
    model_config_from_dict,
)

DEFAULT_LEADS = (1, 2, 4, 12, 20, 28)
G0 = 9.80665
DISPLAY_CHANNELS = {
    "Z500": ("geopotential@500hPa", 1.0 / G0, "gpm"),
    "T850": ("temperature@850hPa", 1.0, "K"),
    "T2M": ("2m_temperature", 1.0, "K"),
    "U10": ("10m_u_component_of_wind", 1.0, "m s^-1"),
    "V10": ("10m_v_component_of_wind", 1.0, "m s^-1"),
    "MSLP": ("mean_sea_level_pressure", 0.01, "hPa"),
    "Q850": ("specific_humidity@850hPa", 1000.0, "g kg^-1"),
    "U850": ("u_component_of_wind@850hPa", 1.0, "m s^-1"),
    "V850": ("v_component_of_wind@850hPa", 1.0, "m s^-1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SUBMISSION_ROOT / "checkpoints/stage2/best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="One monthly store or chronologically ordered, non-overlapping shards",
    )
    parser.add_argument("--stats-dir", type=Path, default=SUBMISSION_ROOT / "data/stats")
    parser.add_argument("--static-forcing", type=Path, default=SUBMISSION_ROOT / "data/static_forcing_0p5deg.npz")
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION_ROOT / "results/era5_2020_07")
    parser.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS))
    parser.add_argument("--max-samples", type=int, help="Optional evenly spaced diagnostic subset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum < 1:
        raise ValueError("max-samples must be positive")
    return np.linspace(0, length - 1, maximum, dtype=int).tolist()


def main() -> None:
    args = parse_args()
    leads = tuple(sorted(set(args.leads)))
    if not leads or leads[0] < 1:
        raise ValueError("All lead steps must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics_all_channels.csv").exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")

    device = choose_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    saved_runtime = checkpoint.get("runtime", {})
    current_runtime = validate_supported_runtime(device)
    runtime_mismatches = checkpoint_runtime_mismatches(saved_runtime, current_runtime)
    if runtime_mismatches:
        print("Runtime differences from training: " + ", ".join(runtime_mismatches), flush=True)
    model_config = model_config_from_dict(checkpoint["model_config"])
    training_config = checkpoint.get("training_config", {})
    if int(training_config.get("time_step", 1)) != 1:
        raise ValueError("This protocol requires the checkpoint's six-hour time step")
    model = build_model(model_config, device)
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    gc.collect()
    if model_config.forcing_channels:
        model.set_static_forcing(load_static_forcing(args.static_forcing, model_config.img_size).to(device))
    model.eval()

    stats_dir = args.stats_dir.expanduser().resolve()
    means = np.load(stats_dir / "means.npy")
    stds = np.load(stats_dir / "stds.npy")
    climatology = np.load(stats_dir / "time_means.npy")
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(
            data_path=tuple(path.expanduser().resolve() for path in args.data),
            means_path=stats_dir / "means.npy",
            stds_path=stats_dir / "stds.npy",
            history=model_config.in_channels // model_config.out_channels - 1,
            rollout_steps=max(leads),
            time_step=1,
            normalize=True,
            lazy_targets=True,
            time_conditioning=model_config.forcing_channels > 0,
        ),
        training=False,
    )
    if dataset.spatial_shape != model_config.img_size:
        raise ValueError(f"Grid mismatch: data={dataset.spatial_shape}, model={model_config.img_size}")
    if not (str(dataset.times[0]).startswith("2020-07") and str(dataset.times[-1]).startswith("2020-07")):
        raise ValueError("Evaluation store must contain only July 2020")

    selected_indices = choose_indices(len(dataset), args.max_samples)
    channel_stds = channel_stds_tensor(stds, device)
    normalized_climatology = normalized_climatology_tensor(climatology, means, stds, device)
    latitudes = torch.as_tensor(dataset.latitudes, device=device)
    methods = ("model", "persistence", "training_climatology")
    accumulators = {
        method: {lead: LeadMetricAccumulator() for lead in leads} for method in methods
    }
    start = time.perf_counter()

    with torch.inference_mode():
        completed = 0
        for batch_start in range(0, len(selected_indices), args.batch_size):
            batch_indices = selected_indices[batch_start : batch_start + args.batch_size]
            samples = [dataset[index] for index in batch_indices]
            state = torch.stack([sample["input"] for sample in samples]).to(device)
            persistence = state[:, -model_config.out_channels :].clone()
            forcing_items = [sample.get("forcing_time_features") for sample in samples]
            if any(value is None for value in forcing_items) and not all(
                value is None for value in forcing_items
            ):
                raise ValueError("Batch has inconsistent forcing_time_features")
            forcing = (
                None
                if forcing_items[0] is None
                else torch.stack(
                    [value for value in forcing_items if value is not None]
                ).to(device)
            )
            target_indices = [sample["target_time_indices"].tolist() for sample in samples]
            source = dataset._get_dataset()

            for step in range(1, max(leads) + 1):
                step_forcing = forcing[:, step - 1] if forcing is not None else None
                prediction = model(state, step_forcing)
                if model_config.in_channels > model_config.out_channels:
                    state = torch.cat((state[:, model_config.out_channels :], prediction), dim=1)
                else:
                    state = prediction
                if step not in accumulators["model"]:
                    continue
                raw_targets = [
                    dataset.normalizer.normalize(
                        dataset._read_timestep_channels(source, int(indices[step - 1]))
                    )
                    for indices in target_indices
                ]
                target = torch.from_numpy(np.stack(raw_targets)).to(device)
                for method, candidate in (
                    ("model", prediction),
                    ("persistence", persistence),
                    (
                        "training_climatology",
                        normalized_climatology.expand(target.shape[0], -1, -1, -1),
                    ),
                ):
                    accumulators[method][step].update(
                        candidate.float(), target.float(), channel_stds, latitudes, normalized_climatology
                    )
            completed += len(samples)
            if completed % 10 < len(samples) or completed == len(selected_indices):
                elapsed = time.perf_counter() - start
                print(f"{completed}/{len(selected_indices)} initializations, {elapsed:.1f}s", flush=True)

    all_rows: list[dict[str, Any]] = []
    for method in methods:
        for lead in leads:
            for row in accumulators[method][lead].channel_results(
                lead, dataset.channel_names, dataset.cadence_hours, dataset.channel_units
            ):
                all_rows.append({"method": method, **row})
    write_csv(output_dir / "metrics_all_channels.csv", all_rows)

    lookup = {(row["method"], row["lead"], row["variable"]): row for row in all_rows}
    priority_rows: list[dict[str, Any]] = []
    for method in methods:
        for lead in leads:
            for label, (channel, scale, unit) in DISPLAY_CHANNELS.items():
                source_row = lookup[(method, lead, channel)]
                priority_rows.append(
                    {
                        "method": method,
                        "lead_hours": int(lead * dataset.cadence_hours),
                        "variable": label,
                        "unit": unit,
                        "rmse": float(source_row["physical_rmse"]) * scale,
                        "acc": source_row["normalized_acc"],
                        "bias": float(source_row["physical_bias"]) * scale,
                        "initializations": accumulators[method][lead].samples,
                        "acc_valid_initializations": source_row[
                            "acc_valid_initializations"
                        ],
                    }
                )
    write_csv(output_dir / "metrics_priority.csv", priority_rows)

    metadata = {
        "protocol": "SFNO-style long-term-climatology ACC",
        "test_month": "2020-07",
        "reason": "Held-out 2020 test month, more than six months after the 2019 validation boundary.",
        "training_range": "1983-01-01..2018-12-31",
        "validation_range": "2019-01-01..2019-12-31",
        "data_source": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        "source_grid": [721, 1440],
        "regridding": "first-order conservative area overlap to aligned periodic 0.5-degree grid",
        "state_channels": 71,
        "raw_data_in_submission": False,
        "cadence_hours": dataset.cadence_hours,
        "grid": list(dataset.spatial_shape),
        "available_initializations": len(dataset),
        "evaluated_initializations": len(selected_indices),
        "batch_size": args.batch_size,
        "selected_indices": selected_indices,
        "initialization_times": [
            str(
                dataset.times[
                    dataset.config.history * dataset.config.time_step
                    + int(dataset.sample_indices[index])
                ]
            )
            for index in selected_indices
        ],
        "lead_steps": list(leads),
        "seed": args.seed,
        "deterministic_algorithms": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "data_stores": [path.expanduser().resolve().name for path in args.data],
        "normalization": "training-only per-channel mean/std; metrics restored to physical units",
        "rmse_bias_weighting": "exact latitude-cell-area weights, pooled over initializations and longitude",
        "acc": "mean per-initialization spatial anomaly correlation after subtracting training-only long-term spatial climatology",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "training_runtime": saved_runtime,
        "evaluation_runtime": current_runtime,
        "runtime_mismatches": list(runtime_mismatches),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Keep the recorded invocation portable and free of workstation-specific paths.
    command = shlex.join(["python", *sys.argv]) + "\n"
    (output_dir / "command.txt").write_text(command, encoding="utf-8")
    dataset.close()


if __name__ == "__main__":
    main()
