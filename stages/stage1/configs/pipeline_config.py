"""Shared defaults for the atmospheric training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from configs.download_data_config import (
    DEFAULT_DATASET_FILENAME,
    HISTORICAL_DATASET_FILENAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = Path(
    os.environ.get("NAO_DATA_DIR", PROJECT_ROOT / "data" / "dataset")
).expanduser()


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Conventional filesystem layout used by pipeline commands."""

    # The persistent network volume is intentionally not a runtime data source.
    # Keep the training corpus on the container NVMe unless a caller provides an
    # explicit DataPaths root for an isolated test.
    root: Path = field(default_factory=lambda: LOCAL_DATA_ROOT)

    @property
    def dataset(self) -> Path:
        return self.root / DEFAULT_DATASET_FILENAME

    @property
    def historical_dataset(self) -> Path:
        return self.root / HISTORICAL_DATASET_FILENAME

    def dataset_paths(self, split: str) -> tuple[Path, ...]:
        """Return chronologically ordered physical stores for a logical split."""
        if split == "train":
            return (self.historical_dataset, self.dataset)
        if split in {"valid", "test"}:
            return (self.dataset,)
        raise ValueError(f"Unknown split: {split}")

    @property
    def train(self) -> Path:
        return self.dataset

    @property
    def valid(self) -> Path:
        return self.dataset

    @property
    def test(self) -> Path:
        return self.dataset

    def split_time_range(self, split: str) -> tuple[str, str]:
        """Return non-overlapping inclusive logical ranges across the stores."""
        ranges = {
            "train": ("1983-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2020-12-31"),
        }
        try:
            return ranges[split]
        except KeyError as error:
            raise ValueError(f"Unknown split: {split}") from error

    @property
    def means(self) -> Path:
        return self.root / "stats" / "means.npy"

    @property
    def stds(self) -> Path:
        return self.root / "stats" / "stds.npy"

    @property
    def climatology(self) -> Path:
        return self.root / "stats" / "time_means.npy"

    def time_diff_means(self, time_step: int = 1) -> Path:
        return self.root / "stats" / f"time_diff_means_dt{time_step}.npy"

    def time_diff_stds(self, time_step: int = 1) -> Path:
        return self.root / "stats" / f"time_diff_stds_dt{time_step}.npy"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Conservative defaults for the 36-year 0.5-degree SFNO corpus."""

    # Training is operator-controlled and continues until manually interrupted.
    epochs: int = 0
    # An infinite run still needs a finite cosine-decay horizon. After this
    # many epochs the scheduler remains at min_learning_rate.
    scheduler_epochs: int = 50
    # Full loss/backward pipeline maximum on the 31.37 GiB RTX PRO 4500: B8
    # exhausts 31.24 GiB, so B7 is the largest safe measured training batch.
    batch_size: int = 7
    validation_batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    gradient_clip: float = 1.0
    rollout_steps: int = 1
    history: int = 0
    time_step: int = 1
    # Four workers sustain ~10.5 full-grid samples/s, comfortably above the
    # K1 GPU consumption rate while preserving 10 GiB cgroup RAM headroom.
    num_workers: int = 4
    validation_num_workers: int = 2
    prefetch_factor: int = 1
    drop_file_cache: bool = True
    # Training workers are recreated at epoch boundaries so checkpointed
    # sampler/worker RNG state produces the same next epoch after a restart.
    persistent_workers: bool = False
    # Zero disables automatic early stopping; the operator controls termination.
    patience: int = 0
    early_stopping_min_delta: float = 0.0
    rollout_discount: float = 1.0
    channel_weighting: str = "graphcast"
    input_noise_std: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    amp_dtype: str = "bfloat16"
