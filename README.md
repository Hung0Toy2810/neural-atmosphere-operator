# Neural Atmosphere Operator

This two-stage weather forecasting project includes an SFNO baseline and a
modified model with history inputs, forcing and a spherical correction network.
Code, reports and evaluation tables are stored in this repository. The retained
checkpoints and inference statistics are distributed through GitHub Releases.

## Layout

```text
stages/stage1/       SFNO E192/L6 baseline
stages/stage2/       final model: history/forcing/spherical corrector
checkpoints/         stage1/best.pt, stage2/best.pt
configs/             evaluation and download configuration
data/                downloader, training statistics, static forcing (no raw data)
eval/                monthly evaluation implementation
scripts/             reproducible entry points
results/             newly generated tables, metadata, command and report
```

## Download checkpoints and inference data

From the repository root, restore the retained binary artifacts before running
inference or checking `MANIFEST.sha256`:

```bash
gh release download artifacts-v1 \
  --repo Hung0Toy2810/neural-atmosphere-operator \
  --pattern neural-inference-artifacts.tar.gz
tar -xzf neural-inference-artifacts.tar.gz
rm neural-inference-artifacts.tar.gz
```

The archive restores `checkpoints/`, `data/stats/` and
`data/static_forcing_0p5deg.npz`. Raw ERA5 data is downloaded separately.

## Reports and archived results

- [Full report](reports/NeuralAtmosphereOperator_Full_Report.pdf)
- [Archived 2020 benchmark](reports/evidence/rollout_2020/)
- [Checkpoint verification](results/stage_smoke_tests/)

The archived benchmark contains 60 forecast starting times and leads up to
seven days. At seven days, Z500 RMSE is 72.674 geopotential metres for Stage 2
and 75.012 for Stage 1 (3.1% lower). This archived benchmark and the 12-case
six-hour checkpoint verification are separate evaluations. Archived metadata
does not contain complete timestamps and checkpoint hashes; see the
[report review](reports/COMPACT_REPORT_REVIEW.md) for the evidence scope.

## Installation

Python 3.11 and a CUDA GPU are recommended. The submitted checkpoint is large;
CPU evaluation is supported but impractically slow for the full month.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch==2.8.0` and `torch-harmonics==0.7.4` are pinned because the SHT API and
checkpoint numerics depend on them. Verify retained binary artifacts with
`shasum -a 256 -c MANIFEST.sha256`.

The submitted final model is `checkpoints/stage2/best.pt`. Both retained
checkpoints have smoke-test records under `results/stage_smoke_tests/`.

## Data and held-out month

The test month is **July 2020**. Training is 1983–2018 and validation is 2019,
so July 2020 is held out and lies more than six months beyond the validation
boundary. Keeping all verification times inside July yields 95 valid 7-day
initializations at six-hour cadence.

```bash
mkdir -p data/raw
python data/download_era5.py \
  --start-date 2020-07-01 --end-date 2020-07-31 \
  --output data/raw/era5_2020_07_0p5deg_71ch.zarr \
  --batch-days 7 --confirm-full-download
```

The source is the public WeatherBench2 ERA5 archive. The fixed 71-channel state
is conservatively regridded from 0.25° to 0.5° and stored at 6-hour cadence.

## Evaluation

```bash
PYTHON_BIN=.venv/bin/python EVAL_DEVICE=cuda bash scripts/run_evaluation.sh
```

The fixed seed is 42. Leads are 6, 12, 24, 72, 120, and 168 hours. Outputs
include all 71 channels plus a presentation table for Z500, T850, T2M, U10,
V10, MSLP, Q850, U850, and V850. Persistence and training-climatology baselines
are evaluated on exactly the same initializations.

The Stage 2 monthly evaluator is included, but no multi-lead monthly result is
bundled in this package. Run it without `--max-samples` to score all 95 valid
windows under the documented protocol.

Both retained checkpoints were tested through their matching stage
evaluator on 12 held-out ERA5 initializations at 6 hours. See
[`results/stage_smoke_tests`](results/stage_smoke_tests/) for the summary and
all 71 per-channel rows, including physical RMSE, ACC, bias and spatial
variance ratio.

RMSE and bias are pooled with exact latitude-cell-area weights. ACC is the mean
of per-initialization spatial anomaly correlations after subtracting the
training-only long-term spatial climatology (the SFNO convention). Values are
restored to physical units; Z500 is additionally reported in geopotential
metres, MSLP in hPa, and specific humidity in g/kg.

For a quick pipeline check only, append `--max-samples 1 --device cpu` to the
Python command. Do not present that diagnostic subset as the monthly result.

## Training

Each stage is independently installable and uses explicit paths:

```bash
cd stages/stage1
python -m pip install -e .
python scripts/train.py --data-dir /path/to/training-data --run-dir runs/stage1
```

Repeat in `stage2`, passing Stage 1's selected checkpoint through
`--init-checkpoint`. See the stage README. `NAO_DATA_DIR`
may replace `--data-dir`; no machine-specific path is embedded in executable
defaults.

## Reproducibility notes

- Model selection used validation only; July 2020 is evaluation-only.
- The included statistics are from 1983–2018 and are never recomputed on test.
- A monthly run writes `metrics_all_channels.csv`, `metrics_priority.csv`,
  `run_metadata.json`, and the exact command under `results/era5_2020_07/`.
- Literature numbers should not be mixed into the main table unless their
  resolution, initialization schedule, climatology and lead protocol match.
