#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_path="${ERA5_TEST_DATA:-$project_root/data/raw/era5_2020_07_0p5deg_71ch.zarr}"
device="${EVAL_DEVICE:-cuda}"

cd "$project_root"
"$python_bin" eval/evaluate_era5_month.py \
  --data "$data_path" \
  --checkpoint checkpoints/stage2/best.pt \
  --stats-dir data/stats \
  --static-forcing data/static_forcing_0p5deg.npz \
  --output-dir results/era5_2020_07 \
  --leads 1 2 4 12 20 28 \
  --batch-size "${EVAL_BATCH_SIZE:-1}" \
  --seed 42 \
  --device "$device"
"$python_bin" eval/render_report.py results/era5_2020_07
