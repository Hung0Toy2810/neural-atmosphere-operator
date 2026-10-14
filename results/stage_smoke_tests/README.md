# Two-stage checkpoint verification

Both retained checkpoints were loaded by their matching stage code and
evaluated end-to-end on real held-out ERA5 July 2020 data at 0.5° resolution.
Each test used 12 evenly spaced initializations, seed 42, deterministic MPS,
no AMP, and the checkpoint's trained horizon K=1 (6 hours). All predictions
and metrics were finite.

| Stage | Architecture | Normalized RMSE | ACC | Objective | Status |
|---|---|---:|---:|---:|---|
| 1 | SFNO E192/L6, one-state input | 0.157215 | 0.983585 | 0.140222 | Passed |
| 2 | E192/L6, history + E84 corrector | 0.154885 | 0.984121 | 0.135325 | Passed |

The per-channel CSV files retain physical RMSE, latitude-weighted ACC, bias,
and spatial variance ratio for all 71 variables. Stage 1 has no history input,
so its
evenly-spaced sliding-window indices start one time step earlier; this is a
functional checkpoint verification, not a strict scientific ablation table.

See `METRICS_REPORT.md` for the readable priority-variable table,
`priority_stage_metrics.csv` for its machine-readable form, and
`all_stage_metrics.csv` for all 142 stage/channel rows.

The evaluation runtime was local Apple MPS with Torch 2.13.0, while training
used CUDA with Torch 2.8.0. Runtime mismatch was explicitly recorded and
allowed; the `torch_harmonics` operator implementation hash matched training.
