# Two-stage metric report

## Aggregate normalized metrics and baselines

| Stage | Model RMSE | Model ACC | Training objective | Persistence RMSE | Persistence ACC | Climatology RMSE | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 0.157215 | 0.983585 | 0.140222 | 0.331214 | 0.923327 | 0.761003 | 12 |
| stage2 | 0.154885 | 0.984121 | 0.135325 | 0.332342 | 0.922711 | 0.761305 | 12 |

## Physical metrics by priority variable

RMSE and bias are in the displayed physical unit; ACC is latitude-weighted.

| Stage | Variable | RMSE | ACC | Bias | Variance ratio | Unit | N |
|---|---|---:|---:|---:|---:|---|---:|
| stage1 | Z500 | 3.4934 | 0.9997 | 0.0599 | 0.9953 | gpm | 12 |
| stage1 | T850 | 0.5804 | 0.9966 | 0.0081 | 0.9938 | K | 12 |
| stage1 | T2M | 0.6439 | 0.9963 | 0.0456 | 0.9980 | K | 12 |
| stage1 | U10 | 0.7507 | 0.9837 | 0.0040 | 0.9903 | m s^-1 | 12 |
| stage1 | V10 | 0.8017 | 0.9810 | -0.0027 | 0.9823 | m s^-1 | 12 |
| stage1 | MSLP | 0.4555 | 0.9982 | -0.0177 | 0.9972 | hPa | 12 |
| stage1 | Q850 | 0.6033 | 0.9742 | -0.0013 | 0.9824 | g kg^-1 | 12 |
| stage1 | U850 | 1.1411 | 0.9836 | 0.0448 | 0.9898 | m s^-1 | 12 |
| stage1 | V850 | 1.2003 | 0.9766 | -0.0220 | 0.9727 | m s^-1 | 12 |
| stage2 | Z500 | 3.4330 | 0.9997 | -0.0065 | 1.0003 | gpm | 12 |
| stage2 | T850 | 0.5744 | 0.9966 | -0.0030 | 0.9977 | K | 12 |
| stage2 | T2M | 0.6200 | 0.9966 | 0.0198 | 0.9982 | K | 12 |
| stage2 | U10 | 0.7290 | 0.9846 | 0.0080 | 0.9899 | m s^-1 | 12 |
| stage2 | V10 | 0.7768 | 0.9822 | 0.0165 | 0.9776 | m s^-1 | 12 |
| stage2 | MSLP | 0.4547 | 0.9982 | -0.0133 | 1.0019 | hPa | 12 |
| stage2 | Q850 | 0.5784 | 0.9762 | 0.0006 | 0.9818 | g kg^-1 | 12 |
| stage2 | U850 | 1.1160 | 0.9843 | 0.0448 | 0.9844 | m s^-1 | 12 |
| stage2 | V850 | 1.1735 | 0.9777 | 0.0205 | 0.9625 | m s^-1 | 12 |

`all_stage_metrics.csv` contains the same metrics for all 71 channels
(142 rows). This 6-hour table verifies each checkpoint with its trained
K=1 horizon.
