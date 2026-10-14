# Evaluation protocol

## Split and samples

- Training: 1983-01-01 through 2018-12-31.
- Validation/model selection: 2019-01-01 through 2019-12-31.
- Test: July 2020, never used for normalization or model selection.
- ERA5 source cadence: 6 hours; target grid: global 361 × 720 (0.5°).
- Default panel: all 95 valid initializations from 2020-07-01 06:00 through
  2020-07-24 18:00, so every 168-hour verification remains within July.
- A smaller evenly spaced panel is allowed only when explicitly recorded by
  `--max-samples`; the generated metadata stores every selected index/time.

## State and normalization

The prognostic state has 71 channels: six surface variables and u, v,
geopotential, temperature, and specific humidity at 13 pressure levels. Input
and output use the fixed training-only per-channel mean and standard deviation.
The final model consumes two states (t−6 h and t) and advances autoregressively
in six-hour increments. Static topography/land-sea fields and analytic solar/time
features are supplied exactly as in training.

## Metrics

For latitude cell weights `w(i)` derived from exact latitude-cell boundaries:

```text
RMSE = sqrt(sum[n,i,j] w(i) (forecast - truth)^2 / sum[n,i,j] w(i))
bias =      sum[n,i,j] w(i) (forecast - truth)   / sum[n,i,j] w(i)
```

ACC follows the SFNO convention. For each initialization and channel, subtract
the training-only long-term spatial climatology from forecast and truth,
compute their latitude-weighted spatial correlation, then average correlations
over initializations. This differs from WeatherBench 2's day-of-year/hour
climatology and must be stated when comparing scores.

Metrics are emitted in native physical units for all 71 channels. The priority
table converts geopotential at 500 hPa to geopotential metres, MSLP to hPa, and
specific humidity to g/kg. Leads are 6, 12, 24, 72, 120, and 168 hours.

## Baselines and external comparisons

Persistence and training climatology are recomputed from the identical test
panel. Published Pangu-Weather, FourCastNet, GraphCast, and WeatherBench 2
scores generally use different years, 0.25° resolution, initialization times,
and climatologies; they are contextual references, not rows in the like-for-like
main table unless forecasts are rerun under this exact protocol.

- WeatherBench 2: https://arxiv.org/abs/2308.15560
- SFNO: https://proceedings.mlr.press/v202/bonev23a.html
- FourCastNet: https://arxiv.org/abs/2202.11214
- Pangu-Weather: https://doi.org/10.1038/s41586-023-06185-3
- GraphCast: https://doi.org/10.1126/science.adi2336
