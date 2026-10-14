# Data

Raw ERA5 data is deliberately excluded from the submission package. Download
the independent July 2020 test month from the public WeatherBench2 archive:

```bash
python data/download_era5.py \
  --start-date 2020-07-01 --end-date 2020-07-31 \
  --output data/raw/era5_2020_07_0p5deg_71ch.zarr \
  --batch-days 7 --confirm-full-download
```

The downloader selects the fixed 71-channel, 6-hourly ERA5 state and performs
first-order conservative area-overlap regridding from 0.25° to 0.5°. The
included `stats/` bundle was computed only from the 1983–2018 training split;
it must not be recomputed from test data.
