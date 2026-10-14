"""Create the tiny 0.5-degree static-conditioning bundle for enhanced runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT.parents[1]
sys.path[:0] = [str(SUBMISSION_ROOT), str(ROOT), str(ROOT / "src")]

from configs.download_data_config import DEFAULT_ZARR_URL
from data.download_era5 import REGRIDDING_METHOD, REGRIDDING_VERSION, _regrid_data_array
from neural_atmosphere_operator.data.normalization import latitude_cell_weights


def area_standardize(values: np.ndarray, latitudes: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize one static map using exact latitude-cell area weights."""
    field = np.asarray(values, dtype=np.float64)
    weights = np.asarray(latitude_cell_weights(latitudes), dtype=np.float64)[:, None]
    denominator = float(weights.sum() * field.shape[1])
    mean = float((field * weights).sum() / denominator)
    variance = float(((field - mean) ** 2 * weights).sum() / denominator)
    std = variance**0.5
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Static field must have positive finite area variance")
    return ((field - mean) / std).astype(np.float32), mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_ZARR_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/dataset/static_forcing_0p5deg.npz",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing static forcing bundle")

    source = xr.open_zarr(
        args.source, storage_options={"token": "anon"}, consolidated=True
    )
    try:
        required = ("geopotential_at_surface", "land_sea_mask")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError("Source is missing static variables: " + ", ".join(missing))
        regridded = {
            name: np.asarray(_regrid_data_array(source[name]).compute(), dtype=np.float32)
            for name in required
        }
        latitudes = np.linspace(90.0, -90.0, 361, dtype=np.float32)
        longitudes = np.arange(720, dtype=np.float32) * 0.5
    finally:
        source.close()

    normalized_orography, orography_mean, orography_std = area_standardize(
        regridded["geopotential_at_surface"], latitudes
    )
    normalized_mask, mask_mean, mask_std = area_standardize(
        regridded["land_sea_mask"], latitudes
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".tmp.npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            normalized_surface_geopotential=normalized_orography,
            normalized_land_sea_mask=normalized_mask,
            latitude=latitudes,
            longitude=longitudes,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "version": 1,
        "source": args.source,
        "variables": list(required),
        "regridding_method": REGRIDDING_METHOD,
        "regridding_version": REGRIDDING_VERSION,
        "shape": [361, 720],
        "normalization": "spherical_latitude_cell_area_zscore",
        "surface_geopotential_mean": orography_mean,
        "surface_geopotential_std": orography_std,
        "land_sea_mask_mean": mask_mean,
        "land_sea_mask_std": mask_std,
        "bundle_sha256": digest,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_path)
    print(f"wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
