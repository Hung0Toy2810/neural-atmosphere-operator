"""Combine the two stage smoke-test CSVs into submission-ready tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import cast


PRIORITY = {
    "geopotential@500hPa": ("Z500", "gpm", 1.0 / 9.80665),
    "temperature@850hPa": ("T850", "K", 1.0),
    "2m_temperature": ("T2M", "K", 1.0),
    "10m_u_component_of_wind": ("U10", "m s^-1", 1.0),
    "10m_v_component_of_wind": ("V10", "m s^-1", 1.0),
    "mean_sea_level_pressure": ("MSLP", "hPa", 0.01),
    "specific_humidity@850hPa": ("Q850", "g kg^-1", 1000.0),
    "u_component_of_wind@850hPa": ("U850", "m s^-1", 1.0),
    "v_component_of_wind@850hPa": ("V850", "m s^-1", 1.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.expanduser().resolve()

    all_rows: list[dict[str, object]] = []
    priority_rows: list[dict[str, object]] = []
    for stage in (1, 2):
        for row in read_csv(result_dir / f"stage{stage}_metrics_by_channel.csv"):
            common = {
                "stage": f"stage{stage}",
                "lead_hours": float(row["lead_hours"]),
                "channel": int(row["channel"]),
                "variable": row["variable"],
                "unit": row["unit"],
                "rmse": float(row["physical_rmse"]),
                "acc": float(row["normalized_acc"]),
                "bias": float(row["physical_bias"]),
                "spatial_variance_ratio": float(row["spatial_variance_ratio"]),
                "initializations": int(row["acc_valid_initializations"]),
            }
            all_rows.append(common)
            if row["variable"] in PRIORITY:
                short_name, unit, scale = PRIORITY[row["variable"]]
                priority_rows.append(
                    {
                        "stage": f"stage{stage}",
                        "lead_hours": float(row["lead_hours"]),
                        "variable": short_name,
                        "unit": unit,
                        "rmse": float(row["physical_rmse"]) * scale,
                        "acc": float(row["normalized_acc"]),
                        "bias": float(row["physical_bias"]) * scale,
                        "spatial_variance_ratio": float(row["spatial_variance_ratio"]),
                        "initializations": int(row["acc_valid_initializations"]),
                    }
                )

    write_csv(result_dir / "all_stage_metrics.csv", all_rows)
    write_csv(result_dir / "priority_stage_metrics.csv", priority_rows)

    summary = read_csv(result_dir / "summary.csv")
    lines = [
        "# Two-stage metric report",
        "",
        "## Aggregate normalized metrics and baselines",
        "",
        "| Stage | Model RMSE | Model ACC | Training objective | Persistence RMSE | Persistence ACC | Climatology RMSE | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['stage']} | {float(row['normalized_rmse']):.6f} | "
            f"{float(row['normalized_acc']):.6f} | {float(row['objective_loss']):.6f} | "
            f"{float(row['persistence_rmse']):.6f} | {float(row['persistence_acc']):.6f} | "
            f"{float(row['climatology_rmse']):.6f} | {row['samples']} |"
        )

    lines.extend(
        [
            "",
            "## Physical metrics by priority variable",
            "",
            "RMSE and bias are in the displayed physical unit; ACC is latitude-weighted.",
            "",
            "| Stage | Variable | RMSE | ACC | Bias | Variance ratio | Unit | N |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    order = {name: index for index, name in enumerate(
        ("Z500", "T850", "T2M", "U10", "V10", "MSLP", "Q850", "U850", "V850")
    )}
    for row in sorted(priority_rows, key=lambda item: (item["stage"], order[str(item["variable"])])):
        lines.append(
            f"| {row['stage']} | {row['variable']} | {cast(float, row['rmse']):.4f} | "
            f"{cast(float, row['acc']):.4f} | {cast(float, row['bias']):.4f} | "
            f"{cast(float, row['spatial_variance_ratio']):.4f} | {row['unit']} | "
            f"{row['initializations']} |"
        )

    lines.extend(
        [
            "",
            "`all_stage_metrics.csv` contains the same metrics for all 71 channels",
            "(142 rows). This 6-hour table verifies each checkpoint with its trained",
            "K=1 horizon.",
        ]
    )
    (result_dir / "METRICS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
