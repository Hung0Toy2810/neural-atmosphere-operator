"""Render the machine-readable monthly evaluation outputs as a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def number(value: str, digits: int = 4) -> str:
    parsed = float(value)
    return "—" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    with (result_dir / "metrics_priority.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        "# Independent ERA5 evaluation — July 2020",
        "",
        "## Protocol",
        "",
        f"- Test month: {metadata['test_month']} (held out from 1983–2018 training and 2019 validation).",
        f"- Source/regrid: public WeatherBench2 ERA5 0.25°; {metadata['regridding']}.",
        f"- Grid/cadence: {metadata['grid'][0]} × {metadata['grid'][1]}, {metadata['cadence_hours']:g} hours.",
        f"- Initializations: {metadata['evaluated_initializations']} of {metadata['available_initializations']} valid 7-day windows.",
        f"- Seed/device: {metadata['seed']} / `{metadata['device']}`.",
        f"- Normalization: {metadata['normalization']}.",
        f"- RMSE/bias: {metadata['rmse_bias_weighting']}.",
        f"- ACC: {metadata['acc']}.",
        f"- Runtime: {metadata['elapsed_seconds'] / 3600:.2f} hours; evaluation Torch {metadata['torch']}.",
        "- Reproducibility note: the SFNO operator source hash matched training; the local",
        "  MPS/Torch runtime differed from the recorded CUDA/Torch 2.8.0 training runtime.",
        "",
        "## Final model",
        "",
        "| Lead | Variable | RMSE | ACC | Bias | Unit | N |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row["method"] != "model":
            continue
        lines.append(
            "| {lead} h | {variable} | {rmse} | {acc} | {bias} | {unit} | {n} |".format(
                lead=row["lead_hours"],
                variable=row["variable"],
                rmse=number(row["rmse"]),
                acc=number(row["acc"]),
                bias=number(row["bias"]),
                unit=row["unit"],
                n=row["initializations"],
            )
        )

    indexed = {
        (row["method"], int(row["lead_hours"]), row["variable"]): row
        for row in rows
    }
    lines.extend(
        [
            "",
            "## Like-for-like internal baselines",
            "",
            "All methods below use the identical initializations, targets and area weights.",
            "Skill is `100 × (1 − MSE_model / MSE_persistence)`; positive is better.",
            "The complete baseline rows are retained in both CSV files.",
            "",
            "| Lead | Variable | Model RMSE | Persistence RMSE | Climatology RMSE | Skill vs persistence | Model ACC | Persistence ACC |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lead in (24, 72, 120, 168):
        for variable in ("Z500", "T850", "T2M", "U10", "V10"):
            model = indexed[("model", lead, variable)]
            persistence = indexed[("persistence", lead, variable)]
            climatology = indexed[("training_climatology", lead, variable)]
            model_rmse = float(model["rmse"])
            persistence_rmse = float(persistence["rmse"])
            skill = 100.0 * (1.0 - (model_rmse / persistence_rmse) ** 2)
            lines.append(
                f"| {lead} h | {variable} | {model_rmse:.4f} | "
                f"{persistence_rmse:.4f} | {float(climatology['rmse']):.4f} | "
                f"{skill:.1f}% | {number(model['acc'])} | {number(persistence['acc'])} |"
            )

    lines.extend(
        [
            "",
            "Climatology ACC is undefined because its anomaly is identically zero.",
            "",
            "## Published context (not a like-for-like ranking)",
            "",
            "Pangu-Weather's official 0.25° results report the following RMSE values;",
            "Z500 has been converted from m² s⁻² to gpm using 9.80665. Those tests use",
            "different years and initialization/climatology protocols, so they must not",
            "be interpreted as a direct comparison with the July 2020 table above.",
            "",
            "| Model | Lead | Z500 (gpm) | T850 (K) | T2M (K) | U10 (m s⁻¹) |",
            "|---|---:|---:|---:|---:|---:|",
            "| Pangu-Weather | 72 h | 13.72 | 1.14 | 1.05 | 1.61 |",
            "| Pangu-Weather | 120 h | 30.25 | 1.79 | 1.53 | 2.53 |",
            "| FourCastNet | 120 h | 49.41 | — | — | — |",
            "",
            "Sources: [Pangu-Weather official repository](https://github.com/198808xc/Pangu-Weather),",
            "[Pangu-Weather paper](https://doi.org/10.1038/s41586-023-06185-3), and",
            "[FourCastNet paper](https://arxiv.org/abs/2202.11214). SFNO/FNO and GraphCast",
            "are not assigned numbers here because forecasts evaluated on this exact panel",
            "were unavailable. For a genuinely comparable ranking, rerun every model on",
            "the same July 2020 0.5° panel and score it with this evaluator.",
            "",
            "## Reproduction",
            "",
            "The exact executed command is in `command.txt`; complete runtime and sample",
            "provenance is in `run_metadata.json`.",
        ]
    )
    (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
