#!/usr/bin/env python3
"""Validate and summarize mountain/centroid hybrid pilot results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


METRICS = (
    "normalized_mountain_objective",
    "normalized_centroid_gain",
    "base_pair_f1",
    "normalized_squared_mountain_distance",
    "predicted_pair_count",
    "median_selected_pair_bpp",
    "fraction_selected_pair_bpp_below_0.01",
    "hybrid_seconds",
)


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            converted["alpha"] = float(row["alpha"])
            for metric in METRICS:
                converted[metric] = float(row[metric]) if row[metric] else np.nan
            rows.append(converted)
    if not rows:
        raise ValueError("hybrid metrics file is empty")

    alphas = sorted({row["alpha"] for row in rows})
    expected = set(alphas)
    by_id: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        if row["alpha"] in by_id[row["id"]]:
            raise ValueError(f"duplicate alpha for {row['id']}: {row['alpha']}")
        by_id[row["id"]].add(row["alpha"])
    incomplete = [record_id for record_id, values in by_id.items() if values != expected]
    if incomplete:
        raise ValueError(f"{len(incomplete)} records have incomplete alpha grids")
    return rows, alphas


def median(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return float(np.nanmedian(values))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_row(
    scope: str,
    family: str,
    alpha: float,
    rows: Sequence[dict[str, Any]],
    endpoint_structures: dict[float, dict[str, str]],
) -> dict[str, Any]:
    return {
        "scope": scope,
        "family": family,
        "alpha": alpha,
        "n": len(rows),
        **{f"median_{metric}": median(rows, metric) for metric in METRICS},
        "fraction_identical_to_alpha_0": float(
            np.mean(
                [
                    row["predicted_structure"] == endpoint_structures[0.0][row["id"]]
                    for row in rows
                ]
            )
        ),
        "fraction_identical_to_alpha_1": float(
            np.mean(
                [
                    row["predicted_structure"] == endpoint_structures[1.0][row["id"]]
                    for row in rows
                ]
            )
        ),
    }


def paired_row(
    alpha: float,
    endpoint_alpha: float,
    rows: Sequence[dict[str, Any]],
    rows_by_id_alpha: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    f1_differences = []
    nmsmd_differences = []
    for row in rows:
        endpoint = rows_by_id_alpha[(row["id"], endpoint_alpha)]
        f1_differences.append(row["base_pair_f1"] - endpoint["base_pair_f1"])
        nmsmd_differences.append(
            row["normalized_squared_mountain_distance"]
            - endpoint["normalized_squared_mountain_distance"]
        )
    f1 = np.asarray(f1_differences, dtype=float)
    nmsmd = np.asarray(nmsmd_differences, dtype=float)
    return {
        "alpha": alpha,
        "endpoint_alpha": endpoint_alpha,
        "n": len(rows),
        "median_delta_base_pair_f1": float(np.median(f1)),
        "fraction_higher_base_pair_f1": float(np.mean(f1 > 0.0)),
        "fraction_equal_base_pair_f1": float(np.mean(f1 == 0.0)),
        "median_delta_nmsmd": float(np.median(nmsmd)),
        "fraction_lower_nmsmd": float(np.mean(nmsmd < 0.0)),
        "fraction_equal_nmsmd": float(np.mean(nmsmd == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, alphas = load_rows(args.metrics)
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("alpha endpoints 0 and 1 are required")
    rows_by_id_alpha = {(row["id"], row["alpha"]): row for row in rows}
    endpoint_structures = {
        endpoint: {
            row["id"]: row["predicted_structure"]
            for row in rows
            if row["alpha"] == endpoint
        }
        for endpoint in (0.0, 1.0)
    }

    summary_rows = []
    for alpha in alphas:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        summary_rows.append(
            summary_row("all", "all_families", alpha, alpha_rows, endpoint_structures)
        )
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in alpha_rows:
            by_family[row["family"]].append(row)
        for family in sorted(by_family):
            summary_rows.append(
                summary_row("family", family, alpha, by_family[family], endpoint_structures)
            )

    paired_rows = []
    for alpha in alphas:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        for endpoint in (0.0, 1.0):
            paired_rows.append(
                paired_row(alpha, endpoint, alpha_rows, rows_by_id_alpha)
            )

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_summary.csv", paired_rows)
    print(
        f"Summarized {len({row['id'] for row in rows})} records, "
        f"{len(alphas)} alpha values, and {len(rows)} predictions"
    )


if __name__ == "__main__":
    main()
