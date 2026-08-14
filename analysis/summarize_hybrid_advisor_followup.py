#!/usr/bin/env python3
"""Summarize low-BPP pairs and paired gamma-2 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.evaluate_gamma_centroid import file_sha256
from analysis.summarize_hybrid_objective_pilot import load_rows, write_csv


def load_gamma2_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load exactly one gamma-2 result per benchmark record."""
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["gamma"]) != 2.0:
                continue
            record_id = row["id"]
            if record_id in rows:
                raise ValueError(f"duplicate gamma-2 row for {record_id}")
            rows[record_id] = {
                **row,
                "base_pair_f1": float(row["base_pair_f1"]),
                "normalized_squared_mountain_distance": float(
                    row["normalized_squared_mountain_distance"]
                ),
            }
    if not rows:
        raise ValueError("gamma metrics contain no gamma-2 rows")
    return rows


def validate_record_match(
    hybrid_rows: Sequence[dict[str, Any]],
    gamma2_rows: dict[str, dict[str, Any]],
) -> None:
    """Require identical records and annotations in both inputs."""
    hybrid_records = {}
    for row in hybrid_rows:
        identity = (row["sequence"], row["reference_structure"])
        previous = hybrid_records.setdefault(row["id"], identity)
        if previous != identity:
            raise ValueError(f"inconsistent hybrid record identity for {row['id']}")
    if set(hybrid_records) != set(gamma2_rows):
        raise ValueError("hybrid and gamma-2 inputs contain different record IDs")
    for record_id, identity in hybrid_records.items():
        gamma_identity = (
            gamma2_rows[record_id]["sequence"],
            gamma2_rows[record_id]["reference_structure"],
        )
        if identity != gamma_identity:
            raise ValueError(f"hybrid and gamma-2 records differ for {record_id}")


def summarize_alpha(
    alpha: float,
    hybrid_rows: Sequence[dict[str, Any]],
    gamma2_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return low-BPP and paired comparison statistics for one alpha."""
    low_bpp_fractions = np.asarray(
        [
            0.0
            if np.isnan(row["fraction_selected_pair_bpp_below_0.01"])
            and row["predicted_pair_count"] == 0.0
            else row["fraction_selected_pair_bpp_below_0.01"]
            for row in hybrid_rows
        ],
        dtype=float,
    )
    if np.any(np.isnan(low_bpp_fractions)):
        raise ValueError("low-BPP fraction is missing for a nonempty structure")

    hybrid_f1 = np.asarray([row["base_pair_f1"] for row in hybrid_rows])
    hybrid_nmsmd = np.asarray(
        [row["normalized_squared_mountain_distance"] for row in hybrid_rows]
    )
    gamma_f1 = np.asarray([gamma2_rows[row["id"]]["base_pair_f1"] for row in hybrid_rows])
    gamma_nmsmd = np.asarray(
        [
            gamma2_rows[row["id"]]["normalized_squared_mountain_distance"]
            for row in hybrid_rows
        ]
    )
    delta_f1 = hybrid_f1 - gamma_f1
    delta_nmsmd = hybrid_nmsmd - gamma_nmsmd
    equal = (delta_f1 == 0.0) & (delta_nmsmd == 0.0)
    hybrid_dominates = (
        (delta_f1 >= 0.0)
        & (delta_nmsmd <= 0.0)
        & ~equal
    )
    gamma2_dominates = (
        (delta_f1 <= 0.0)
        & (delta_nmsmd >= 0.0)
        & ~equal
    )
    mixed = ~(equal | hybrid_dominates | gamma2_dominates)
    if not np.all(equal | hybrid_dominates | gamma2_dominates | mixed):
        raise AssertionError("paired categories are incomplete")

    return {
        "alpha": alpha,
        "n": len(hybrid_rows),
        "median_fraction_selected_pair_bpp_below_0.01": float(
            np.median(low_bpp_fractions)
        ),
        "fraction_records_with_any_selected_pair_bpp_below_0.01": float(
            np.mean(low_bpp_fractions > 0.0)
        ),
        "fraction_f1_at_least_and_nmsmd_at_most_gamma2": float(
            np.mean((delta_f1 >= 0.0) & (delta_nmsmd <= 0.0))
        ),
        "fraction_hybrid_pareto_dominates_gamma2": float(np.mean(hybrid_dominates)),
        "fraction_gamma2_pareto_dominates_hybrid": float(np.mean(gamma2_dominates)),
        "fraction_equal_to_gamma2": float(np.mean(equal)),
        "fraction_mixed_vs_gamma2": float(np.mean(mixed)),
        "median_delta_base_pair_f1_vs_gamma2": float(np.median(delta_f1)),
        "median_delta_nmsmd_vs_gamma2": float(np.median(delta_nmsmd)),
    }


def build_summary(
    hybrid_rows: Sequence[dict[str, Any]],
    alphas: Sequence[float],
    gamma2_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate inputs and summarize every alpha value."""
    validate_record_match(hybrid_rows, gamma2_rows)
    return [
        summarize_alpha(
            alpha,
            [row for row in hybrid_rows if row["alpha"] == alpha],
            gamma2_rows,
        )
        for alpha in alphas
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-metrics", type=Path, required=True)
    parser.add_argument("--gamma-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hybrid_rows, alphas = load_rows(args.hybrid_metrics)
    gamma2_rows = load_gamma2_rows(args.gamma_metrics)
    summary = build_summary(hybrid_rows, alphas, gamma2_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "advisor_followup_summary.csv"
    write_csv(output_path, summary)
    with (args.output_dir / "advisor_followup_manifest.json").open("w") as handle:
        json.dump(
            {
                "record_count": len(gamma2_rows),
                "alpha_count": len(alphas),
                "analysis_script_sha256": file_sha256(Path(__file__)),
                "hybrid_metrics_sha256": file_sha256(args.hybrid_metrics),
                "gamma_metrics_sha256": file_sha256(args.gamma_metrics),
                "output_sha256": file_sha256(output_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"Summarized {len(gamma2_rows)} records at {len(alphas)} alpha values")


if __name__ == "__main__":
    main()
