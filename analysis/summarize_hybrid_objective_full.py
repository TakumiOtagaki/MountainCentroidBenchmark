#!/usr/bin/env python3
"""Summarize full hybrid results by dataset scope and RNA family."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.evaluate_gamma_centroid import file_sha256
from analysis.summarize_hybrid_objective_pilot import (
    METRICS,
    load_rows,
    write_csv,
)


def deduplicated_record_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    """Retain one ID per sequence when all reference annotations agree."""
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        value = (str(row["sequence"]), str(row["reference_structure"]))
        previous = records.setdefault(str(row["id"]), value)
        if previous != value:
            raise ValueError(f"sequence or reference differs across rows for {row['id']}")

    by_sequence: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record_id, (sequence, reference) in records.items():
        by_sequence[sequence].append((record_id, reference))

    retained = set()
    for sequence_records in by_sequence.values():
        if len({reference for _, reference in sequence_records}) == 1:
            retained.add(min(record_id for record_id, _ in sequence_records))
    return retained


def quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 3
    array = np.asarray(values, dtype=float)
    return (
        float(np.nanmedian(array)),
        float(np.nanpercentile(array, 25)),
        float(np.nanpercentile(array, 75)),
    )


def distribution_row(
    dataset_scope: str,
    group: str,
    alpha: float,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "dataset_scope": dataset_scope,
        "group": group,
        "alpha": alpha,
        "n": len(rows),
    }
    for metric in METRICS:
        median, q1, q3 = quantiles([float(row[metric]) for row in rows])
        output[f"median_{metric}"] = median
        output[f"q1_{metric}"] = q1
        output[f"q3_{metric}"] = q3
    low_bpp_fractions = np.asarray(
        [
            0.0
            if np.isnan(float(row["fraction_selected_pair_bpp_below_0.01"]))
            and float(row["predicted_pair_count"]) == 0.0
            else float(row["fraction_selected_pair_bpp_below_0.01"])
            for row in rows
        ],
        dtype=float,
    )
    output["fraction_records_with_any_selected_pair_bpp_below_0.01"] = float(
        np.mean(low_bpp_fractions > 0.0)
    )
    return output


def difference_statistics(values: Sequence[float], prefix: str) -> dict[str, float]:
    median, q1, q3 = quantiles(values)
    return {
        f"median_delta_{prefix}": median,
        f"q1_delta_{prefix}": q1,
        f"q3_delta_{prefix}": q3,
    }


def paired_row(
    dataset_scope: str,
    group: str,
    alpha: float,
    endpoint_alpha: float,
    rows: Sequence[dict[str, Any]],
    rows_by_id_alpha: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    endpoints = [rows_by_id_alpha[(str(row["id"]), endpoint_alpha)] for row in rows]
    f1 = np.asarray(
        [float(row["base_pair_f1"]) - float(endpoint["base_pair_f1"])
         for row, endpoint in zip(rows, endpoints)],
        dtype=float,
    )
    nmsmd = np.asarray(
        [float(row["normalized_squared_mountain_distance"])
         - float(endpoint["normalized_squared_mountain_distance"])
         for row, endpoint in zip(rows, endpoints)],
        dtype=float,
    )
    mountain_objective = [
        float(row["normalized_mountain_objective"])
        - float(endpoint["normalized_mountain_objective"])
        for row, endpoint in zip(rows, endpoints)
    ]
    centroid_gain = [
        float(row["normalized_centroid_gain"])
        - float(endpoint["normalized_centroid_gain"])
        for row, endpoint in zip(rows, endpoints)
    ]
    low_bpp_fraction = [
        float(row["fraction_selected_pair_bpp_below_0.01"])
        - float(endpoint["fraction_selected_pair_bpp_below_0.01"])
        for row, endpoint in zip(rows, endpoints)
        if not np.isnan(float(row["fraction_selected_pair_bpp_below_0.01"]))
        and not np.isnan(float(endpoint["fraction_selected_pair_bpp_below_0.01"]))
    ]
    return {
        "dataset_scope": dataset_scope,
        "group": group,
        "alpha": alpha,
        "endpoint_alpha": endpoint_alpha,
        "n": len(rows),
        **difference_statistics(f1, "base_pair_f1"),
        "fraction_higher_base_pair_f1": float(np.mean(f1 > 0.0)),
        "fraction_equal_base_pair_f1": float(np.mean(f1 == 0.0)),
        **difference_statistics(nmsmd, "nmsmd"),
        "fraction_lower_nmsmd": float(np.mean(nmsmd < 0.0)),
        "fraction_equal_nmsmd": float(np.mean(nmsmd == 0.0)),
        **difference_statistics(mountain_objective, "normalized_mountain_objective"),
        **difference_statistics(centroid_gain, "normalized_centroid_gain"),
        **difference_statistics(
            low_bpp_fraction,
            "fraction_selected_pair_bpp_below_0.01",
        ),
        "fraction_identical_to_endpoint": float(
            np.mean(
                [
                    row["predicted_structure"] == endpoint["predicted_structure"]
                    for row, endpoint in zip(rows, endpoints)
                ]
            )
        ),
    }


def build_summaries(
    rows: Sequence[dict[str, Any]],
    alphas: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    record_ids = {str(row["id"]) for row in rows}
    deduplicated_ids = deduplicated_record_ids(rows)
    scopes = {
        "all": record_ids,
        "deduplicated": deduplicated_ids,
    }
    rows_by_id_alpha = {
        (str(row["id"]), float(row["alpha"])): row
        for row in rows
    }
    distribution_rows = []
    paired_rows = []
    family_counts: dict[str, dict[str, int]] = {}
    for dataset_scope, retained_ids in scopes.items():
        retained_rows = [row for row in rows if str(row["id"]) in retained_ids]
        families = sorted({str(row["family"]) for row in retained_rows})
        groups = ["all_families", *families]
        family_counts[dataset_scope] = {
            group: len(
                {
                    str(row["id"])
                    for row in retained_rows
                    if group == "all_families" or row["family"] == group
                }
            )
            for group in groups
        }
        for group in groups:
            group_rows = [
                row
                for row in retained_rows
                if group == "all_families" or row["family"] == group
            ]
            for alpha in alphas:
                alpha_rows = [row for row in group_rows if row["alpha"] == alpha]
                distribution_rows.append(
                    distribution_row(dataset_scope, group, alpha, alpha_rows)
                )
                for endpoint_alpha in (0.0, 1.0):
                    paired_rows.append(
                        paired_row(
                            dataset_scope,
                            group,
                            alpha,
                            endpoint_alpha,
                            alpha_rows,
                            rows_by_id_alpha,
                        )
                    )
    metadata = {
        "prediction_count": len(rows),
        "alpha_count": len(alphas),
        "alphas": list(alphas),
        "record_counts": {scope: len(ids) for scope, ids in scopes.items()},
        "family_counts": family_counts,
        "deduplication": (
            "one lexicographically first ID per sequence when all reference "
            "structures for that sequence agree; conflicting references excluded"
        ),
    }
    return distribution_rows, paired_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, alphas = load_rows(args.metrics)
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("alpha endpoints 0 and 1 are required")
    distribution_rows, paired_rows, metadata = build_summaries(rows, alphas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = args.output_dir / "distribution_summary.csv"
    paired_path = args.output_dir / "paired_summary.csv"
    write_csv(distribution_path, distribution_rows)
    write_csv(paired_path, paired_rows)
    with (args.output_dir / "analysis_manifest.json").open("w") as handle:
        json.dump(
            {
                **metadata,
                "analysis_script_sha256": file_sha256(Path(__file__)),
                "input_metrics": str(args.metrics.resolve()),
                "input_metrics_sha256": file_sha256(args.metrics),
                "output_sha256": {
                    distribution_path.name: file_sha256(distribution_path),
                    paired_path.name: file_sha256(paired_path),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        f"Summarized {metadata['record_counts']['all']} full and "
        f"{metadata['record_counts']['deduplicated']} deduplicated records"
    )


if __name__ == "__main__":
    main()
