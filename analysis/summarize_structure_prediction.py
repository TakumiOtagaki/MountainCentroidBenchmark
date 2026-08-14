#!/usr/bin/env python3
"""Summarize overall, family-stratified, deduplicated, and outlier results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "vienna_mfe",
    "vienna_centroid",
    "mountain_centroid_relaxed",
    "mountain_centroid_sequence_constrained",
)
METRICS = (
    "base_pair_f1",
    "mean_squared_mountain_distance",
    "normalized_squared_mountain_distance",
)


def load_cases(path: Path) -> list[dict]:
    cases_by_id: dict[str, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case = cases_by_id.setdefault(
                row["id"],
                {
                    "id": row["id"],
                    "family": row["family"],
                    "subfamily": row["subfamily"],
                    "length": int(row["length"]),
                    "sequence": row["sequence"],
                    "reference_structure": row["reference_structure"],
                    "methods": {},
                },
            )
            case["methods"][row["method"]] = {
                "predicted_structure": row["predicted_structure"],
                "base_pair_f1": float(row["base_pair_f1"]),
                "squared_mountain_distance": float(
                    row["squared_mountain_distance"]
                ),
                "mean_squared_mountain_distance": float(
                    row["mean_squared_mountain_distance"]
                ),
                "normalized_squared_mountain_distance": float(
                    row["normalized_squared_mountain_distance"]
                ),
                "prediction_seconds": float(row["prediction_seconds"]),
            }
    cases = [
        case
        for case in cases_by_id.values()
        if all(method in case["methods"] for method in METHODS)
    ]
    cases.sort(key=lambda case: case["id"])
    return cases


def deduplicate(cases: list[dict]) -> list[dict]:
    cases_by_sequence = defaultdict(list)
    for case in cases:
        cases_by_sequence[case["sequence"]].append(case)

    retained = []
    for sequence_cases in cases_by_sequence.values():
        references = {
            case["reference_structure"] for case in sequence_cases
        }
        if len(references) != 1:
            continue
        retained.append(sequence_cases[0])
    retained.sort(key=lambda case: case["id"])
    return retained


def distribution_row(scope: str, group: str, cases: list[dict], method: str, metric: str):
    values = np.asarray(
        [case["methods"][method][metric] for case in cases],
        dtype=float,
    )
    return {
        "scope": scope,
        "group": group,
        "method": method,
        "metric": metric,
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def paired_row(scope: str, group: str, cases: list[dict], method: str, metric: str):
    differences = np.asarray(
        [
            case["methods"][method][metric]
            - case["methods"]["vienna_mfe"][metric]
            for case in cases
        ],
        dtype=float,
    )
    strictly_favorable = (
        differences > 0 if metric == "base_pair_f1" else differences < 0
    )
    tied = differences == 0
    favorable = strictly_favorable | tied
    return {
        "scope": scope,
        "group": group,
        "method": method,
        "metric": metric,
        "n": len(differences),
        "mean_difference_from_mfe": float(np.mean(differences)),
        "median_difference_from_mfe": float(np.median(differences)),
        "q1_difference": float(np.percentile(differences, 25)),
        "q3_difference": float(np.percentile(differences, 75)),
        "strictly_favorable_fraction": float(np.mean(strictly_favorable)),
        "tie_fraction": float(np.mean(tied)),
        "favorable_fraction": float(np.mean(favorable)),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/raw/four_method_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/structure_prediction"),
    )
    parser.add_argument("--top-outliers", type=int, default=100)
    args = parser.parse_args()

    all_cases = load_cases(args.input)
    scopes = {
        "all": all_cases,
        "deduplicated": deduplicate(all_cases),
    }

    distribution_rows = []
    paired_rows = []
    for scope, cases in scopes.items():
        grouped = defaultdict(list)
        for case in cases:
            grouped[case["family"]].append(case)
        groups = {"all_families": cases, **dict(sorted(grouped.items()))}
        for group, group_cases in groups.items():
            for method in METHODS:
                for metric in METRICS:
                    distribution_rows.append(
                        distribution_row(
                            scope,
                            group,
                            group_cases,
                            method,
                            metric,
                        )
                    )
            for method in METHODS[1:]:
                for metric in METRICS:
                    paired_rows.append(
                        paired_row(
                            scope,
                            group,
                            group_cases,
                            method,
                            metric,
                        )
                    )

    write_rows(args.output_dir / "summary_distribution.csv", distribution_rows)
    write_rows(args.output_dir / "summary_paired.csv", paired_rows)

    length_groups = defaultdict(list)
    for case in all_cases:
        lower = ((case["length"] - 1) // 50) * 50 + 1
        upper = min(lower + 49, 300)
        length_groups[f"{lower:03d}-{upper:03d}"].append(case)
    length_distribution_rows = []
    length_paired_rows = []
    for group, group_cases in sorted(length_groups.items()):
        for method in METHODS:
            for metric in METRICS:
                length_distribution_rows.append(
                    distribution_row("all", group, group_cases, method, metric)
                )
        for method in METHODS[1:]:
            for metric in METRICS:
                length_paired_rows.append(
                    paired_row("all", group, group_cases, method, metric)
                )
    write_rows(
        args.output_dir / "summary_length_distribution.csv",
        length_distribution_rows,
    )
    write_rows(
        args.output_dir / "summary_length_paired.csv",
        length_paired_rows,
    )

    unique_cases = scopes["deduplicated"]
    ranked_outliers = sorted(
        unique_cases,
        key=lambda case: (
            case["methods"]["mountain_centroid_sequence_constrained"][
                "mean_squared_mountain_distance"
            ]
            - case["methods"]["vienna_mfe"]["mean_squared_mountain_distance"]
        ),
        reverse=True,
    )[: args.top_outliers]
    outlier_rows = []
    for rank, case in enumerate(ranked_outliers, start=1):
        mountain = case["methods"]["mountain_centroid_sequence_constrained"]
        mfe = case["methods"]["vienna_mfe"]
        outlier_rows.append(
            {
                "rank": rank,
                "id": case["id"],
                "family": case["family"],
                "subfamily": case["subfamily"],
                "length": case["length"],
                "mountain_minus_mfe_msmd": (
                    mountain["mean_squared_mountain_distance"]
                    - mfe["mean_squared_mountain_distance"]
                ),
                "mountain_msmd": mountain["mean_squared_mountain_distance"],
                "mfe_msmd": mfe["mean_squared_mountain_distance"],
                "mountain_bp_f1": mountain["base_pair_f1"],
                "mfe_bp_f1": mfe["base_pair_f1"],
                "sequence": case["sequence"],
                "reference_structure": case["reference_structure"],
                "mountain_structure": mountain["predicted_structure"],
                "mfe_structure": mfe["predicted_structure"],
            }
        )
    write_rows(args.output_dir / "mountain_outliers.csv", outlier_rows)

    print(
        f"Summarized {len(all_cases)} complete cases and "
        f"{len(scopes['deduplicated'])} unique sequences"
    )


if __name__ == "__main__":
    main()
