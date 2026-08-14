#!/usr/bin/env python3
"""Summarize objective cost and resource use of sequence constraints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTIVE_GAP_REL_TOLERANCE = 1e-8


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)
    retained = []
    for sequence_rows in by_sequence.values():
        if len({row["reference_structure"] for row in sequence_rows}) == 1:
            retained.append(sequence_rows[0])
    return retained


def objective_gaps_with_roundoff_clamped(
    rows: list[dict[str, str]],
) -> np.ndarray:
    """Return objective gaps with tolerated negative roundoff set to zero."""
    values = np.asarray([float(row["objective_gap"]) for row in rows], dtype=float)
    constrained = np.asarray(
        [float(row["constrained_objective"]) for row in rows], dtype=float
    )
    tolerances = OBJECTIVE_GAP_REL_TOLERANCE * np.maximum(1.0, np.abs(constrained))
    violations = values < -tolerances
    if np.any(violations):
        worst = int(np.argmin(values))
        raise ValueError(
            "sequence-constrained objective violated the geometry-only lower "
            f"bound: gap={values[worst]}, tolerance={tolerances[worst]}"
        )
    return np.where(values < 0.0, 0.0, values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/structure_prediction/summary_constraint_cost.csv"),
    )
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("sequence-constrained metrics contain duplicate IDs")

    scopes = {"all": rows, "deduplicated": deduplicate(rows)}
    metrics = (
        "objective_gap",
        "absolute_reference_nmsmd_change",
        "base_pair_f1_change",
        "constrained_seconds",
        "states_evaluated",
        "partner_transitions_evaluated",
        "effective_depth_levels",
    )
    output_rows = []
    for scope, scope_rows in scopes.items():
        by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scope_rows:
            by_family[row["family"]].append(row)
        groups = {"all_families": scope_rows, **dict(sorted(by_family.items()))}
        for group, group_rows in groups.items():
            for metric in metrics:
                if metric == "objective_gap":
                    values = objective_gaps_with_roundoff_clamped(group_rows)
                elif metric == "absolute_reference_nmsmd_change":
                    values = np.asarray(
                        [
                            abs(
                                float(
                                    row[
                                        "constrained_normalized_squared_mountain_distance"
                                    ]
                                )
                                - float(
                                    row[
                                        "relaxed_normalized_squared_mountain_distance"
                                    ]
                                )
                            )
                            for row in group_rows
                        ],
                        dtype=float,
                    )
                elif metric == "base_pair_f1_change":
                    values = np.asarray(
                        [
                            float(row["constrained_base_pair_f1"])
                            - float(row["relaxed_base_pair_f1"])
                            for row in group_rows
                        ],
                        dtype=float,
                    )
                else:
                    values = np.asarray(
                        [float(row[metric]) for row in group_rows], dtype=float
                    )
                output_rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q1": float(np.percentile(values, 25)),
                        "q3": float(np.percentile(values, 75)),
                        "maximum": float(np.max(values)),
                        "positive_fraction": float(np.mean(values > 0)),
                        "zero_fraction": float(np.mean(values == 0)),
                        "negative_fraction": float(np.mean(values < 0)),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"Summarized {len(rows)} cases and {len(scopes['deduplicated'])} "
        f"unique consistent sequences"
    )


if __name__ == "__main__":
    main()
