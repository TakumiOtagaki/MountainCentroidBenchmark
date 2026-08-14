#!/usr/bin/env python3
"""Merge baseline rows with sequence-constrained wide metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASELINE_METHODS = {
    "vienna_mfe",
    "vienna_centroid",
    "mountain_centroid_relaxed",
}
CONSTRAINED_METHOD = "mountain_centroid_sequence_constrained"
FIELDNAMES = (
    "id",
    "family",
    "subfamily",
    "length",
    "method",
    "sequence",
    "reference_structure",
    "predicted_structure",
    "base_pair_f1",
    "squared_mountain_distance",
    "mean_squared_mountain_distance",
    "normalized_squared_mountain_distance",
    "prediction_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--sequence-constrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_case: dict[str, dict[str, dict[str, str]]] = {}
    with args.baselines.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in BASELINE_METHODS:
                rows_by_case.setdefault(row["id"], {})[row["method"]] = {
                    field: row[field] for field in FIELDNAMES
                }

    constrained_ids: set[str] = set()
    with args.sequence_constrained.open(newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["id"]
            if record_id in constrained_ids:
                raise ValueError(f"duplicate constrained ID: {record_id}")
            constrained_ids.add(record_id)
            methods = rows_by_case.get(record_id)
            if methods is None or set(methods) != BASELINE_METHODS:
                raise ValueError(f"missing baseline methods for {record_id}")
            relaxed = methods["mountain_centroid_relaxed"]
            if relaxed["predicted_structure"] != row["relaxed_structure"]:
                raise ValueError(f"relaxed structure mismatch for {record_id}")
            methods[CONSTRAINED_METHOD] = {
                "id": record_id,
                "family": row["family"],
                "subfamily": row["subfamily"],
                "length": row["length"],
                "method": CONSTRAINED_METHOD,
                "sequence": row["sequence"],
                "reference_structure": row["reference_structure"],
                "predicted_structure": row["constrained_structure"],
                "base_pair_f1": row["constrained_base_pair_f1"],
                "squared_mountain_distance": row[
                    "constrained_squared_mountain_distance"
                ],
                "mean_squared_mountain_distance": row[
                    "constrained_mean_squared_mountain_distance"
                ],
                "normalized_squared_mountain_distance": row[
                    "constrained_normalized_squared_mountain_distance"
                ],
                "prediction_seconds": row["constrained_seconds"],
            }

    baseline_ids = set(rows_by_case)
    if baseline_ids != constrained_ids:
        missing = sorted(baseline_ids - constrained_ids)[:5]
        extra = sorted(constrained_ids - baseline_ids)[:5]
        raise ValueError(
            f"ID sets differ: baseline={len(baseline_ids)}, "
            f"constrained={len(constrained_ids)}, missing={missing}, extra={extra}"
        )

    method_order = (*sorted(BASELINE_METHODS), CONSTRAINED_METHOD)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f".{args.output.name}.partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for record_id in sorted(rows_by_case):
            methods = rows_by_case[record_id]
            if set(methods) != {*BASELINE_METHODS, CONSTRAINED_METHOD}:
                raise ValueError(f"incomplete case: {record_id}")
            for method in method_order:
                writer.writerow(methods[method])
    partial.replace(args.output)
    print(f"Wrote {4 * len(rows_by_case)} rows for {len(rows_by_case)} cases")


if __name__ == "__main__":
    main()
