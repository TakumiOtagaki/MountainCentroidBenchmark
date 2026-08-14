#!/usr/bin/env python3
"""Recalculate reference-based metrics while reusing cached predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from analysis.reference_metrics import (
        base_pair_f1,
        mean_squared_mountain_distance,
        normalized_squared_mountain_distance,
        squared_mountain_distance,
    )
except ModuleNotFoundError:
    from reference_metrics import (
        base_pair_f1,
        mean_squared_mountain_distance,
        normalized_squared_mountain_distance,
        squared_mountain_distance,
    )


def load_references(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {
            row["id"]: row["secondary_structure"]
            for row in csv.DictReader(handle)
        }


def update_metrics(row: dict[str, str], structure: str, prefix: str = "") -> None:
    metrics = {
        "base_pair_f1": base_pair_f1(structure, row["reference_structure"]),
        "squared_mountain_distance": squared_mountain_distance(
            structure, row["reference_structure"]
        ),
        "mean_squared_mountain_distance": mean_squared_mountain_distance(
            structure, row["reference_structure"]
        ),
        "normalized_squared_mountain_distance": normalized_squared_mountain_distance(
            structure, row["reference_structure"]
        ),
    }
    for name, value in metrics.items():
        field = f"{prefix}{name}"
        if field in row:
            row[field] = str(value)


def recalculate(dataset: Path, source: Path, output: Path) -> tuple[int, int]:
    references = load_references(dataset)
    changed_references: set[str] = set()
    row_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as input_handle, output.open(
        "w", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("metrics CSV has no header")
        writer = csv.DictWriter(output_handle, reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            row_count += 1
            reference = references.get(row["id"])
            if reference is None:
                raise ValueError(f"dataset has no reference for {row['id']}")
            if reference != row["reference_structure"]:
                row["reference_structure"] = reference
                changed_references.add(row["id"])
                if "predicted_structure" in row:
                    update_metrics(row, row["predicted_structure"])
                else:
                    update_metrics(row, row["relaxed_structure"], "relaxed_")
                    update_metrics(row, row["constrained_structure"], "constrained_")
            writer.writerow(row)
    return row_count, len(changed_references)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, references = recalculate(args.dataset, args.input, args.output)
    print(f"Wrote {rows} rows; updated {references} reference structures")


if __name__ == "__main__":
    main()
