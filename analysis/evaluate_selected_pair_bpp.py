#!/usr/bin/env python3
"""Measure BPP support for base pairs present in benchmark structures."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import RNA

try:
    from analysis.evaluate_gamma_centroid import (
        atomic_write_json,
        file_sha256,
        git_revision,
        git_worktree_is_clean,
    )
except ModuleNotFoundError:
    from evaluate_gamma_centroid import (
        atomic_write_json,
        file_sha256,
        git_revision,
        git_worktree_is_clean,
    )
try:
    from analysis.reference_metrics import pairs_from_extended_dot_bracket
except ModuleNotFoundError:
    from reference_metrics import pairs_from_extended_dot_bracket


METHODS = (
    "reference",
    "vienna_mfe",
    "vienna_centroid",
    "gamma_centroid_2",
    "mountain_centroid_sequence_constrained",
)
THRESHOLDS = (0.01, 0.05, 0.1)


def load_predictions(
    baseline_path: Path,
    gamma_path: Path,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    baseline_methods = set(METHODS) - {"reference", "gamma_centroid_2"}
    with baseline_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in baseline_methods:
                continue
            record = records.setdefault(
                row["id"],
                {
                    "id": row["id"],
                    "family": row.get("family", ""),
                    "subfamily": row.get("subfamily", ""),
                    "sequence": row["sequence"],
                    "structures": {"reference": row["reference_structure"]},
                },
            )
            if record["sequence"] != row["sequence"]:
                raise ValueError(f"sequence differs across rows for {row['id']}")
            record["structures"][method] = row["predicted_structure"]

    with gamma_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not math.isclose(float(row["gamma"]), 2.0):
                continue
            if row["id"] not in records:
                raise ValueError(f"gamma result has unknown id {row['id']}")
            record = records[row["id"]]
            if record["sequence"] != row["sequence"]:
                raise ValueError(f"sequence differs in gamma result for {row['id']}")
            record["structures"]["gamma_centroid_2"] = row[
                "predicted_structure"
            ]

    for record in records.values():
        missing = set(METHODS) - set(record["structures"])
        if missing:
            raise ValueError(f"{record['id']} is missing methods {sorted(missing)}")
    return sorted(records.values(), key=lambda record: record["id"])


def selected_pair_probabilities(
    bpp: Sequence[Sequence[float]],
    structure: str,
) -> list[float]:
    values = []
    for left, right in pairs_from_extended_dot_bracket(structure):
        probability = float(bpp[left][right])
        if probability < -1e-12 or probability > 1.0 + 1e-12:
            raise ValueError(f"invalid BPP for pair ({left}, {right}): {probability}")
        values.append(min(1.0, max(0.0, probability)))
    return values


def case_path(output_dir: Path, record_id: str) -> Path:
    digest = hashlib.sha256(record_id.encode()).hexdigest()[:20]
    return output_dir / "cases" / f"{digest}.json"


def evaluate_record(task: dict[str, Any]) -> tuple[str, bool]:
    record = task["record"]
    output_path = Path(task["output_path"])
    signature = task["config_signature"]
    if output_path.is_file():
        with output_path.open() as handle:
            existing = json.load(handle)
        if existing.get("id") == record["id"] and existing.get(
            "config_signature"
        ) == signature:
            return record["id"], True

    model = RNA.md()
    model.temperature = float(task["temperature"])
    fold_compound = RNA.fold_compound(record["sequence"], model)
    _, mfe_energy = fold_compound.mfe()
    fold_compound.exp_params_rescale(mfe_energy)
    fold_compound.pf()
    bpp = fold_compound.bpp()
    selected = {
        method: selected_pair_probabilities(bpp, record["structures"][method])
        for method in METHODS
    }
    atomic_write_json(
        output_path,
        {
            "id": record["id"],
            "family": record["family"],
            "subfamily": record["subfamily"],
            "length": len(record["sequence"]),
            "selected_pair_bpps": selected,
            "config_signature": signature,
        },
    )
    return record["id"], False


def _fraction_below(values: Sequence[float], threshold: float) -> float:
    if not values:
        return math.nan
    return sum(value < threshold for value in values) / len(values)


def write_summaries(output_dir: Path, records: Sequence[dict[str, Any]]) -> None:
    pooled = {method: [] for method in METHODS}
    sequence_rows = []
    for record in records:
        with case_path(output_dir, record["id"]).open() as handle:
            case = json.load(handle)
        for method in METHODS:
            values = case["selected_pair_bpps"][method]
            pooled[method].extend(values)
            row = {
                "id": case["id"],
                "family": case["family"],
                "subfamily": case["subfamily"],
                "length": case["length"],
                "method": method,
                "pair_count": len(values),
                "median_bpp": np.median(values) if values else "",
            }
            for threshold in THRESHOLDS:
                row[f"fraction_below_{threshold:g}"] = (
                    _fraction_below(values, threshold) if values else ""
                )
            sequence_rows.append(row)

    sequence_fields = tuple(sequence_rows[0])
    with (output_dir / "sequence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sequence_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sequence_rows)

    pooled_fields = (
        "method",
        "sequence_count",
        "sequences_with_pairs",
        "selected_pair_count",
        "median_bpp",
        "q1_bpp",
        "q3_bpp",
        "median_sequence_median_bpp",
        *(f"fraction_below_{threshold:g}" for threshold in THRESHOLDS),
        *(f"median_sequence_fraction_below_{threshold:g}" for threshold in THRESHOLDS),
    )
    with (output_dir / "pooled_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pooled_fields, lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            values = np.asarray(pooled[method], dtype=float)
            method_rows = [row for row in sequence_rows if row["method"] == method]
            paired_rows = [row for row in method_rows if row["pair_count"]]
            row: dict[str, Any] = {
                "method": method,
                "sequence_count": len(method_rows),
                "sequences_with_pairs": len(paired_rows),
                "selected_pair_count": len(values),
                "median_bpp": np.median(values),
                "q1_bpp": np.percentile(values, 25),
                "q3_bpp": np.percentile(values, 75),
                "median_sequence_median_bpp": np.median(
                    [float(item["median_bpp"]) for item in paired_rows]
                ),
            }
            for threshold in THRESHOLDS:
                key = f"fraction_below_{threshold:g}"
                row[key] = float(np.mean(values < threshold))
                row[f"median_sequence_{key}"] = np.median(
                    [float(item[key]) for item in paired_rows]
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        required=True,
        help="Verified four-method benchmark metrics",
    )
    parser.add_argument(
        "--gamma-metrics",
        type=Path,
        default=Path("results/raw/gamma_centroid/metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--temperature", type=float, default=37.0)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    records = load_predictions(args.baseline_metrics, args.gamma_metrics)
    if args.limit is not None and args.limit < len(records):
        records = random.Random(args.seed).sample(records, args.limit)
        records.sort(key=lambda record: record["id"])

    repository = Path(__file__).resolve().parents[1]
    software = repository / "software" / "MountainCentroid"
    config = {
        "analysis_revision": git_revision(repository),
        "analysis_worktree_clean": git_worktree_is_clean(repository),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "software_revision": git_revision(software),
        "software_worktree_clean": git_worktree_is_clean(software),
        "baseline_metrics_sha256": file_sha256(args.baseline_metrics),
        "gamma_metrics_sha256": file_sha256(args.gamma_metrics),
        "temperature_celsius": args.temperature,
        "methods": METHODS,
        "thresholds": THRESHOLDS,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = {
        **config,
        "config_signature": signature,
        "baseline_metrics": str(args.baseline_metrics.resolve()),
        "gamma_metrics": str(args.gamma_metrics.resolve()),
        "selected_cases": len(records),
        "workers": args.workers,
        "seed": args.seed,
        "limit": args.limit,
        "python": sys.version,
        "numpy": np.__version__,
        "viennarna": RNA.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    tasks = [
        {
            "record": record,
            "output_path": str(case_path(args.output_dir, record["id"])),
            "config_signature": signature,
            "temperature": args.temperature,
        }
        for record in records
    ]
    completed = resumed = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(evaluate_record, task) for task in tasks]
        for future in as_completed(futures):
            _, was_resumed = future.result()
            completed += 1
            resumed += was_resumed
            if completed % 250 == 0 or completed == len(tasks):
                logging.warning(
                    "Completed %d/%d (%d resumed)", completed, len(tasks), resumed
                )
    write_summaries(args.output_dir, records)


if __name__ == "__main__":
    main()
