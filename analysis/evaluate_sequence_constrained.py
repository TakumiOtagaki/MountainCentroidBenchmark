#!/usr/bin/env python3
"""Resumable evaluation of sequence-constrained Mountain Centroid."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy
import RNA

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
from mountain_centroid.bpp_mu import compute_bpp_and_mu
from mountain_centroid.cpp_constrained import (
    cpp_sequence_constrained_mountain_centroid,
    default_cpp_constrained_path,
)
from mountain_centroid.relaxed import relaxed_mountain_centroid


FIELDNAMES = (
    "id",
    "family",
    "subfamily",
    "length",
    "sequence",
    "reference_structure",
    "relaxed_structure",
    "constrained_structure",
    "relaxed_objective",
    "constrained_objective",
    "objective_gap",
    "relaxed_base_pair_f1",
    "constrained_base_pair_f1",
    "relaxed_squared_mountain_distance",
    "constrained_squared_mountain_distance",
    "relaxed_mean_squared_mountain_distance",
    "constrained_mean_squared_mountain_distance",
    "relaxed_normalized_squared_mountain_distance",
    "constrained_normalized_squared_mountain_distance",
    "bpp_seconds",
    "relaxed_seconds",
    "constrained_seconds",
    "states_evaluated",
    "partner_transitions_evaluated",
    "effective_depth_levels",
    "bpp_backend",
    "bpp_beam_size",
    "bpp_cutoff",
    "config_signature",
)


def load_dataset(path: Path, max_length: int) -> list[dict[str, str]]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row["sequence"].upper().replace("T", "U")
            reference = row["secondary_structure"]
            if not sequence or set(sequence) - set("ACGU"):
                continue
            if len(sequence) != len(reference) or len(sequence) > max_length:
                continue
            if set(reference) - set(".()[]"):
                continue
            records.append(
                {
                    "id": row["id"],
                    "family": row.get("family", ""),
                    "subfamily": row.get("subfamily", ""),
                    "sequence": sequence,
                    "reference_structure": reference,
                }
            )
    records.sort(key=lambda record: record["id"])
    return records


def load_relaxed_structures(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    structures = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "mountain_centroid_relaxed":
                structures[row["id"]] = row["predicted_structure"]
    return structures


def case_path(output_dir: Path, record_id: str) -> Path:
    digest = hashlib.sha256(record_id.encode()).hexdigest()[:20]
    return output_dir / "cases" / f"{digest}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    partial.replace(path)


def metric_values(structure: str, reference: str) -> dict[str, float]:
    return {
        "base_pair_f1": base_pair_f1(structure, reference),
        "squared_mountain_distance": squared_mountain_distance(
            structure,
            reference,
        ),
        "mean_squared_mountain_distance": mean_squared_mountain_distance(
            structure,
            reference,
        ),
        "normalized_squared_mountain_distance": (
            normalized_squared_mountain_distance(structure, reference)
        ),
    }


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

    sequence = record["sequence"]
    reference = record["reference_structure"]
    bpp_started = time.perf_counter()
    _, expected_heights = compute_bpp_and_mu(
        sequence,
        temperature=task["temperature"],
        backend=task["bpp_backend"],
        beam_size=task["bpp_beam_size"],
        cutoff=task["bpp_cutoff"],
        linearpartition_path=task["linearpartition_path"],
    )
    bpp_seconds = time.perf_counter() - bpp_started

    relaxed_started = time.perf_counter()
    relaxed = relaxed_mountain_centroid(expected_heights)
    relaxed_seconds = time.perf_counter() - relaxed_started
    expected_relaxed = task.get("expected_relaxed_structure")
    if expected_relaxed is not None and relaxed.structure != expected_relaxed:
        raise RuntimeError(
            f"Relaxed structure changed for {record['id']}: "
            f"{relaxed.structure} != {expected_relaxed}"
        )

    constrained_started = time.perf_counter()
    constrained = cpp_sequence_constrained_mountain_centroid(
        sequence,
        expected_heights,
        executable=task["constrained_binary_path"],
    )
    constrained_seconds = time.perf_counter() - constrained_started
    objective_gap = constrained.squared_error - relaxed.squared_error
    tolerance = 1e-8 * max(1.0, abs(constrained.squared_error))
    if objective_gap < -tolerance:
        raise RuntimeError(
            f"Constrained objective violated relaxed lower bound for "
            f"{record['id']}: "
            f"gap={objective_gap}"
        )

    relaxed_metrics = metric_values(relaxed.structure, reference)
    constrained_metrics = metric_values(constrained.structure, reference)
    diagnostics = constrained.diagnostics
    row: dict[str, Any] = {
        **record,
        "length": len(sequence),
        "relaxed_structure": relaxed.structure,
        "constrained_structure": constrained.structure,
        "relaxed_objective": relaxed.squared_error,
        "constrained_objective": constrained.squared_error,
        "objective_gap": objective_gap,
        "bpp_seconds": bpp_seconds,
        "relaxed_seconds": relaxed_seconds,
        "constrained_seconds": constrained_seconds,
        "states_evaluated": diagnostics.states_evaluated,
        "partner_transitions_evaluated": (
            diagnostics.partner_transitions_evaluated
        ),
        "effective_depth_levels": diagnostics.effective_depth_levels,
        "bpp_backend": task["bpp_backend"],
        "bpp_beam_size": (
            task["bpp_beam_size"]
            if task["bpp_backend"] == "linearpartition"
            else None
        ),
        "bpp_cutoff": (
            task["bpp_cutoff"]
            if task["bpp_backend"] == "linearpartition"
            else None
        ),
        "config_signature": signature,
    }
    for prefix, metrics in (
        ("relaxed", relaxed_metrics),
        ("constrained", constrained_metrics),
    ):
        for metric, value in metrics.items():
            row[f"{prefix}_{metric}"] = value
    atomic_write_json(output_path, row)
    return record["id"], False


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def git_worktree_is_clean(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return not completed.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_merged_csv(output_dir: Path, records: list[dict[str, str]]) -> Path:
    rows = []
    for record in records:
        path = case_path(output_dir, record["id"])
        with path.open() as handle:
            row = json.load(handle)
        rows.append(row)
    rows.sort(key=lambda row: row["id"])
    output = output_dir / "metrics.csv"
    partial = output.with_name(f".{output.name}.partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(output)
    return output


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rnastralign.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--bpp-backend",
        choices=("vienna", "linearpartition"),
        default="vienna",
    )
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--bpp-beam-size", type=int, default=100)
    parser.add_argument("--bpp-cutoff", type=float, default=0.0)
    parser.add_argument("--linearpartition-path", type=Path)
    parser.add_argument(
        "--constrained-binary-path",
        type=Path,
        default=default_cpp_constrained_path(),
    )
    parser.add_argument("--baseline-metrics", type=Path)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")
    setup_logging(args.log)
    records = load_dataset(args.dataset, args.max_length)
    if args.limit is not None and args.limit < len(records):
        records = random.Random(args.seed).sample(records, args.limit)
        records.sort(key=lambda record: record["id"])
    relaxed_structures = load_relaxed_structures(args.baseline_metrics)

    repository = Path(__file__).resolve().parents[1]
    software = repository / "software" / "MountainCentroid"
    constrained_binary_path = args.constrained_binary_path.resolve()
    linearpartition_path: Path | None = None
    linearpartition_revision: str | None = None
    if args.bpp_backend == "linearpartition":
        linearpartition_path = (
            args.linearpartition_path.resolve()
            if args.linearpartition_path is not None
            else (
                software / "vendor" / "LinearPartition" / "linearpartition"
            ).resolve()
        )
        linearpartition_revision = git_revision(
            software / "vendor" / "LinearPartition"
        )
    config = {
        "analysis_revision": git_revision(repository),
        "analysis_worktree_clean": git_worktree_is_clean(repository),
        "software_revision": git_revision(software),
        "software_worktree_clean": git_worktree_is_clean(software),
        "constrained_binary_path": str(constrained_binary_path),
        "constrained_binary_sha256": file_sha256(constrained_binary_path),
        "linearpartition_path": (
            str(linearpartition_path) if linearpartition_path is not None else None
        ),
        "linearpartition_revision": linearpartition_revision,
        "bpp_backend": args.bpp_backend,
        "temperature_celsius": args.temperature,
        "bpp_beam_size": (
            args.bpp_beam_size if args.bpp_backend == "linearpartition" else None
        ),
        "bpp_cutoff": (
            args.bpp_cutoff if args.bpp_backend == "linearpartition" else None
        ),
        "max_length": args.max_length,
    }
    signature = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        **config,
        "config_signature": signature,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "selected_cases": len(records),
        "workers": args.workers,
        "seed": args.seed,
        "limit": args.limit,
        "python": sys.version,
        "numpy": numpy.__version__,
        "viennarna": RNA.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    logging.info("Evaluating %d sequences with %d workers", len(records), args.workers)
    completed_count = 0
    resumed_count = 0
    bins: dict[int, list[dict[str, str]]] = {}
    for record in records:
        lower = ((len(record["sequence"]) - 1) // 50) * 50 + 1
        bins.setdefault(lower, []).append(record)

    for lower in sorted(bins, reverse=True):
        bin_records = bins[lower]
        logging.info("Starting length bin %d-%d (%d cases)", lower, min(lower + 49, 300), len(bin_records))
        tasks = []
        for record in bin_records:
            tasks.append(
                {
                    "record": record,
                    "output_path": str(case_path(args.output_dir, record["id"])),
                    "config_signature": signature,
                    "bpp_backend": args.bpp_backend,
                    "temperature": args.temperature,
                    "bpp_beam_size": args.bpp_beam_size,
                    "bpp_cutoff": args.bpp_cutoff,
                    "linearpartition_path": (
                        str(linearpartition_path)
                        if linearpartition_path is not None
                        else None
                    ),
                    "constrained_binary_path": str(constrained_binary_path),
                    "expected_relaxed_structure": relaxed_structures.get(record["id"]),
                }
            )
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks))
        ) as executor:
            futures = [executor.submit(evaluate_record, task) for task in tasks]
            for future in as_completed(futures):
                _, resumed = future.result()
                completed_count += 1
                resumed_count += resumed
                if completed_count % 250 == 0 or completed_count == len(records):
                    logging.info(
                        "Completed %d/%d (%d resumed)",
                        completed_count,
                        len(records),
                        resumed_count,
                    )

    output = write_merged_csv(args.output_dir, records)
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
