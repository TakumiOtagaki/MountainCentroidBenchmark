#!/usr/bin/env python3
"""Resumable gamma-centroid sensitivity evaluation."""

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
from typing import Any, Sequence

import numpy
import RNA

try:
    from analysis.reference_metrics import (
        base_pair_f1,
        normalized_squared_mountain_distance,
    )
except ModuleNotFoundError:
    from reference_metrics import (
        base_pair_f1,
        normalized_squared_mountain_distance,
    )
from mountain_centroid.formatting import pairs_from_bracket
from mountain_centroid.sequence import MIN_HAIRPIN_LENGTH, can_pair


DEFAULT_GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FIELDNAMES = (
    "id",
    "family",
    "subfamily",
    "length",
    "gamma",
    "sequence",
    "reference_structure",
    "predicted_structure",
    "base_pair_f1",
    "normalized_squared_mountain_distance",
    "predicted_pair_count",
    "bpp_seconds",
    "gamma_centroid_seconds",
    "prediction_seconds",
    "config_signature",
)


def parse_gammas(value: str) -> tuple[float, ...]:
    gammas = tuple(float(item) for item in value.split(","))
    if not gammas or any(gamma <= 0.0 for gamma in gammas):
        raise argparse.ArgumentTypeError("gammas must be positive")
    if len(set(gammas)) != len(gammas):
        raise argparse.ArgumentTypeError("gammas must be distinct")
    return gammas


def gamma_centroid_structure(
    bpp: Sequence[Sequence[float]],
    gamma: float,
    *,
    turn: int = MIN_HAIRPIN_LENGTH,
) -> str:
    """Return the pseudoknot-free gamma-centroid for a zero-based BPP matrix."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if turn < 0:
        raise ValueError("turn must be nonnegative")
    length = len(bpp)
    if any(len(row) != length for row in bpp):
        raise ValueError("bpp must be a square matrix")

    scores = [[0.0] * length for _ in range(length)]
    partners = [[-1] * length for _ in range(length)]
    for span in range(1, length + 1):
        for left in range(length - span + 1):
            right = left + span - 1
            best = scores[left + 1][right] if left < right else 0.0
            best_partner = -1
            for partner in range(left + turn + 1, right + 1):
                pair_gain = (
                    (gamma + 1.0) * float(bpp[left][partner]) - 1.0
                )
                if pair_gain <= 0.0:
                    continue
                value = pair_gain
                if left + 1 <= partner - 1:
                    value += scores[left + 1][partner - 1]
                if partner + 1 <= right:
                    value += scores[partner + 1][right]
                if value > best + 1e-12:
                    best = value
                    best_partner = partner
            scores[left][right] = best
            partners[left][right] = best_partner

    structure = ["."] * length
    intervals = [(0, length - 1)] if length else []
    while intervals:
        left, right = intervals.pop()
        if left > right:
            continue
        partner = partners[left][right]
        if partner < 0:
            intervals.append((left + 1, right))
            continue
        structure[left] = "("
        structure[partner] = ")"
        intervals.append((left + 1, partner - 1))
        intervals.append((partner + 1, right))
    return "".join(structure)


def validate_prediction(sequence: str, structure: str) -> int:
    """Validate the benchmark structural constraints and return pair count."""
    if len(sequence) != len(structure):
        raise ValueError("sequence and structure lengths differ")
    pairs = pairs_from_bracket(structure)
    for left, right in pairs:
        if right - left - 1 < MIN_HAIRPIN_LENGTH:
            raise ValueError(f"pair ({left}, {right}) violates TURN")
        if not can_pair(sequence[left - 1], sequence[right - 1]):
            raise ValueError(f"pair ({left}, {right}) is not sequence-pairable")
    return len(pairs)


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


def load_centroid_structures(path: Path) -> dict[str, str]:
    structures = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "vienna_centroid":
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
    model = RNA.md()
    model.temperature = float(task["temperature"])
    bpp_started = time.perf_counter()
    fold_compound = RNA.fold_compound(sequence, model)
    _, mfe_energy = fold_compound.mfe()
    fold_compound.exp_params_rescale(mfe_energy)
    fold_compound.pf()
    vienna_bpp = fold_compound.bpp()
    length = len(sequence)
    bpp = [row[1 : length + 1] for row in vienna_bpp[1 : length + 1]]
    bpp_seconds = time.perf_counter() - bpp_started

    predictions = []
    for gamma in task["gammas"]:
        gamma_started = time.perf_counter()
        structure = gamma_centroid_structure(bpp, gamma)
        gamma_seconds = time.perf_counter() - gamma_started
        pair_count = validate_prediction(sequence, structure)
        expected_centroid = task.get("expected_centroid")
        if gamma == 1.0 and expected_centroid is not None:
            if structure != expected_centroid:
                raise RuntimeError(
                    f"gamma=1 differs from ViennaRNA centroid for "
                    f"{record['id']}"
                )
        predictions.append(
            {
                "gamma": gamma,
                "predicted_structure": structure,
                "base_pair_f1": base_pair_f1(structure, reference),
                "normalized_squared_mountain_distance": (
                    normalized_squared_mountain_distance(structure, reference)
                ),
                "predicted_pair_count": pair_count,
                "bpp_seconds": bpp_seconds,
                "gamma_centroid_seconds": gamma_seconds,
                "prediction_seconds": bpp_seconds + gamma_seconds,
            }
        )
    atomic_write_json(
        output_path,
        {
            **record,
            "length": length,
            "predictions": predictions,
            "config_signature": signature,
        },
    )
    return record["id"], False


def write_merged_csv(output_dir: Path, records: list[dict[str, str]]) -> Path:
    output = output_dir / "metrics.csv"
    partial = output.with_name(f".{output.name}.partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            path = case_path(output_dir, record["id"])
            with path.open() as case_handle:
                case = json.load(case_handle)
            for prediction in case["predictions"]:
                writer.writerow(
                    {
                        "id": case["id"],
                        "family": case["family"],
                        "subfamily": case["subfamily"],
                        "length": case["length"],
                        "sequence": case["sequence"],
                        "reference_structure": case["reference_structure"],
                        "config_signature": case["config_signature"],
                        **prediction,
                    }
                )
    partial.replace(output)
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        required=True,
        help="Verified four-method benchmark metrics used to validate gamma=1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--gammas",
        type=parse_gammas,
        default=DEFAULT_GAMMAS,
    )
    parser.add_argument("--temperature", type=float, default=37.0)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")
    setup_logging(args.log)
    records = load_dataset(args.dataset, args.max_length)
    if args.limit is not None and args.limit < len(records):
        records = random.Random(args.seed).sample(records, args.limit)
        records.sort(key=lambda record: record["id"])
    centroids = load_centroid_structures(args.baseline_metrics)
    missing = [record["id"] for record in records if record["id"] not in centroids]
    if missing:
        raise ValueError(f"baseline centroid missing for {len(missing)} records")

    repository = Path(__file__).resolve().parents[1]
    software = repository / "software" / "MountainCentroid"
    dataset_sha256 = file_sha256(args.dataset)
    baseline_metrics_sha256 = file_sha256(args.baseline_metrics)
    config = {
        "analysis_revision": git_revision(repository),
        "analysis_worktree_clean": git_worktree_is_clean(repository),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "software_revision": git_revision(software),
        "software_worktree_clean": git_worktree_is_clean(software),
        "dataset_sha256": dataset_sha256,
        "baseline_metrics_sha256": baseline_metrics_sha256,
        "gammas": args.gammas,
        "temperature_celsius": args.temperature,
        "max_length": args.max_length,
    }
    signature = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        **config,
        "config_signature": signature,
        "dataset": str(args.dataset.resolve()),
        "baseline_metrics": str(args.baseline_metrics.resolve()),
        "selected_cases": len(records),
        "workers": args.workers,
        "seed": args.seed,
        "limit": args.limit,
        "python": sys.version,
        "numpy": numpy.__version__,
        "viennarna": RNA.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
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
        logging.info(
            "Starting length bin %d-%d (%d cases)",
            lower,
            min(lower + 49, args.max_length),
            len(bin_records),
        )
        tasks = [
            {
                "record": record,
                "output_path": str(case_path(args.output_dir, record["id"])),
                "config_signature": signature,
                "temperature": args.temperature,
                "gammas": args.gammas,
                "expected_centroid": centroids[record["id"]],
            }
            for record in bin_records
        ]
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
