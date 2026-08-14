#!/usr/bin/env python3
"""Evaluate ViennaRNA baselines and relaxed Mountain Centroid."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

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
from mountain_centroid import relaxed_mountain_centroid
from mountain_centroid.bpp_mu import (
    bpp_and_mu_from_vienna_fold_compound,
    compute_bpp_and_mu,
)


METHODS = (
    "vienna_mfe",
    "vienna_centroid",
    "mountain_centroid_relaxed",
)
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


def load_dataset(
    path: Path,
    *,
    max_length: int,
    deduplicate_sequences: bool,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_sequences: set[str] = set()
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
            if deduplicate_sequences and sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            records.append(
                {
                    "id": row["id"],
                    "family": row.get("family", ""),
                    "subfamily": row.get("subfamily", ""),
                    "sequence": sequence,
                    "reference_structure": reference,
                }
            )
    return records


def _vienna_predictions(
    sequence: str,
    temperature: float,
) -> tuple[
    tuple[str, float],
    tuple[str, float],
    list[float],
    float,
]:
    md = RNA.md()
    md.temperature = float(temperature)

    mfe_started = time.perf_counter()
    fold_compound = RNA.fold_compound(sequence, md)
    mfe_structure, mfe_energy = fold_compound.mfe()
    mfe_seconds = time.perf_counter() - mfe_started

    ensemble_started = time.perf_counter()
    fold_compound.exp_params_rescale(mfe_energy)
    fold_compound.pf()
    ensemble_seconds = time.perf_counter() - ensemble_started

    centroid_started = time.perf_counter()
    centroid_structure, _ = fold_compound.centroid()
    centroid_seconds = (
        mfe_seconds
        + ensemble_seconds
        + time.perf_counter()
        - centroid_started
    )

    profile_started = time.perf_counter()
    _, expected_heights = bpp_and_mu_from_vienna_fold_compound(fold_compound)
    mountain_input_seconds = (
        mfe_seconds
        + ensemble_seconds
        + time.perf_counter()
        - profile_started
    )
    return (
        (mfe_structure, mfe_seconds),
        (centroid_structure, centroid_seconds),
        expected_heights,
        mountain_input_seconds,
    )


def _metric_row(
    record: dict[str, str],
    method: str,
    predicted_structure: str,
    prediction_seconds: float,
) -> dict[str, str | int | float]:
    reference = record["reference_structure"]
    return {
        "id": record["id"],
        "family": record["family"],
        "subfamily": record["subfamily"],
        "length": len(record["sequence"]),
        "method": method,
        "sequence": record["sequence"],
        "reference_structure": reference,
        "predicted_structure": predicted_structure,
        "base_pair_f1": base_pair_f1(predicted_structure, reference),
        "squared_mountain_distance": squared_mountain_distance(
            predicted_structure,
            reference,
        ),
        "mean_squared_mountain_distance": mean_squared_mountain_distance(
            predicted_structure,
            reference,
        ),
        "normalized_squared_mountain_distance": normalized_squared_mountain_distance(
            predicted_structure,
            reference,
        ),
        "prediction_seconds": prediction_seconds,
    }


def evaluate_record(
    record: dict[str, str],
    *,
    bpp_backend: str,
    temperature: float,
    bpp_beam_size: int,
    bpp_cutoff: float,
    linearpartition_path: str | None,
) -> list[dict[str, str | int | float]]:
    sequence = record["sequence"]
    (
        (mfe_structure, mfe_seconds),
        (centroid_structure, centroid_seconds),
        vienna_expected_heights,
        vienna_mountain_input_seconds,
    ) = _vienna_predictions(sequence, temperature)

    if bpp_backend == "vienna":
        expected_heights = vienna_expected_heights
        bpp_seconds = vienna_mountain_input_seconds
    else:
        bpp_started = time.perf_counter()
        _, expected_heights = compute_bpp_and_mu(
            sequence,
            temperature=temperature,
            backend=bpp_backend,
            beam_size=bpp_beam_size,
            cutoff=bpp_cutoff,
            linearpartition_path=linearpartition_path,
        )
        bpp_seconds = time.perf_counter() - bpp_started

    relaxed_started = time.perf_counter()
    relaxed_prediction = relaxed_mountain_centroid(expected_heights)
    relaxed_seconds = (
        bpp_seconds + time.perf_counter() - relaxed_started
    )

    return [
        _metric_row(record, "vienna_mfe", mfe_structure, mfe_seconds),
        _metric_row(
            record,
            "vienna_centroid",
            centroid_structure,
            centroid_seconds,
        ),
        _metric_row(
            record,
            "mountain_centroid_relaxed",
            relaxed_prediction.structure,
            relaxed_seconds,
        ),
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rnastralign.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/baselines.csv"),
    )
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--deduplicate-sequences", action="store_true")
    parser.add_argument(
        "--bpp-backend",
        choices=("vienna", "linearpartition"),
        default="vienna",
    )
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--bpp-beam-size", type=int, default=100)
    parser.add_argument("--bpp-cutoff", type=float, default=0.0)
    parser.add_argument("--linearpartition-path", type=Path)
    args = parser.parse_args()

    records = load_dataset(
        args.dataset,
        max_length=args.max_length,
        deduplicate_sequences=args.deduplicate_sequences,
    )
    if args.limit is not None and args.limit < len(records):
        records = random.Random(args.seed).sample(records, args.limit)
    records.sort(key=lambda record: record["id"])
    print(
        f"Evaluating {len(records)} sequences with {args.bpp_backend} BPPs",
        file=sys.stderr,
    )

    repository = Path(__file__).resolve().parents[1]
    software = repository / "software" / "MountainCentroid"
    linearpartition_path: str | None = None
    linearpartition_revision: str | None = None
    if args.bpp_backend == "linearpartition":
        resolved_linearpartition_path = (
            args.linearpartition_path.resolve()
            if args.linearpartition_path is not None
            else (
                software / "vendor" / "LinearPartition" / "linearpartition"
            ).resolve()
        )
        linearpartition_path = str(resolved_linearpartition_path)
        linearpartition_revision = git_revision(
            software / "vendor" / "LinearPartition"
        )
    worker = partial(
        evaluate_record,
        bpp_backend=args.bpp_backend,
        temperature=args.temperature,
        bpp_beam_size=args.bpp_beam_size,
        bpp_cutoff=args.bpp_cutoff,
        linearpartition_path=linearpartition_path,
    )

    if args.cpus == 1:
        evaluated = map(worker, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.cpus)
        evaluated = executor.map(worker, records, chunksize=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(f"{args.output.name}.partial")
    try:
        with partial_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for index, method_rows in enumerate(evaluated, start=1):
                writer.writerows(method_rows)
                if index % 250 == 0 or index == len(records):
                    handle.flush()
                    print(f"Completed {index}/{len(records)}", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown()
    partial_output.replace(args.output)
    manifest = {
        "command": sys.argv,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "selected_cases": len(records),
        "max_length": args.max_length,
        "deduplicate_sequences": args.deduplicate_sequences,
        "seed": args.seed,
        "limit": args.limit,
        "cpus": args.cpus,
        "bpp_backend": args.bpp_backend,
        "temperature_celsius": args.temperature,
        "bpp_beam_size": (
            args.bpp_beam_size if args.bpp_backend == "linearpartition" else None
        ),
        "bpp_cutoff": (
            args.bpp_cutoff if args.bpp_backend == "linearpartition" else None
        ),
        "linearpartition_path": linearpartition_path,
        "linearpartition_revision": linearpartition_revision,
        "analysis_revision": git_revision(repository),
        "analysis_worktree_clean": git_worktree_is_clean(repository),
        "software_revision": git_revision(software),
        "software_worktree_clean": git_worktree_is_clean(software),
        "python": sys.version,
        "viennarna": RNA.__version__,
    }
    manifest_path = args.output.with_name(f"{args.output.stem}.run_manifest.json")
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
