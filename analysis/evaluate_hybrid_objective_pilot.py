#!/usr/bin/env python3
"""Evaluate a mountain/centroid hybrid objective on pilot or full records."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import RNA

from analysis.evaluate_gamma_centroid import (
    atomic_write_json,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
)
from analysis.reference_metrics import (
    base_pair_f1,
    normalized_squared_mountain_distance,
)
from mountain_centroid.bpp_mu import compute_bpp_and_mu
from mountain_centroid.evaluation import mountain_heights
from mountain_centroid.formatting import pairs_from_bracket
from mountain_centroid.sequence import MIN_HAIRPIN_LENGTH, can_pair


DEFAULT_ALPHA_ODDS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_ALPHAS = tuple(odds / (1.0 + odds) for odds in DEFAULT_ALPHA_ODDS) + (1.0,)
ENDPOINT_METHODS = (
    "vienna_centroid",
    "mountain_centroid_sequence_constrained",
)
FIELDNAMES = (
    "id",
    "family",
    "subfamily",
    "length",
    "alpha",
    "sequence",
    "reference_structure",
    "predicted_structure",
    "normalized_mountain_objective",
    "normalized_centroid_gain",
    "hybrid_objective",
    "base_pair_f1",
    "normalized_squared_mountain_distance",
    "predicted_pair_count",
    "median_selected_pair_bpp",
    "fraction_selected_pair_bpp_below_0.01",
    "bpp_seconds",
    "hybrid_seconds",
    "prediction_seconds",
    "states_evaluated",
    "partner_transitions_evaluated",
    "effective_depth_levels",
    "endpoint_structure_match",
    "config_signature",
)


def parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(item) for item in value.split(","))
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must lie in [0,1]")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must be distinct")
    return alphas


def load_records(path: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in ENDPOINT_METHODS:
                continue
            record = records.setdefault(
                row["id"],
                {
                    "id": row["id"],
                    "family": row.get("family", ""),
                    "subfamily": row.get("subfamily", ""),
                    "sequence": row["sequence"],
                    "reference_structure": row["reference_structure"],
                    "endpoint_structures": {},
                },
            )
            if record["sequence"] != row["sequence"]:
                raise ValueError(f"sequence differs across rows for {row['id']}")
            if record["reference_structure"] != row["reference_structure"]:
                raise ValueError(f"reference differs across rows for {row['id']}")
            record["endpoint_structures"][method] = row["predicted_structure"]

    complete = []
    for record in records.values():
        missing = set(ENDPOINT_METHODS) - set(record["endpoint_structures"])
        if missing:
            raise ValueError(f"{record['id']} is missing endpoints {sorted(missing)}")
        complete.append(record)
    complete.sort(key=lambda record: record["id"])
    return complete


def stratified_pilot(records: Sequence[dict[str, Any]], per_family: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(record["family"], []).append(record)

    selected = []
    for family in sorted(by_family):
        family_records = sorted(
            by_family[family],
            key=lambda record: (len(record["sequence"]), record["id"]),
        )
        if len(family_records) <= per_family:
            selected.extend(family_records)
            continue
        indices = [
            round(index * (len(family_records) - 1) / (per_family - 1))
            for index in range(per_family)
        ]
        if len(set(indices)) != per_family:
            raise RuntimeError(f"sampling produced duplicate indices for {family}")
        selected.extend(family_records[index] for index in indices)
    selected.sort(key=lambda record: record["id"])
    return selected


def select_records(
    records: Sequence[dict[str, Any]],
    per_family: int,
    all_records: bool,
) -> tuple[list[dict[str, Any]], str]:
    if all_records:
        return list(records), "all_records"
    return stratified_pilot(records, per_family), "length_ordered_even_spacing_within_family"


def mountain_normalizer(length: int) -> float:
    return float(sum(min(cut, length - cut) ** 2 for cut in range(1, length)))


def hybrid_structure(
    sequence: str,
    expected_heights: Sequence[float],
    bpp: np.ndarray,
    alpha: float,
) -> str:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    length = len(sequence)
    if len(expected_heights) != length - 1 or bpp.shape != (length, length):
        raise ValueError("hybrid inputs have inconsistent lengths")
    profile_scale = mountain_normalizer(length)
    pair_scale = float(length // 2)
    if profile_scale <= 0.0 or pair_scale <= 0.0:
        raise ValueError("hybrid pilot requires sequences of length at least two")

    if alpha < 1.0:
        mountain_weight = 1.0
        pair_weight = alpha * profile_scale / ((1.0 - alpha) * pair_scale)
    else:
        mountain_weight = 0.0
        pair_weight = 1.0

    legal_partners = tuple(
        tuple(
            right
            for right in range(left + MIN_HAIRPIN_LENGTH + 1, length)
            if can_pair(sequence[left], sequence[right])
        )
        for left in range(length)
    )

    def cut_cost(position: int, depth: int) -> float:
        if position == length - 1:
            return 0.0
        return mountain_weight * (depth - expected_heights[position]) ** 2

    def subproblem_cost(left: int, right: int, depth: int) -> float:
        if left > right:
            return 0.0
        return solve(left, right, depth)[0]

    @cache
    def solve(left: int, right: int, depth: int) -> tuple[float, int]:
        best_cost = cut_cost(left, depth) + subproblem_cost(left + 1, right, depth)
        best_partner = -1
        for partner in legal_partners[left]:
            if partner > right:
                break
            pair_gain = 2.0 * float(bpp[left, partner]) - 1.0
            candidate = (
                cut_cost(left, depth + 1)
                + subproblem_cost(left + 1, partner - 1, depth + 1)
                + cut_cost(partner, depth)
                + subproblem_cost(partner + 1, right, depth)
                - pair_weight * pair_gain
            )
            if candidate < best_cost:
                best_cost = candidate
                best_partner = partner
        return best_cost, best_partner

    characters = ["."] * length

    def traceback(left: int, right: int, depth: int) -> None:
        if left > right:
            return
        _, partner = solve(left, right, depth)
        if partner < 0:
            traceback(left + 1, right, depth)
            return
        characters[left] = "("
        characters[partner] = ")"
        traceback(left + 1, partner - 1, depth + 1)
        traceback(partner + 1, right, depth)

    traceback(0, length - 1, 0)
    return "".join(characters)


def objective_values(
    structure: str,
    expected_heights: Sequence[float],
    bpp: np.ndarray,
) -> tuple[float, float, list[float]]:
    length = len(structure)
    profile_objective = sum(
        (height - expected) ** 2
        for height, expected in zip(mountain_heights(structure), expected_heights)
    )
    selected_bpps = [
        float(bpp[left - 1, right - 1])
        for left, right in pairs_from_bracket(structure)
    ]
    centroid_gain = sum(2.0 * probability - 1.0 for probability in selected_bpps)
    return (
        profile_objective / mountain_normalizer(length),
        centroid_gain / float(length // 2),
        selected_bpps,
    )


def cpp_hybrid_structures(
    sequence: str,
    expected_heights: Sequence[float],
    bpp: np.ndarray,
    alphas: Sequence[float],
    executable: Path,
) -> list[dict[str, Any]]:
    if not executable.is_file():
        raise FileNotFoundError(f"hybrid solver not found: {executable}")
    length = len(sequence)
    upper_triangle = (
        repr(float(bpp[left, right]))
        for left in range(length)
        for right in range(left + 1, length)
    )
    payload = "\n".join(
        (
            sequence,
            str(len(alphas)),
            " ".join(repr(alpha) for alpha in alphas),
            " ".join(repr(float(value)) for value in expected_heights),
            " ".join(upper_triangle),
        )
    ) + "\n"
    completed = subprocess.run(
        [str(executable)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no diagnostic output"
        raise RuntimeError(f"hybrid C++ solver failed: {detail}")
    lines = completed.stdout.splitlines()
    if len(lines) != len(alphas):
        raise RuntimeError("hybrid C++ solver returned the wrong number of rows")

    results = []
    for expected_alpha, line in zip(alphas, lines):
        fields = line.split("\t")
        if len(fields) != 7:
            raise RuntimeError("hybrid C++ solver returned malformed output")
        alpha = float(fields[0])
        if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("hybrid C++ solver returned alphas out of order")
        results.append(
            {
                "alpha": alpha,
                "structure": fields[1],
                "solver_objective": float(fields[2]),
                "states_evaluated": int(fields[3]),
                "partner_transitions_evaluated": int(fields[4]),
                "effective_depth_levels": int(fields[5]),
                "hybrid_seconds": float(fields[6]),
            }
        )
    return results


def validate_structure(sequence: str, structure: str) -> None:
    if len(sequence) != len(structure):
        raise ValueError("sequence and structure lengths differ")
    for left, right in pairs_from_bracket(structure):
        if right - left - 1 < MIN_HAIRPIN_LENGTH:
            raise ValueError(f"pair ({left},{right}) violates TURN")
        if not can_pair(sequence[left - 1], sequence[right - 1]):
            raise ValueError(f"pair ({left},{right}) is not pairable")


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
        if existing.get("id") == record["id"] and existing.get("config_signature") == signature:
            return record["id"], True

    bpp_started = time.perf_counter()
    bpp, expected_heights = compute_bpp_and_mu(
        record["sequence"],
        temperature=task["temperature"],
        backend="vienna",
    )
    bpp_seconds = time.perf_counter() - bpp_started

    cpp_results = cpp_hybrid_structures(
        record["sequence"],
        expected_heights,
        bpp,
        task["alphas"],
        Path(task["hybrid_binary"]),
    )
    predictions = []
    endpoint_methods = {0.0: "mountain_centroid_sequence_constrained", 1.0: "vienna_centroid"}
    for cpp_result in cpp_results:
        alpha = cpp_result["alpha"]
        structure = cpp_result["structure"]
        validate_structure(record["sequence"], structure)
        mountain_objective, centroid_gain, selected_bpps = objective_values(
            structure,
            expected_heights,
            bpp,
        )
        hybrid_objective = (1.0 - alpha) * mountain_objective - alpha * centroid_gain
        if not math.isclose(
            cpp_result["solver_objective"],
            (
                mountain_objective * mountain_normalizer(len(record["sequence"]))
                - alpha * mountain_normalizer(len(record["sequence"]))
                * centroid_gain / (1.0 - alpha)
                if alpha < 1.0
                else -centroid_gain * (len(record["sequence"]) // 2)
            ),
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise RuntimeError(f"reported hybrid objective differs for {record['id']}")

        if len(record["sequence"]) <= task["python_oracle_max_length"]:
            oracle = hybrid_structure(record["sequence"], expected_heights, bpp, alpha)
            if structure != oracle:
                raise RuntimeError(f"C++/Python hybrid structures differ for {record['id']}")

        endpoint_match: bool | str = ""
        if alpha in endpoint_methods:
            endpoint = record["endpoint_structures"][endpoint_methods[alpha]]
            endpoint_match = structure == endpoint
            endpoint_mountain, endpoint_gain, _ = objective_values(endpoint, expected_heights, bpp)
            observed = mountain_objective if alpha == 0.0 else centroid_gain
            expected = endpoint_mountain if alpha == 0.0 else endpoint_gain
            if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f"endpoint objective differs for {record['id']} alpha={alpha}")

        predictions.append(
            {
                "alpha": alpha,
                "predicted_structure": structure,
                "normalized_mountain_objective": mountain_objective,
                "normalized_centroid_gain": centroid_gain,
                "hybrid_objective": hybrid_objective,
                "base_pair_f1": base_pair_f1(structure, record["reference_structure"]),
                "normalized_squared_mountain_distance": normalized_squared_mountain_distance(
                    structure,
                    record["reference_structure"],
                ),
                "predicted_pair_count": len(selected_bpps),
                "median_selected_pair_bpp": (
                    float(np.median(selected_bpps)) if selected_bpps else ""
                ),
                "fraction_selected_pair_bpp_below_0.01": (
                    float(np.mean(np.asarray(selected_bpps) < 0.01))
                    if selected_bpps
                    else ""
                ),
                "bpp_seconds": bpp_seconds,
                "hybrid_seconds": cpp_result["hybrid_seconds"],
                "prediction_seconds": bpp_seconds + cpp_result["hybrid_seconds"],
                "states_evaluated": cpp_result["states_evaluated"],
                "partner_transitions_evaluated": (
                    cpp_result["partner_transitions_evaluated"]
                ),
                "effective_depth_levels": cpp_result["effective_depth_levels"],
                "endpoint_structure_match": endpoint_match,
            }
        )

    atomic_write_json(
        output_path,
        {
            "id": record["id"],
            "family": record["family"],
            "subfamily": record["subfamily"],
            "sequence": record["sequence"],
            "reference_structure": record["reference_structure"],
            "length": len(record["sequence"]),
            "predictions": predictions,
            "config_signature": signature,
        },
    )
    return record["id"], False


def write_merged_csv(output_dir: Path, records: Sequence[dict[str, Any]]) -> Path:
    output = output_dir / "metrics.csv"
    partial = output.with_name(f".{output.name}.partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for record in records:
            with case_path(output_dir, record["id"]).open() as case_handle:
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
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--all-records", action="store_true")
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--alphas", type=parse_alphas, default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--hybrid-binary",
        type=Path,
        default=Path("build/hybrid_constrained_solver"),
    )
    parser.add_argument("--python-oracle-max-length", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not args.all_records and args.per_family < 2:
        raise ValueError("per-family must be at least two")
    setup_logging(args.log)

    all_records = load_records(args.baseline_metrics)
    records, sampling = select_records(all_records, args.per_family, args.all_records)
    repository = Path(__file__).resolve().parents[1]
    software = repository / "software" / "MountainCentroid"
    hybrid_binary = args.hybrid_binary.resolve()
    if not hybrid_binary.is_file():
        raise FileNotFoundError(f"hybrid solver not found: {hybrid_binary}")
    config = {
        "analysis_revision": git_revision(repository),
        "analysis_worktree_clean": git_worktree_is_clean(repository),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "hybrid_solver_sha256": file_sha256(
            repository / "analysis" / "hybrid_constrained_solver.cpp"
        ),
        "hybrid_binary_sha256": file_sha256(hybrid_binary),
        "software_revision": git_revision(software),
        "software_worktree_clean": git_worktree_is_clean(software),
        "baseline_metrics_sha256": file_sha256(args.baseline_metrics),
        "alphas": args.alphas,
        "per_family": None if args.all_records else args.per_family,
        "temperature_celsius": args.temperature,
        "sampling": sampling,
        "python_oracle_max_length": args.python_oracle_max_length,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    family_counts: dict[str, int] = {}
    for record in records:
        family_counts[record["family"]] = family_counts.get(record["family"], 0) + 1
    manifest = {
        **config,
        "config_signature": signature,
        "baseline_metrics": str(args.baseline_metrics.resolve()),
        "hybrid_binary": str(hybrid_binary),
        "selected_cases": len(records),
        "selected_ids": [record["id"] for record in records],
        "family_counts": family_counts,
        "workers": args.workers,
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
            "alphas": args.alphas,
            "hybrid_binary": str(hybrid_binary),
            "python_oracle_max_length": args.python_oracle_max_length,
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
            if completed % 25 == 0 or completed == len(tasks):
                logging.info("Completed %d/%d (%d resumed)", completed, len(tasks), resumed)
    output = write_merged_csv(args.output_dir, records)
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
