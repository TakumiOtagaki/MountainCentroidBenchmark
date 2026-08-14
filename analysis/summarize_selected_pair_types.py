#!/usr/bin/env python3
"""Summarize nucleotide-pair types among pairs with low BPP."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analysis.evaluate_selected_pair_bpp import METHODS, case_path, load_predictions
from analysis.reference_metrics import pairs_from_extended_dot_bracket


PAIR_CLASSES = {
    "AU": "AU",
    "UA": "AU",
    "GC": "GC",
    "CG": "GC",
    "GU": "GU",
    "UG": "GU",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--gamma-metrics", type=Path, required=True)
    parser.add_argument("--selected-pair-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    records = load_predictions(args.baseline_metrics, args.gamma_metrics)
    all_counts = {method: Counter() for method in METHODS}
    low_counts = {method: Counter() for method in METHODS}
    for record in records:
        with case_path(args.selected_pair_dir, record["id"]).open() as handle:
            selected_bpps = json.load(handle)["selected_pair_bpps"]
        sequence = record["sequence"]
        for method in METHODS:
            pairs = pairs_from_extended_dot_bracket(record["structures"][method])
            probabilities = selected_bpps[method]
            if len(pairs) != len(probabilities):
                raise ValueError(f"pair/BPP count differs for {record['id']} {method}")
            for (left, right), probability in zip(pairs, probabilities):
                pair_type = sequence[left - 1] + sequence[right - 1]
                pair_class = PAIR_CLASSES.get(pair_type, "other")
                all_counts[method][pair_class] += 1
                if probability < args.threshold:
                    low_counts[method][pair_class] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "method",
                "pair_class",
                "selected_pair_count",
                "below_threshold_count",
                "fraction_below_threshold",
                "fraction_of_below_threshold_pairs",
            )
        )
        for method in METHODS:
            total_low = sum(low_counts[method].values())
            for pair_class in ("AU", "GC", "GU", "other"):
                selected = all_counts[method][pair_class]
                below = low_counts[method][pair_class]
                writer.writerow(
                    (
                        method,
                        pair_class,
                        selected,
                        below,
                        below / selected if selected else "",
                        below / total_low if total_low else "",
                    )
                )


if __name__ == "__main__":
    main()
