#!/usr/bin/env python3
"""Select a representative, favorable Mountain Centroid case deterministically."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


SELECTION_METHODS = (
    "vienna_mfe",
    "mountain_centroid_relaxed",
    "mountain_centroid_sequence_constrained",
)
OUTPUT_METHODS = (
    "vienna_mfe",
    "vienna_centroid",
    "mountain_centroid_relaxed",
    "mountain_centroid_sequence_constrained",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/raw/four_method_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/structure_prediction/case_study.csv"),
    )
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=200)
    args = parser.parse_args()

    cases: dict[str, dict] = {}
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in OUTPUT_METHODS:
                continue
            case = cases.setdefault(
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
            case["methods"][row["method"]] = row

    constrained_f1_median = statistics.median(
        float(case["methods"]["mountain_centroid_sequence_constrained"]["base_pair_f1"])
        for case in cases.values()
        if "mountain_centroid_sequence_constrained" in case["methods"]
    )
    eligible = []
    for case in cases.values():
        if not args.min_length <= case["length"] <= args.max_length:
            continue
        if not all(method in case["methods"] for method in OUTPUT_METHODS):
            continue
        mfe = case["methods"]["vienna_mfe"]
        relaxed = case["methods"]["mountain_centroid_relaxed"]
        constrained = case["methods"]["mountain_centroid_sequence_constrained"]
        mfe_md = float(mfe["mean_squared_mountain_distance"])
        relaxed_md = float(relaxed["mean_squared_mountain_distance"])
        constrained_md = float(constrained["mean_squared_mountain_distance"])
        if not relaxed_md < constrained_md < mfe_md:
            continue
        relaxed_f1 = float(relaxed["base_pair_f1"])
        constrained_f1 = float(constrained["base_pair_f1"])
        mfe_f1 = float(mfe["base_pair_f1"])
        if not relaxed_f1 < constrained_f1 < mfe_f1:
            continue
        if constrained_f1 < constrained_f1_median:
            continue
        case["constrained_minus_mfe_msmd"] = constrained_md - mfe_md
        eligible.append(case)

    if not eligible:
        raise SystemExit("No case satisfies the case-study selection policy")
    median_difference = statistics.median(
        case["constrained_minus_mfe_msmd"] for case in eligible
    )
    selected = min(
        eligible,
        key=lambda case: (
            abs(case["constrained_minus_mfe_msmd"] - median_difference),
            case["id"],
        ),
    )

    rows = []
    for method in OUTPUT_METHODS:
        prediction = selected["methods"][method]
        rows.append(
            {
                "selection_eligible_n": len(eligible),
                "constrained_f1_selection_floor": constrained_f1_median,
                "eligible_median_constrained_minus_mfe_msmd": median_difference,
                "id": selected["id"],
                "family": selected["family"],
                "subfamily": selected["subfamily"],
                "length": selected["length"],
                "method": method,
                "base_pair_f1": prediction["base_pair_f1"],
                "mean_squared_mountain_distance": prediction[
                    "mean_squared_mountain_distance"
                ],
                "sequence": selected["sequence"],
                "reference_structure": selected["reference_structure"],
                "predicted_structure": prediction["predicted_structure"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Selected {selected['id']} from {len(eligible)} eligible cases; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
