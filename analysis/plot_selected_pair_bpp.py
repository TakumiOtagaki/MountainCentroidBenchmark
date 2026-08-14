#!/usr/bin/env python3
"""Plot BPP support for base pairs present in benchmark structures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "reference",
    "vienna_mfe",
    "vienna_centroid",
    "gamma_centroid_2",
    "mountain_centroid_sequence_constrained",
)
LABELS = (
    "Reference",
    "MFE",
    r"Centroid ($\gamma=1$)",
    r"Centroid ($\gamma=2$)",
    "Mountain Centroid",
)
COLORS = ("#222222", "#777777", "#4C78A8", "#72B7B2", "#E45756")
LINESTYLES = ("--", ":", "-", "-.", "-")


def load_pair_probabilities(case_dir: Path) -> dict[str, list[float]]:
    values = {method: [] for method in METHODS}
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no case files found in {case_dir}")
    for path in paths:
        with path.open() as handle:
            case = json.load(handle)
        for method in METHODS:
            values[method].extend(case["selected_pair_bpps"][method])
    return values


def load_sequence_fractions(path: Path) -> dict[str, list[float]]:
    values = {method: [] for method in METHODS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            value = row["fraction_below_0.01"]
            if method in values and value:
                values[method].append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Verified output directory from evaluate_selected_pair_bpp.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/selected_pair_bpp"),
    )
    args = parser.parse_args()

    pair_values = load_pair_probabilities(args.input_dir / "cases")
    sequence_values = load_sequence_fractions(
        args.input_dir / "sequence_summary.csv"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    probability_axis, sequence_axis = axes
    probability_grid = np.linspace(0.0, 1.0, 1001)
    for method, label, color, linestyle in zip(
        METHODS,
        LABELS,
        COLORS,
        LINESTYLES,
    ):
        values = np.sort(np.asarray(pair_values[method], dtype=float))
        cumulative_fraction = np.searchsorted(
            values,
            probability_grid,
            side="right",
        ) / len(values)
        probability_axis.plot(
            probability_grid,
            cumulative_fraction,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
    probability_axis.set_xlim(0.0, 1.0)
    probability_axis.set_ylim(0.0, 1.0)
    probability_axis.set_xlabel("Base-pair probability")
    probability_axis.set_ylabel("Cumulative fraction of base pairs")
    probability_axis.set_title("(a) All base pairs", loc="left")
    probability_axis.grid(alpha=0.25)
    probability_axis.spines[["top", "right"]].set_visible(False)

    distributions = [sequence_values[method] for method in METHODS]
    boxes = sequence_axis.boxplot(
        distributions,
        tick_labels=LABELS,
        patch_artist=True,
        widths=0.62,
        whis=(5, 95),
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.72)
        box.set_edgecolor(color)
    sequence_axis.set_ylim(0.0, 0.65)
    mfe_nonzero = [value for value in sequence_values["vienna_mfe"] if value > 0.0]
    sequence_axis.scatter(
        [2] * len(mfe_nonzero),
        mfe_nonzero,
        s=22,
        facecolor="white",
        edgecolor=COLORS[1],
        linewidth=1.2,
        zorder=4,
    )
    sequence_axis.text(
        2,
        max(mfe_nonzero) + 0.018,
        "2 sequences > 0",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    for position, color in zip((3, 4), COLORS[2:4]):
        sequence_axis.scatter(
            position,
            0.0,
            marker="_",
            s=120,
            linewidth=2.0,
            color=color,
            clip_on=False,
            zorder=4,
        )
        sequence_axis.text(
            position,
            0.015,
            "all zero",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )
    sequence_axis.set_ylabel("Fraction with BPP < 0.01")
    sequence_axis.set_title("(b) Fractions within benchmark records", loc="left")
    sequence_axis.tick_params(axis="x", rotation=24, labelsize=8)
    sequence_axis.grid(axis="y", alpha=0.25)
    sequence_axis.spines[["top", "right"]].set_visible(False)

    handles, labels = probability_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
