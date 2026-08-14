#!/usr/bin/env python3
"""Plot low-BPP and paired gamma-2 summaries across hybrid weights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def alpha_label(alpha: float) -> str:
    if alpha == 0.0 or alpha == 1.0:
        return f"{alpha:g}"
    if alpha < 0.001:
        return f"{alpha:.1g}"
    return f"{alpha:.3g}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    field: (
                        int(value)
                        if field == "n"
                        else float(value)
                    )
                    for field, value in row.items()
                }
            )
    if not rows:
        raise ValueError(f"summary contains no rows: {path}")
    return sorted(rows, key=lambda row: row["alpha"])


def save_figure(figure: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def format_alpha_axis(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    positions = list(range(len(rows)))
    axis.set_xticks(positions, [alpha_label(row["alpha"]) for row in rows])
    axis.tick_params(axis="x", rotation=35, labelsize=8)
    axis.set_xlabel(r"Hybrid weight $\alpha$")
    axis.set_xlim(-0.35, len(rows) - 0.65)


def plot_low_bpp(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(7.4, 4.1))
    axis.plot(
        positions,
        [
            row["fraction_records_with_any_selected_pair_bpp_below_0.01"]
            for row in rows
        ],
        color="#E45756",
        linewidth=1.9,
        marker="o",
        markersize=4.5,
        label="Benchmark records with at least one low-BPP pair",
    )
    axis.plot(
        positions,
        [row["median_fraction_selected_pair_bpp_below_0.01"] for row in rows],
        color="#4C78A8",
        linewidth=1.9,
        marker="s",
        markersize=4.2,
        label="Median within-structure fraction of low-BPP pairs",
    )
    format_alpha_axis(axis, rows)
    axis.set_ylim(0.0, 0.85)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Fraction")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False, fontsize=8.5)
    figure.tight_layout()
    save_figure(figure, output_prefix)


def plot_gamma2_comparison(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    positions = list(range(len(rows)))
    categories = (
        (
            "fraction_hybrid_pareto_dominates_gamma2",
            "Hybrid no worse; at least one metric better",
            "#4C78A8",
        ),
        ("fraction_equal_to_gamma2", "Equal on both metrics", "#BAB0AC"),
        ("fraction_mixed_vs_gamma2", "One metric favors each", "#F28E2B"),
        (
            "fraction_gamma2_pareto_dominates_hybrid",
            r"Centroid ($\gamma=2$) no worse; at least one metric better",
            "#59A14F",
        ),
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    bottoms = [0.0] * len(rows)
    for field, label, color in categories:
        values = [row[field] for row in rows]
        axis.bar(
            positions,
            values,
            width=0.78,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    format_alpha_axis(axis, rows)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Fraction of benchmark records")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.3,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(figure, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--low-bpp-output-prefix",
        type=Path,
        default=Path("figures/hybrid_low_bpp"),
    )
    parser.add_argument(
        "--gamma2-output-prefix",
        type=Path,
        default=Path("figures/hybrid_gamma2_paired"),
    )
    args = parser.parse_args()

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_rows(args.summary)
    plot_low_bpp(rows, args.low_bpp_output_prefix)
    plot_gamma2_comparison(rows, args.gamma2_output_prefix)


if __name__ == "__main__":
    main()
