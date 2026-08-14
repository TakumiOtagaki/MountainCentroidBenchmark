#!/usr/bin/env python3
"""Plot benchmark distributions with gamma-centroid sensitivity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_gamma_centroid_sensitivity import (
    GAMMA_COLOR,
    MFE_COLOR,
    MOUNTAIN_COLOR,
    load_gamma_metrics,
    summary,
)
from plot_structure_prediction import (
    COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    draw_violin,
    sqrt_scale,
)


def load_benchmark_metrics(path: Path) -> list[dict[str, dict[str, float]]]:
    paired_rows: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in METHOD_ORDER:
                continue
            paired_rows.setdefault(row["id"], {})[method] = {
                "base_pair_f1": float(row["base_pair_f1"]),
                "normalized_squared_mountain_distance": float(
                    row["normalized_squared_mountain_distance"]
                ),
            }
    return [
        methods
        for methods in paired_rows.values()
        if all(method in methods for method in METHOD_ORDER)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Verified four-method benchmark metrics",
    )
    parser.add_argument(
        "--gamma-input",
        type=Path,
        default=Path("results/raw/gamma_centroid/metrics.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/structure_prediction_with_gamma"),
    )
    args = parser.parse_args()

    cases = load_benchmark_metrics(args.input)
    values = {
        method: {
            metric: [case[method][metric] for case in cases]
            for metric in (
                "base_pair_f1",
                "normalized_squared_mountain_distance",
            )
        }
        for method in METHOD_ORDER
    }
    gamma_values = load_gamma_metrics(args.gamma_input)
    gammas = sorted(gamma_values)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.7),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)},
    )
    distribution_specs = (
        ("base_pair_f1", "Base-pair F1\n(higher is better)"),
        ("normalized_squared_mountain_distance", "Root NMSMD\n(lower is better)"),
    )
    for index, (axis, (metric, label)) in enumerate(
        zip(axes[:2], distribution_specs)
    ):
        series = [values[method][metric] for method in METHOD_ORDER]
        if metric == "normalized_squared_mountain_distance":
            series = [sqrt_scale(method_values) for method_values in series]
        draw_violin(axis, series, METHOD_LABELS, COLORS)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(f"({chr(ord('a') + index)}) {label.splitlines()[0]}", loc="left")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_yticks(np.arange(0.0, 0.7, 0.1))

    axis = axes[2]
    f1_medians = [
        summary(gamma_values[gamma]["base_pair_f1"])[1]
        for gamma in gammas
    ]
    nmsmd_medians = [
        summary(
            gamma_values[gamma]["normalized_squared_mountain_distance"]
        )[1]
        for gamma in gammas
    ]
    axis.plot(
        f1_medians,
        nmsmd_medians,
        color=GAMMA_COLOR,
        marker="o",
        markersize=4.5,
        linewidth=1.4,
        zorder=3,
    )
    centroid_index = gammas.index(1.0)
    axis.scatter(
        f1_medians[centroid_index],
        nmsmd_medians[centroid_index],
        s=56,
        facecolor="white",
        edgecolor=GAMMA_COLOR,
        linewidth=1.6,
        zorder=4,
    )
    label_offsets = {
        0.25: (5, 6),
        0.5: (5, 6),
        1.0: (8, 12),
        2.0: (8, -17),
        4.0: (-37, -13),
        8.0: (-37, 0),
        16.0: (-37, 14),
    }
    for gamma, x_value, y_value in zip(gammas, f1_medians, nmsmd_medians):
        x_offset, y_offset = label_offsets[gamma]
        axis.annotate(
            rf"$\gamma={gamma:g}$",
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            va="center",
            fontsize=7.5,
            arrowprops={
                "arrowstyle": "-",
                "color": GAMMA_COLOR,
                "linewidth": 0.5,
                "alpha": 0.7,
            },
        )

    baseline_specs = (
        ("vienna_mfe", "MFE", MFE_COLOR, "D", (-4, -14), "right"),
        (
            "mountain_centroid_sequence_constrained",
            "Mountain Centroid",
            MOUNTAIN_COLOR,
            "s",
            (5, 6),
            "left",
        ),
    )
    for method, label, color, marker, offset, alignment in baseline_specs:
        x_value = float(np.median(values[method]["base_pair_f1"]))
        y_value = float(
            np.median(values[method]["normalized_squared_mountain_distance"])
        )
        axis.scatter(
            x_value,
            y_value,
            s=44,
            color=color,
            marker=marker,
            zorder=4,
        )
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=7.5,
        )
    axis.set_xlabel("Median base-pair F1\n(higher is better)")
    axis.set_ylabel("Median NMSMD\n(lower is better)")
    axis.set_xlim(0.53, 0.73)
    axis.set_ylim(0.005, 0.0153)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("(c) Median comparison", loc="left")

    figure.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
