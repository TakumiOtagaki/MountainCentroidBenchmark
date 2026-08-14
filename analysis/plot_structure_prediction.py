#!/usr/bin/env python3
"""Plot the two primary structure-prediction metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "vienna_mfe",
    "vienna_centroid",
    "mountain_centroid_sequence_constrained",
    "mountain_centroid_relaxed",
)
METHOD_LABELS = (
    "MFE",
    "Centroid",
    "Mountain\nCentroid",
    "Mountain-path\nrelaxation",
)
COLORS = ("#777777", "#4C78A8", "#E45756", "#F2CF5B")


def draw_violin(axis, series, labels, colors) -> None:
    """Draw density, quartiles, and medians with consistent styling."""
    violins = axis.violinplot(
        series,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.25, 0.75] for _ in series],
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.8)
        body.set_alpha(0.78)
    violins["cmedians"].set_color("#111111")
    violins["cmedians"].set_linewidth(1.5)
    violins["cquantiles"].set_color("#333333")
    violins["cquantiles"].set_linewidth(0.9)
    axis.set_xticks(range(1, len(labels) + 1), labels=labels)


def sqrt_scale(values):
    return np.sqrt(np.asarray(values, dtype=float))


def signed_sqrt_scale(values):
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.sqrt(np.abs(array))


def set_signed_sqrt_ticks(axis, ticks) -> None:
    positions = [np.sign(tick) * np.sqrt(abs(tick)) for tick in ticks]
    axis.set_yticks(positions, labels=[f"{tick:g}" for tick in ticks])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/raw/four_method_metrics.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/structure_prediction"),
    )
    parser.add_argument(
        "--exclude-constrained",
        action="store_true",
        help="omit the pairability-constrained series from the plots",
    )
    args = parser.parse_args()

    if args.exclude_constrained:
        retained = [
            index
            for index, method in enumerate(METHOD_ORDER)
            if method != "mountain_centroid_sequence_constrained"
        ]
        method_order = tuple(METHOD_ORDER[index] for index in retained)
        method_labels = tuple(METHOD_LABELS[index] for index in retained)
        colors = tuple(COLORS[index] for index in retained)
    else:
        method_order = METHOD_ORDER
        method_labels = METHOD_LABELS
        colors = COLORS

    paired_rows: dict[str, dict[str, dict[str, float]]] = {}
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in method_order:
                continue
            metrics = {
                "base_pair_f1": float(row["base_pair_f1"]),
                "normalized_squared_mountain_distance": float(
                    row["normalized_squared_mountain_distance"]
                ),
            }
            paired_rows.setdefault(row["id"], {})[method] = metrics

    complete_cases = [
        methods
        for methods in paired_rows.values()
        if all(method in methods for method in method_order)
    ]
    values = {
        method: {
            metric: [case[method][metric] for case in complete_cases]
            for metric in (
                "base_pair_f1",
                "normalized_squared_mountain_distance",
            )
        }
        for method in method_order
    }

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    metric_specs = (
        ("base_pair_f1", "Base-pair F1"),
        (
            "normalized_squared_mountain_distance",
            "NMSMD",
        ),
    )
    for axis, (metric, label) in zip(axes, metric_specs):
        series = [values[method][metric] for method in method_order]
        if metric == "normalized_squared_mountain_distance":
            series = [sqrt_scale(method_values) for method_values in series]
        draw_violin(axis, series, method_labels, colors)
        if metric == "normalized_squared_mountain_distance":
            axis.set_ylabel("Root NMSMD")
        else:
            axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_yticks(np.arange(0.0, 0.7, 0.1))
    figure.tight_layout()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )

    summary_path = args.output_prefix.with_suffix(".summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "metric", "n", "median", "q1", "q3"))
        for method in method_order:
            for metric, _ in metric_specs:
                array = np.asarray(values[method][metric], dtype=float)
                writer.writerow(
                    (
                        method,
                        metric,
                        len(array),
                        float(np.median(array)),
                        float(np.percentile(array, 25)),
                        float(np.percentile(array, 75)),
                    )
                )

    comparison_methods = method_order[1:]
    comparison_labels = tuple(f"{label} − MFE" for label in method_labels[1:])
    paired_differences = {
        method: {"base_pair_f1": [], "normalized_squared_mountain_distance": []}
        for method in comparison_methods
    }
    for methods in complete_cases:
        for method in comparison_methods:
            for metric in paired_differences[method]:
                paired_differences[method][metric].append(
                    methods[method][metric] - methods["vienna_mfe"][metric]
                )

    paired_figure, paired_axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    paired_specs = (
        ("base_pair_f1", "Paired difference in base-pair F1", "positive is better"),
        (
            "normalized_squared_mountain_distance",
            "Paired difference in NMSMD",
            "negative is better",
        ),
    )
    for axis, (metric, label, direction) in zip(paired_axes, paired_specs):
        series = [paired_differences[method][metric] for method in comparison_methods]
        if metric == "normalized_squared_mountain_distance":
            series = [signed_sqrt_scale(method_values) for method_values in series]
        draw_violin(axis, series, comparison_labels, colors[1:])
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        axis.set_ylabel(label)
        axis.set_title(direction)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    set_signed_sqrt_ticks(
        paired_axes[1],
        (-0.2, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 0.2),
    )
    paired_figure.tight_layout()
    for suffix in ("png", "pdf"):
        paired_figure.savefig(
            Path(f"{args.output_prefix}.paired.{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )

    paired_summary_path = Path(f"{args.output_prefix}.paired.summary.csv")
    with paired_summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "method",
                "metric",
                "n",
                "median_difference_from_mfe",
                "strictly_favorable_fraction",
                "tie_fraction",
                "favorable_fraction",
            )
        )
        for method in comparison_methods:
            for metric, _, _ in paired_specs:
                array = np.asarray(paired_differences[method][metric], dtype=float)
                strictly_favorable = (
                    array > 0 if metric == "base_pair_f1" else array < 0
                )
                tied = array == 0
                favorable = strictly_favorable | tied
                writer.writerow(
                    (
                        method,
                        metric,
                        len(array),
                        float(np.median(array)),
                        float(np.mean(strictly_favorable)),
                        float(np.mean(tied)),
                        float(np.mean(favorable)),
                    )
                )
    print(
        f"Wrote {args.output_prefix}.png/.pdf, paired plots, and summaries"
    )


if __name__ == "__main__":
    main()
