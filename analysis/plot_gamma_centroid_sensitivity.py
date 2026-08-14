#!/usr/bin/env python3
"""Plot median BP-F1 versus NMSMD across gamma-centroid settings."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMA_COLOR = "#4C78A8"
MFE_COLOR = "#777777"
MOUNTAIN_COLOR = "#E45756"
BASELINE_METHODS = (
    "vienna_mfe",
    "mountain_centroid_sequence_constrained",
)
METRICS = (
    ("base_pair_f1", "Base-pair F1"),
    ("normalized_squared_mountain_distance", "NMSMD"),
)


def load_gamma_metrics(path: Path) -> dict[float, dict[str, list[float]]]:
    values: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric, _ in METRICS}
    )
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            gamma = float(row["gamma"])
            for metric, _ in METRICS:
                values[gamma][metric].append(float(row[metric]))
    return dict(values)


def load_baseline_metrics(
    path: Path,
) -> dict[str, dict[str, list[float]]]:
    values = {
        method: {metric: [] for metric, _ in METRICS}
        for method in BASELINE_METHODS
    }
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in values:
                continue
            for metric, _ in METRICS:
                values[method][metric].append(float(row[metric]))
    return values


def summary(values: list[float]) -> tuple[int, float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        len(array),
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
    )


def write_summary(
    path: Path,
    gamma_values: dict[float, dict[str, list[float]]],
    baseline_values: dict[str, dict[str, list[float]]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "gamma", "metric", "n", "median", "q1", "q3"))
        for gamma in sorted(gamma_values):
            for metric, _ in METRICS:
                writer.writerow(
                    (
                        "gamma_centroid",
                        gamma,
                        metric,
                        *summary(gamma_values[gamma][metric]),
                    )
                )
        for method in BASELINE_METHODS:
            for metric, _ in METRICS:
                writer.writerow(
                    (
                        method,
                        "",
                        metric,
                        *summary(baseline_values[method][metric]),
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gamma-metrics",
        type=Path,
        default=Path("results/raw/gamma_centroid/metrics.csv"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        required=True,
        help="Verified four-method benchmark metrics",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/gamma_centroid_sensitivity"),
    )
    args = parser.parse_args()

    gamma_values = load_gamma_metrics(args.gamma_metrics)
    baseline_values = load_baseline_metrics(args.baseline_metrics)
    gammas = sorted(gamma_values)
    if 1.0 not in gamma_values:
        raise ValueError("gamma=1 is required to identify the centroid")
    case_counts = {
        len(values[metric])
        for values in gamma_values.values()
        for metric, _ in METRICS
    }
    case_counts.update(
        len(values[metric])
        for values in baseline_values.values()
        for metric, _ in METRICS
    )
    if len(case_counts) != 1:
        raise ValueError(f"metric series have inconsistent sizes: {case_counts}")

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
    figure, axis = plt.subplots(figsize=(5.6, 4.4))
    axis.plot(
        f1_medians,
        nmsmd_medians,
        color=GAMMA_COLOR,
        marker="o",
        markersize=5.5,
        linewidth=1.6,
        label=r"$\gamma$-centroid",
        zorder=3,
    )
    centroid_index = gammas.index(1.0)
    axis.scatter(
        f1_medians[centroid_index],
        nmsmd_medians[centroid_index],
        s=72,
        facecolor="white",
        edgecolor=GAMMA_COLOR,
        linewidth=1.8,
        label=r"Centroid ($\gamma=1$)",
        zorder=4,
    )
    label_offsets = {
        0.25: (7, 8),
        0.5: (7, 8),
        1.0: (10, 14),
        2.0: (10, -22),
        4.0: (-48, -16),
        8.0: (-48, 0),
        16.0: (-48, 17),
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
            fontsize=8.5,
            arrowprops={
                "arrowstyle": "-",
                "color": GAMMA_COLOR,
                "linewidth": 0.6,
                "alpha": 0.7,
            },
        )

    baseline_specs = (
        ("vienna_mfe", "MFE", MFE_COLOR, "D"),
        (
            "mountain_centroid_sequence_constrained",
            "Mountain Centroid",
            MOUNTAIN_COLOR,
            "s",
        ),
    )
    for method, label, color, marker in baseline_specs:
        x_value = summary(baseline_values[method]["base_pair_f1"])[1]
        y_value = summary(
            baseline_values[method]["normalized_squared_mountain_distance"]
        )[1]
        axis.scatter(
            x_value,
            y_value,
            s=58,
            color=color,
            marker=marker,
            label=label,
            zorder=4,
        )

    axis.set_xlabel("Median base-pair F1\n(higher is better)")
    axis.set_ylabel("Median NMSMD\n(lower is better)")
    axis.set_xlim(0.53, 0.73)
    axis.set_ylim(0.005, 0.0153)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    write_summary(
        args.output_prefix.with_suffix(".summary.csv"),
        gamma_values,
        baseline_values,
    )


if __name__ == "__main__":
    main()
