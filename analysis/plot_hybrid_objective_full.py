#!/usr/bin/env python3
"""Plot full and family-stratified hybrid objective trade-offs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCOPE_STYLES = {
    "all": ("Full benchmark", "#4C78A8", "-"),
    "deduplicated": ("Deduplicated", "#E45756", "--"),
}
FAMILY_LABELS = {
    "all_families": "All families",
    "16S_rRNA": "16S rRNA",
    "5S_rRNA": "5S rRNA",
    "RNaseP": "RNase P",
    "SRP": "SRP",
    "group_I_intron": "Group I intron",
    "tRNA": "tRNA",
    "tmRNA": "tmRNA",
}
BASELINE_STYLES = (
    ("MFE", "#777777", "D"),
    (r"Centroid ($\gamma=2$)", "#59A14F", "^"),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            converted["alpha"] = float(row["alpha"])
            if "endpoint_alpha" in row:
                converted["endpoint_alpha"] = float(row["endpoint_alpha"])
            converted["n"] = int(row["n"])
            for field, value in row.items():
                if field.startswith(("median_", "q1_", "q3_", "fraction_")):
                    converted[field] = float(value)
            rows.append(converted)
    return rows


def alpha_label(alpha: float) -> str:
    if alpha == 0.0 or alpha == 1.0:
        return f"{alpha:g}"
    if alpha < 0.001:
        return f"{alpha:.1g}"
    return f"{alpha:.3g}"


def scope_group_rows(
    rows: list[dict[str, Any]],
    dataset_scope: str,
    group: str,
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["dataset_scope"] == dataset_scope and row["group"] == group
        ),
        key=lambda row: row["alpha"],
    )


def load_baseline_points(
    benchmark_summary_path: Path,
    gamma_metrics_path: Path,
) -> list[tuple[str, float, float, str, str]]:
    mfe_metrics = {}
    with benchmark_summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["scope"] == "all"
                and row["group"] == "all_families"
                and row["method"] == "vienna_mfe"
            ):
                mfe_metrics[row["metric"]] = float(row["median"])

    gamma_f1 = []
    gamma_nmsmd = []
    with gamma_metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["gamma"]) == 2.0:
                gamma_f1.append(float(row["base_pair_f1"]))
                gamma_nmsmd.append(float(row["normalized_squared_mountain_distance"]))

    points = (
        (
            mfe_metrics["base_pair_f1"],
            mfe_metrics["normalized_squared_mountain_distance"],
        ),
        (median(gamma_f1), median(gamma_nmsmd)),
    )
    return [
        (label, x_value, y_value, color, marker)
        for (label, color, marker), (x_value, y_value) in zip(
            BASELINE_STYLES,
            points,
        )
    ]


def save_figure(figure: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_overall(
    distribution_rows: list[dict[str, Any]],
    baseline_points: list[tuple[str, float, float, str, str]],
    output_prefix: Path,
) -> None:
    figure, tradeoff_axis = plt.subplots(figsize=(5.6, 4.4))
    distributions = scope_group_rows(distribution_rows, "all", "all_families")
    tradeoff_axis.plot(
        [row["median_base_pair_f1"] for row in distributions],
        [row["median_normalized_squared_mountain_distance"] for row in distributions],
        color=SCOPE_STYLES["all"][1],
        linewidth=1.8,
        marker="o",
        markersize=4.8,
        label="Hybrid objective",
    )

    for label, x_value, y_value, color, marker in baseline_points:
        tradeoff_axis.scatter(
            x_value,
            y_value,
            s=55,
            color=color,
            marker=marker,
            label=label,
            zorder=4,
        )

    full_rows = scope_group_rows(distribution_rows, "all", "all_families")
    annotated_indices = (0, 7, 8, len(full_rows) - 1)
    offsets = ((5, -13), (-45, -13), (5, -13), (-28, 8))
    for index, offset in zip(annotated_indices, offsets):
        row = full_rows[index]
        tradeoff_axis.annotate(
            rf"$\alpha={alpha_label(row['alpha'])}$",
            (
                row["median_base_pair_f1"],
                row["median_normalized_squared_mountain_distance"],
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )

    tradeoff_axis.set_xlabel("Median base-pair F1 (higher is better)")
    tradeoff_axis.set_ylabel("Median NMSMD (lower is better)")
    tradeoff_axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    tradeoff_axis.spines[["top", "right"]].set_visible(False)
    handles, labels = tradeoff_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(figure, output_prefix)
    plt.close(figure)


def plot_families(
    distribution_rows: list[dict[str, Any]],
    output_prefix: Path,
) -> None:
    families = ["all_families"] + sorted(
        (family for family in FAMILY_LABELS if family != "all_families"),
        key=lambda family: -scope_group_rows(
            distribution_rows,
            "all",
            family,
        )[0]["n"],
    )
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.5))
    for axis, family in zip(axes.flat, families):
        counts = {}
        for scope, (label, color, linestyle) in SCOPE_STYLES.items():
            rows = scope_group_rows(distribution_rows, scope, family)
            counts[scope] = rows[0]["n"]
            axis.plot(
                [row["median_base_pair_f1"] for row in rows],
                [row["median_normalized_squared_mountain_distance"] for row in rows],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                marker="o",
                markersize=3.4,
                label=label,
            )
        full_rows = scope_group_rows(distribution_rows, "all", family)
        for index, text_offset in ((0, (4, -12)), (-1, (-23, -14))):
            row = full_rows[index]
            axis.annotate(
                rf"$\alpha={alpha_label(row['alpha'])}$",
                (
                    row["median_base_pair_f1"],
                    row["median_normalized_squared_mountain_distance"],
                ),
                xytext=text_offset,
                textcoords="offset points",
                fontsize=7.2,
            )
        axis.set_title(
            f"{FAMILY_LABELS[family]}\n"
            f"n={counts['all']:,}; deduplicated n={counts['deduplicated']:,}",
            fontsize=9.5,
        )
        axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    figure.supylabel("Median NMSMD", x=0.01)
    figure.supxlabel("Median base-pair F1", y=0.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.96))
    save_figure(figure, output_prefix)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-summary", type=Path, required=True)
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("results/structure_prediction/summary_distribution.csv"),
    )
    parser.add_argument(
        "--gamma-metrics",
        type=Path,
        default=Path("results/raw/gamma_centroid/metrics.csv"),
    )
    parser.add_argument(
        "--overall-output-prefix",
        type=Path,
        default=Path("figures/hybrid_objective_full_tradeoff"),
    )
    parser.add_argument(
        "--family-output-prefix",
        type=Path,
        default=Path("figures/hybrid_objective_full_by_family"),
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
    distribution_rows = load_rows(args.distribution_summary)
    baseline_points = load_baseline_points(
        args.benchmark_summary,
        args.gamma_metrics,
    )
    plot_overall(
        distribution_rows,
        baseline_points,
        args.overall_output_prefix,
    )
    plot_families(distribution_rows, args.family_output_prefix)


if __name__ == "__main__":
    main()
