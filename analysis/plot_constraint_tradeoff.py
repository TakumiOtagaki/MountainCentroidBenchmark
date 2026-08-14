#!/usr/bin/env python3
"""Summarize and plot the effect of RNA structure constraints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ALLOWED_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}
TURN = 3
OBJECTIVE_GAP_REL_TOLERANCE = 1e-8
VIOLATION_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2--3", 2, 3),
    ("4--7", 4, 7),
    ("8+", 8, None),
)


def parse_pairs(structure: str) -> list[tuple[int, int]]:
    """Return zero-indexed pairs from a pseudoknot-free dot-bracket string."""
    stack: list[int] = []
    pairs = []
    for position, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced closing parenthesis")
            pairs.append((stack.pop(), position))
        elif symbol != ".":
            raise ValueError(f"unsupported structure symbol: {symbol}")
    if stack:
        raise ValueError("unbalanced opening parenthesis")
    return pairs


def count_violating_pairs(
    sequence: str,
    structure: str,
    *,
    turn: int = TURN,
) -> int:
    """Count pairs violating pairability or the minimum hairpin length."""
    if len(sequence) != len(structure):
        raise ValueError("sequence and structure lengths differ")
    return sum(
        sequence[left] + sequence[right] not in ALLOWED_PAIRS
        or right - left - 1 < turn
        for left, right in parse_pairs(structure)
    )


def mountain_distance_upper_scale(length: int) -> int:
    """Return the length-specific denominator used to normalize NMSMD."""
    if length < 2:
        raise ValueError("length must be at least two")
    return sum(min(k, length - k) ** 2 for k in range(1, length))


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including ties, without requiring SciPy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Spearman's rho using average ranks for ties."""
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def load_tradeoff_rows(path: Path) -> list[dict[str, float | int | str]]:
    """Load, validate, and derive per-sequence constraint-effect metrics."""
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("constraint metrics are empty")
    if len({row["id"] for row in source_rows}) != len(source_rows):
        raise ValueError("constraint metrics contain duplicate IDs")

    output = []
    for row in source_rows:
        sequence = row["sequence"]
        relaxed = row["relaxed_structure"]
        constrained = row["constrained_structure"]
        if count_violating_pairs(sequence, constrained) != 0:
            raise ValueError(f"constrained structure violates rules: {row['id']}")

        constrained_objective = float(row["constrained_objective"])
        objective_gap = float(row["objective_gap"])
        tolerance = OBJECTIVE_GAP_REL_TOLERANCE * max(
            1.0, abs(constrained_objective)
        )
        if objective_gap < -tolerance:
            raise ValueError(
                f"negative objective gap exceeds tolerance: {row['id']}"
            )
        objective_gap = max(0.0, objective_gap)
        length = int(row["length"])
        output.append(
            {
                "id": row["id"],
                "length": length,
                "family": row["family"],
                "sequence": sequence,
                "reference_structure": row["reference_structure"],
                "violating_pair_count": count_violating_pairs(
                    sequence,
                    relaxed,
                ),
                "normalized_objective_gap": (
                    objective_gap / mountain_distance_upper_scale(length)
                ),
                "delta_base_pair_f1": (
                    float(row["constrained_base_pair_f1"])
                    - float(row["relaxed_base_pair_f1"])
                ),
            }
        )
    return output


def deduplicate_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Retain one row per sequence when all references for it agree."""
    by_sequence: dict[str, list[dict[str, float | int | str]]] = defaultdict(
        list
    )
    for row in rows:
        by_sequence[str(row["sequence"])].append(row)
    return [
        sequence_rows[0]
        for sequence_rows in by_sequence.values()
        if len(
            {
                str(row["reference_structure"])
                for row in sequence_rows
            }
        )
        == 1
    ]


def rows_in_violation_bin(
    rows: list[dict[str, float | int | str]],
    lower: int,
    upper: int | None,
) -> list[dict[str, float | int | str]]:
    """Select rows in an inclusive violation-count bin."""
    return [
        row
        for row in rows
        if int(row["violating_pair_count"]) >= lower
        and (upper is None or int(row["violating_pair_count"]) <= upper)
    ]


def write_summaries(
    rows: list[dict[str, float | int | str]],
    overall_path: Path,
    bins_path: Path,
) -> None:
    """Write overall and violation-bin summaries."""
    scopes = {"all": rows, "deduplicated": deduplicate_rows(rows)}
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    with overall_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("scope", "metric", "value"))
        for scope, scope_rows in scopes.items():
            normalized_gaps = np.asarray(
                [row["normalized_objective_gap"] for row in scope_rows],
                dtype=float,
            )
            delta_f1 = np.asarray(
                [row["delta_base_pair_f1"] for row in scope_rows],
                dtype=float,
            )
            violation_counts = np.asarray(
                [row["violating_pair_count"] for row in scope_rows],
                dtype=float,
            )
            overall = {
                "sequence_count": len(scope_rows),
                "normalized_objective_gap_median": float(
                    np.median(normalized_gaps)
                ),
                "normalized_objective_gap_q1": float(
                    np.percentile(normalized_gaps, 25)
                ),
                "normalized_objective_gap_q3": float(
                    np.percentile(normalized_gaps, 75)
                ),
                "delta_base_pair_f1_mean": float(np.mean(delta_f1)),
                "delta_base_pair_f1_median": float(np.median(delta_f1)),
                "delta_base_pair_f1_q1": float(
                    np.percentile(delta_f1, 25)
                ),
                "delta_base_pair_f1_q3": float(
                    np.percentile(delta_f1, 75)
                ),
                "delta_base_pair_f1_improved_fraction": float(
                    np.mean(delta_f1 > 0)
                ),
                "delta_base_pair_f1_unchanged_fraction": float(
                    np.mean(delta_f1 == 0)
                ),
                "delta_base_pair_f1_decreased_fraction": float(
                    np.mean(delta_f1 < 0)
                ),
                "violating_pair_count_median": float(
                    np.median(violation_counts)
                ),
                "violating_pair_count_q1": float(
                    np.percentile(violation_counts, 25)
                ),
                "violating_pair_count_q3": float(
                    np.percentile(violation_counts, 75)
                ),
                "violating_pair_count_maximum": float(
                    np.max(violation_counts)
                ),
                "normalized_gap_delta_f1_spearman": spearman_correlation(
                    normalized_gaps,
                    delta_f1,
                ),
                "violation_count_delta_f1_spearman": spearman_correlation(
                    violation_counts,
                    delta_f1,
                ),
            }
            writer.writerows(
                (scope, metric, value)
                for metric, value in overall.items()
            )

    with bins_path.open("w", newline="") as handle:
        fieldnames = (
            "scope",
            "violating_pair_count",
            "n",
            "delta_base_pair_f1_median",
            "delta_base_pair_f1_q1",
            "delta_base_pair_f1_q3",
            "improved_fraction",
            "unchanged_fraction",
            "decreased_fraction",
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for scope, scope_rows in scopes.items():
            for label, lower, upper in VIOLATION_BINS:
                bin_rows = rows_in_violation_bin(scope_rows, lower, upper)
                values = np.asarray(
                    [row["delta_base_pair_f1"] for row in bin_rows],
                    dtype=float,
                )
                writer.writerow(
                    {
                        "scope": scope,
                        "violating_pair_count": label,
                        "n": len(values),
                        "delta_base_pair_f1_median": float(np.median(values)),
                        "delta_base_pair_f1_q1": float(
                            np.percentile(values, 25)
                        ),
                        "delta_base_pair_f1_q3": float(
                            np.percentile(values, 75)
                        ),
                        "improved_fraction": float(np.mean(values > 0)),
                        "unchanged_fraction": float(np.mean(values == 0)),
                        "decreased_fraction": float(np.mean(values < 0)),
                    }
                )


def plot_tradeoff(
    rows: list[dict[str, float | int | str]],
    output_prefix: Path,
) -> None:
    """Create the two-panel constraint-effect figure."""
    normalized_gaps = np.asarray(
        [row["normalized_objective_gap"] for row in rows],
        dtype=float,
    )
    delta_f1 = np.asarray(
        [row["delta_base_pair_f1"] for row in rows],
        dtype=float,
    )
    violation_counts = np.asarray(
        [row["violating_pair_count"] for row in rows],
        dtype=float,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
    density = axes[0].hexbin(
        np.sqrt(normalized_gaps),
        delta_f1,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axes[0].axhline(0, color="#333333", linewidth=0.9, alpha=0.7)
    axes[0].set_xticks((0, 0.01, 0.02, 0.03, 0.04))
    axes[0].set_xlabel(r"$\sqrt{\Delta J_{\mathrm{norm}}}$")
    axes[0].set_ylabel(r"$\Delta$ base-pair F1")
    axes[0].set_title("Objective and base-pair trade-off")
    rho = spearman_correlation(normalized_gaps, delta_f1)
    axes[0].text(
        0.03,
        0.95,
        rf"Spearman $\rho={rho:.3f}$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
    )
    colorbar = figure.colorbar(density, ax=axes[0], pad=0.02)
    colorbar.set_label("Sequence count")

    bin_rows = [
        rows_in_violation_bin(rows, lower, upper)
        for _, lower, upper in VIOLATION_BINS
    ]
    improved = np.asarray(
        [
            np.mean(
                [float(row["delta_base_pair_f1"]) > 0 for row in group]
            )
            for group in bin_rows
        ]
    )
    unchanged = np.asarray(
        [
            np.mean(
                [float(row["delta_base_pair_f1"]) == 0 for row in group]
            )
            for group in bin_rows
        ]
    )
    decreased = 1.0 - improved - unchanged
    positions = np.arange(len(VIOLATION_BINS))
    axes[1].bar(
        positions,
        decreased,
        color="#D55E00",
        label="Decreased",
    )
    axes[1].bar(
        positions,
        unchanged,
        bottom=decreased,
        color="#BDBDBD",
        label="Unchanged",
    )
    axes[1].bar(
        positions,
        improved,
        bottom=decreased + unchanged,
        color="#0072B2",
        label="Improved",
    )
    for position, fraction in zip(positions, improved):
        axes[1].text(
            position,
            1.02,
            f"{100 * fraction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(
        positions,
        labels=[label for label, _, _ in VIOLATION_BINS],
    )
    axes[1].set_ylim(0, 1.10)
    axes[1].set_xlabel("Violating pairs in the path relaxation")
    axes[1].set_ylabel("Fraction of sequences")
    axes[1].set_title("Base-pair F1 change after RNA pairing constraints")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        labels=("F1 decreased", "F1 unchanged", "F1 improved"),
        columnspacing=1.2,
        handlelength=1.5,
    )
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout(w_pad=1.2)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Per-sequence output from evaluate_sequence_constrained.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/constraint_tradeoff"),
    )
    parser.add_argument(
        "--overall-summary",
        type=Path,
        default=Path(
            "results/structure_prediction/summary_constraint_tradeoff.csv"
        ),
    )
    parser.add_argument(
        "--bin-summary",
        type=Path,
        default=Path(
            "results/structure_prediction/summary_constraint_violation_bins.csv"
        ),
    )
    args = parser.parse_args()

    rows = load_tradeoff_rows(args.input)
    write_summaries(rows, args.overall_summary, args.bin_summary)
    plot_tradeoff(rows, args.output_prefix)
    print(f"Summarized and plotted {len(rows)} sequences")


if __name__ == "__main__":
    main()
