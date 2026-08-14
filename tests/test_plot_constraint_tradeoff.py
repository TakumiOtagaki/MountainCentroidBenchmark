from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "plot_constraint_tradeoff.py"
)
SPEC = importlib.util.spec_from_file_location("plot_constraint_tradeoff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_pairs_and_count_violations() -> None:
    assert MODULE.parse_pairs("((...))") == [(1, 5), (0, 6)]
    assert MODULE.count_violating_pairs("GCAAAUC", "((...))") == 1


def test_parse_pairs_rejects_unbalanced_structure() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        MODULE.parse_pairs("(()")


def test_mountain_distance_upper_scale() -> None:
    assert MODULE.mountain_distance_upper_scale(5) == 10


def test_spearman_correlation_uses_average_tied_ranks() -> None:
    first = np.asarray([0.0, 1.0, 1.0, 2.0])
    second = np.asarray([0.0, 1.0, 1.0, 2.0])

    assert MODULE.spearman_correlation(first, second) == pytest.approx(1.0)


def test_violation_bins_cover_each_nonnegative_count_once() -> None:
    rows = [
        {"violating_pair_count": count}
        for count in range(20)
    ]
    selected = []
    for _, lower, upper in MODULE.VIOLATION_BINS:
        selected.extend(MODULE.rows_in_violation_bin(rows, lower, upper))

    assert sorted(row["violating_pair_count"] for row in selected) == list(
        range(20)
    )


def test_deduplicate_rows_requires_consistent_references() -> None:
    rows = [
        {"id": "a", "sequence": "AAAA", "reference_structure": "...."},
        {"id": "b", "sequence": "AAAA", "reference_structure": "...."},
        {"id": "c", "sequence": "CCCC", "reference_structure": "...."},
        {"id": "d", "sequence": "CCCC", "reference_structure": "(())"},
    ]

    retained = MODULE.deduplicate_rows(rows)

    assert [row["id"] for row in retained] == ["a"]
