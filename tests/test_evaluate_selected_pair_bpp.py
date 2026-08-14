from __future__ import annotations

import pytest

from analysis.evaluate_selected_pair_bpp import (
    _fraction_below,
    selected_pair_probabilities,
)


def test_selected_pair_probabilities_uses_one_based_bpp_matrix() -> None:
    bpp = [[0.0] * 7 for _ in range(7)]
    bpp[1][6] = 0.2
    bpp[2][5] = 0.8

    assert selected_pair_probabilities(bpp, "((..))") == [0.2, 0.8]


def test_fraction_below_uses_strict_threshold() -> None:
    assert _fraction_below([0.01, 0.009, 0.2], 0.01) == pytest.approx(1 / 3)
