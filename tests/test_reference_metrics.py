from __future__ import annotations

import pytest

from analysis.reference_metrics import (
    base_pair_f1,
    mean_squared_mountain_distance,
    mountain_heights,
    normalized_squared_mountain_distance,
    pairs_from_extended_dot_bracket,
    squared_mountain_distance,
)


def test_square_brackets_are_parsed_independently_from_round_brackets() -> None:
    assert pairs_from_extended_dot_bracket("([)]") == [(1, 3), (2, 4)]


def test_base_pair_f1_matches_endpoints_regardless_of_bracket_type() -> None:
    assert base_pair_f1("(.).", "[.].") == 1.0
    assert base_pair_f1("(.).", "([)]") == pytest.approx(2 / 3)


def test_mountain_heights_count_pairs_spanning_each_boundary() -> None:
    assert mountain_heights("([)]") == (1, 2, 1)
    assert squared_mountain_distance("....", "([)]") == 6.0
    assert mean_squared_mountain_distance("....", "([)]") == 2.0
    assert normalized_squared_mountain_distance("....", "([)]") == 1.0


@pytest.mark.parametrize("structure", ["([)", "(]", "<.>"])
def test_invalid_extended_dot_bracket_is_rejected(structure: str) -> None:
    with pytest.raises(ValueError):
        pairs_from_extended_dot_bracket(structure)
