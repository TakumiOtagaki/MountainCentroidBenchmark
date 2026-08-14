from __future__ import annotations

import math

import pytest

from analysis.summarize_hybrid_advisor_followup import (
    build_summary,
    validate_record_match,
)


def hybrid_row(record_id, f1, nmsmd, low_bpp, pair_count=1.0):
    return {
        "id": record_id,
        "sequence": "AAAA",
        "reference_structure": "....",
        "alpha": 0.25,
        "base_pair_f1": f1,
        "normalized_squared_mountain_distance": nmsmd,
        "fraction_selected_pair_bpp_below_0.01": low_bpp,
        "predicted_pair_count": pair_count,
    }


def gamma_row(record_id, f1, nmsmd):
    return {
        "id": record_id,
        "sequence": "AAAA",
        "reference_structure": "....",
        "base_pair_f1": f1,
        "normalized_squared_mountain_distance": nmsmd,
    }


def test_summary_separates_paired_categories_and_low_bpp_quantities():
    hybrid = [
        hybrid_row("hybrid", 0.7, 0.1, 0.2),
        hybrid_row("gamma", 0.3, 0.4, 0.0),
        hybrid_row("equal", 0.5, 0.2, math.nan, pair_count=0.0),
        hybrid_row("mixed", 0.7, 0.4, 0.1),
    ]
    gamma = {
        "hybrid": gamma_row("hybrid", 0.6, 0.2),
        "gamma": gamma_row("gamma", 0.4, 0.3),
        "equal": gamma_row("equal", 0.5, 0.2),
        "mixed": gamma_row("mixed", 0.6, 0.3),
    }

    result = build_summary(hybrid, (0.25,), gamma)[0]

    assert result["median_fraction_selected_pair_bpp_below_0.01"] == pytest.approx(0.05)
    assert result["fraction_records_with_any_selected_pair_bpp_below_0.01"] == 0.5
    assert result["fraction_f1_at_least_and_nmsmd_at_most_gamma2"] == 0.5
    assert result["fraction_hybrid_pareto_dominates_gamma2"] == 0.25
    assert result["fraction_gamma2_pareto_dominates_hybrid"] == 0.25
    assert result["fraction_equal_to_gamma2"] == 0.25
    assert result["fraction_mixed_vs_gamma2"] == 0.25


def test_record_match_requires_the_same_annotations():
    hybrid = [hybrid_row("record", 0.5, 0.2, 0.0)]
    gamma = {"record": gamma_row("record", 0.5, 0.2)}
    gamma["record"]["reference_structure"] = "(())"

    with pytest.raises(ValueError, match="records differ"):
        validate_record_match(hybrid, gamma)
