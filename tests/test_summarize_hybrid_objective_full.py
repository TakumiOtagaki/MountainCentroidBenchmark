from __future__ import annotations

import pytest

from analysis.summarize_hybrid_objective_full import (
    build_summaries,
    deduplicated_record_ids,
)


def row(
    record_id: str,
    family: str,
    sequence: str,
    reference: str,
    alpha: float,
    f1: float,
    nmsmd: float,
) -> dict[str, float | str]:
    return {
        "id": record_id,
        "family": family,
        "sequence": sequence,
        "reference_structure": reference,
        "alpha": alpha,
        "predicted_structure": "...." if alpha == 0.0 else "(..)",
        "normalized_mountain_objective": alpha,
        "normalized_centroid_gain": alpha,
        "base_pair_f1": f1,
        "normalized_squared_mountain_distance": nmsmd,
        "predicted_pair_count": alpha,
        "median_selected_pair_bpp": alpha,
        "fraction_selected_pair_bpp_below_0.01": alpha,
        "hybrid_seconds": alpha,
    }


def test_deduplicated_record_ids_requires_consistent_references() -> None:
    rows = [
        row("a", "family_a", "AAAA", "....", 0.0, 0.1, 0.2),
        row("a", "family_a", "AAAA", "....", 1.0, 0.2, 0.3),
        row("b", "family_a", "AAAA", "....", 0.0, 0.1, 0.2),
        row("b", "family_a", "AAAA", "....", 1.0, 0.2, 0.3),
        row("c", "family_b", "CCCC", "....", 0.0, 0.1, 0.2),
        row("c", "family_b", "CCCC", "....", 1.0, 0.2, 0.3),
        row("d", "family_b", "CCCC", "(())", 0.0, 0.1, 0.2),
        row("d", "family_b", "CCCC", "(())", 1.0, 0.2, 0.3),
        row("e", "family_b", "GGGG", "....", 0.0, 0.1, 0.2),
        row("e", "family_b", "GGGG", "....", 1.0, 0.2, 0.3),
    ]

    assert deduplicated_record_ids(rows) == {"a", "e"}


def test_build_summaries_includes_scopes_families_and_paired_differences() -> None:
    rows = [
        row("a", "family_a", "AAAA", "....", 0.0, 0.2, 0.2),
        row("a", "family_a", "AAAA", "....", 1.0, 0.5, 0.3),
        row("b", "family_b", "CCCC", "....", 0.0, 0.4, 0.4),
        row("b", "family_b", "CCCC", "....", 1.0, 0.6, 0.2),
    ]

    distributions, paired, metadata = build_summaries(rows, (0.0, 1.0))

    assert metadata["record_counts"] == {"all": 2, "deduplicated": 2}
    assert metadata["family_counts"]["all"] == {
        "all_families": 2,
        "family_a": 1,
        "family_b": 1,
    }
    assert len(distributions) == 12
    assert distributions[0][
        "fraction_records_with_any_selected_pair_bpp_below_0.01"
    ] == 0.0
    comparison = next(
        item
        for item in paired
        if item["dataset_scope"] == "all"
        and item["group"] == "all_families"
        and item["alpha"] == 1.0
        and item["endpoint_alpha"] == 0.0
    )
    assert comparison["median_delta_base_pair_f1"] == pytest.approx(0.25)
    assert comparison["fraction_higher_base_pair_f1"] == 1.0
    assert comparison["median_delta_nmsmd"] == pytest.approx(-0.05)
