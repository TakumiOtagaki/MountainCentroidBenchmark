from __future__ import annotations

from analysis.evaluate_hybrid_objective_pilot import select_records


def record(record_id: str, family: str, length: int) -> dict[str, str]:
    return {"id": record_id, "family": family, "sequence": "A" * length}


def test_select_records_retains_stratified_pilot_mode() -> None:
    records = [
        record("a-short", "a", 10),
        record("a-middle", "a", 20),
        record("a-long", "a", 30),
        record("b-only", "b", 15),
    ]

    selected, sampling = select_records(records, per_family=2, all_records=False)

    assert [item["id"] for item in selected] == ["a-long", "a-short", "b-only"]
    assert sampling == "length_ordered_even_spacing_within_family"


def test_select_records_retains_all_records_without_subsampling() -> None:
    records = [
        record("a-short", "a", 10),
        record("a-middle", "a", 20),
        record("a-long", "a", 30),
    ]

    selected, sampling = select_records(records, per_family=2, all_records=True)

    assert selected == records
    assert selected is not records
    assert sampling == "all_records"
