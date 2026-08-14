from __future__ import annotations

import csv
from pathlib import Path

from analysis.recalculate_reference_metrics import recalculate


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_recalculate_uses_preserved_square_brackets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    source = tmp_path / "metrics.csv"
    output = tmp_path / "updated.csv"
    write_csv(
        dataset,
        ["id", "secondary_structure"],
        [{"id": "crossing", "secondary_structure": "([)]"}],
    )
    fields = [
        "id",
        "reference_structure",
        "predicted_structure",
        "base_pair_f1",
        "normalized_squared_mountain_distance",
    ]
    write_csv(
        source,
        fields,
        [
            {
                "id": "crossing",
                "reference_structure": "(())",
                "predicted_structure": "(.).",
                "base_pair_f1": "0",
                "normalized_squared_mountain_distance": "0",
            }
        ],
    )

    assert recalculate(dataset, source, output) == (1, 1)
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["reference_structure"] == "([)]"
    assert float(row["base_pair_f1"]) == 2 / 3
    assert float(row["normalized_squared_mountain_distance"]) == 1 / 3


def test_recalculate_updates_relaxed_and_constrained_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    source = tmp_path / "metrics.csv"
    output = tmp_path / "updated.csv"
    write_csv(
        dataset,
        ["id", "secondary_structure"],
        [{"id": "crossing", "secondary_structure": "([)]"}],
    )
    fields = [
        "id",
        "reference_structure",
        "relaxed_structure",
        "constrained_structure",
        "relaxed_base_pair_f1",
        "constrained_base_pair_f1",
    ]
    write_csv(
        source,
        fields,
        [
            {
                "id": "crossing",
                "reference_structure": "(())",
                "relaxed_structure": "....",
                "constrained_structure": "(.).",
                "relaxed_base_pair_f1": "1",
                "constrained_base_pair_f1": "0",
            }
        ],
    )

    assert recalculate(dataset, source, output) == (1, 1)
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["relaxed_base_pair_f1"]) == 0
    assert float(row["constrained_base_pair_f1"]) == 2 / 3
