from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "summarize_sequence_constraint_cost.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_sequence_constraint_cost", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_objective_gap_roundoff_is_clamped_to_zero() -> None:
    rows = [
        {"objective_gap": "-1e-14", "constrained_objective": "10.0"},
        {"objective_gap": "0.25", "constrained_objective": "10.0"},
    ]

    values = MODULE.objective_gaps_with_roundoff_clamped(rows)

    np.testing.assert_array_equal(values, np.asarray([0.0, 0.25]))


def test_objective_gap_violation_is_rejected() -> None:
    rows = [{"objective_gap": "-1e-3", "constrained_objective": "10.0"}]

    with pytest.raises(ValueError, match="violated"):
        MODULE.objective_gaps_with_roundoff_clamped(rows)
