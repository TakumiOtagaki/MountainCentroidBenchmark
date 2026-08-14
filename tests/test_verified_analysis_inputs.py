from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "required_option"),
    (
        ("evaluate_gamma_centroid.py", "--baseline-metrics"),
        ("plot_gamma_centroid_sensitivity.py", "--baseline-metrics"),
        ("plot_structure_prediction_with_gamma.py", "--input"),
        ("evaluate_selected_pair_bpp.py", "--baseline-metrics"),
        ("plot_selected_pair_bpp.py", "--input-dir"),
    ),
)
def test_verified_analysis_input_is_required(
    script: str, required_option: str
) -> None:
    tree = ast.parse((Path("analysis") / script).read_text())
    keywords = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = node.args[0]
        if isinstance(option, ast.Constant) and option.value == required_option:
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            break

    assert keywords is not None
    assert isinstance(keywords.get("required"), ast.Constant)
    assert keywords["required"].value is True
    assert "default" not in keywords
