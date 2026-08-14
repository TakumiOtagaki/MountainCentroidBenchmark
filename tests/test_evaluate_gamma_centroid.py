from __future__ import annotations

import RNA

from analysis.evaluate_gamma_centroid import (
    gamma_centroid_structure,
    validate_prediction,
)


def test_gamma_changes_positive_pair_gain_threshold() -> None:
    bpp = [[0.0] * 5 for _ in range(5)]
    bpp[0][4] = 0.4

    assert gamma_centroid_structure(bpp, 1.0) == "....."
    assert gamma_centroid_structure(bpp, 2.0) == "(...)"


def test_gamma_one_matches_vienna_centroid() -> None:
    sequence = "GGGAAACCC"
    fold_compound = RNA.fold_compound(sequence)
    _, mfe_energy = fold_compound.mfe()
    fold_compound.exp_params_rescale(mfe_energy)
    fold_compound.pf()
    vienna_bpp = fold_compound.bpp()
    bpp = [row[1:] for row in vienna_bpp[1:]]

    prediction = gamma_centroid_structure(bpp, 1.0)

    assert prediction == fold_compound.centroid()[0]
    assert validate_prediction(sequence, prediction) == 3
