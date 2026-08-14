# Reproducing the benchmark

Run all commands from the repository root after completing the setup in the
README. The primary analysis used ViennaRNA 2.7.2 at 37 degrees Celsius and
retained sequences of at most 300 nt.

## Four-method benchmark

```sh
PY=software/MountainCentroid/.venv/bin/python

$PY analysis/evaluate_baselines_and_relaxed.py \
  --dataset data/rnastralign.csv --bpp-backend vienna --cpus 8 \
  --output results/raw/baselines.csv

$PY analysis/evaluate_sequence_constrained.py \
  --dataset data/rnastralign.csv \
  --baseline-metrics results/raw/baselines.csv \
  --output-dir results/raw/sequence_constrained --workers 8

$PY analysis/merge_sequence_constrained_metrics.py \
  --baselines results/raw/baselines.csv \
  --sequence-constrained results/raw/sequence_constrained/metrics.csv \
  --output results/raw/four_method_metrics.csv
```

The summarization and plotting programs under `analysis/` accept explicit
input and output paths. Their versioned outputs are under
`results/structure_prediction/` and `figures/`.

## Gamma-centroid and selected-pair BPP analyses

```sh
$PY analysis/evaluate_gamma_centroid.py \
  --dataset data/rnastralign.csv \
  --baseline-metrics results/raw/four_method_metrics.csv \
  --output-dir results/raw/gamma_centroid --workers 8

$PY analysis/evaluate_selected_pair_bpp.py \
  --baseline-metrics results/raw/four_method_metrics.csv \
  --gamma-metrics results/raw/gamma_centroid/metrics.csv \
  --output-dir results/raw/selected_pair_bpp --workers 8
```

## Hybrid objective

The full analysis used the 13 alpha values defined in
`analysis/evaluate_hybrid_objective_pilot.py`. Compile the analysis solver and
run locally as follows, or use `scripts/job_scripts/tsubame_hybrid_objective_full.sh`
on TSUBAME.

```sh
mkdir -p build
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
  analysis/hybrid_constrained_solver.cpp -o build/hybrid_constrained_solver

PYTHONPATH=.:software/MountainCentroid/src \
$PY -m analysis.evaluate_hybrid_objective_pilot \
  --baseline-metrics results/raw/four_method_metrics.csv \
  --output-dir results/raw/hybrid/full_alpha_grid \
  --hybrid-binary build/hybrid_constrained_solver \
  --all-records --workers 8

PYTHONPATH=.:software/MountainCentroid/src \
$PY -m analysis.summarize_hybrid_objective_full \
  --metrics results/raw/hybrid/full_alpha_grid/metrics.csv \
  --output-dir results/raw/hybrid/full_alpha_grid/full_analysis
```

Expected record counts and SHA-256 values are listed in
`provenance/README.md`.
