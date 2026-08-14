#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -o results/raw/hybrid
#$ -l cpu_40=1
#$ -l h_rt=04:00:00
#$ -N mc_hybrid_full

set -euo pipefail

REPO_ROOT="${SGE_O_WORKDIR:-$PWD}"
BASELINE="$REPO_ROOT/results/raw/four_method_metrics.csv"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/raw/hybrid/full_alpha_grid}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/software/MountainCentroid/.venv/bin/python}"
HYBRID_BINARY="$REPO_ROOT/build/hybrid_constrained_solver"
WORKERS="${WORKERS:-32}"
EXPECTED_INPUT_SHA256="30cc2408d0d38cde651eb011fc8a5e319fe43cc9b09b5b96b1125cf8949e5b48"
EXPECTED_RECORDS=21254
EXPECTED_ALPHAS=13
EXPECTED_ROWS=276302

cd "$REPO_ROOT"
[[ -f analysis/hybrid_constrained_solver.cpp ]] || {
  echo "ERROR: submit this job from the MountainCentroidBenchmark repository root" >&2
  exit 1
}
[[ -f "$BASELINE" ]] || { echo "ERROR: baseline metrics missing: $BASELINE" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python environment missing: $PYTHON_BIN" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "ERROR: tracked worktree changes would make the run provenance ambiguous" >&2
  exit 1
}
if [[ -n "${NSLOTS:-}" && "$WORKERS" -gt "$NSLOTS" ]]; then
  echo "ERROR: WORKERS=$WORKERS exceeds NSLOTS=$NSLOTS" >&2
  exit 1
fi

observed_input_sha256="$(sha256sum "$BASELINE" | awk '{print $1}')"
[[ "$observed_input_sha256" == "$EXPECTED_INPUT_SHA256" ]] || {
  echo "ERROR: baseline SHA-256 mismatch: $observed_input_sha256" >&2
  exit 1
}

"$PYTHON_BIN" -c '
import sys
import numpy
import RNA
assert sys.version_info >= (3, 10), sys.version
assert numpy.__version__ == "2.0.2", numpy.__version__
assert RNA.__version__ == "2.7.2", RNA.__version__
print(f"Python {sys.version.split()[0]}, NumPy {numpy.__version__}, ViennaRNA {RNA.__version__}")
'

mkdir -p build "$OUTPUT_DIR"
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
  analysis/hybrid_constrained_solver.cpp \
  -o "$HYBRID_BINARY"

export PYTHONPATH=".:software/MountainCentroid/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-not_scheduled}"
echo "NSLOTS: ${NSLOTS:-not_scheduled}"
echo "Revision: $(git rev-parse HEAD)"
echo "Workers: $WORKERS"

"$PYTHON_BIN" -m analysis.evaluate_hybrid_objective_pilot \
  --baseline-metrics "$BASELINE" \
  --output-dir "$OUTPUT_DIR" \
  --hybrid-binary "$HYBRID_BINARY" \
  --all-records \
  --workers "$WORKERS" \
  --log "$OUTPUT_DIR/evaluator.log"

"$PYTHON_BIN" -m analysis.summarize_hybrid_objective_pilot \
  --metrics "$OUTPUT_DIR/metrics.csv" \
  --output-dir "$OUTPUT_DIR/summary"

"$PYTHON_BIN" - "$OUTPUT_DIR" "$EXPECTED_RECORDS" "$EXPECTED_ALPHAS" "$EXPECTED_ROWS" <<'PY'
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
expected_records, expected_alphas, expected_rows = map(int, sys.argv[2:])
with (output_dir / "metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
with (output_dir / "run_manifest.json").open() as handle:
    manifest = json.load(handle)

ids = {row["id"] for row in rows}
alphas = {row["alpha"] for row in rows}
grids = defaultdict(set)
endpoint_matches = Counter()
for row in rows:
    grids[row["id"]].add(row["alpha"])
    if row["alpha"] in {"0.0", "1.0"}:
        endpoint_matches[row["alpha"]] += row["endpoint_structure_match"] == "True"

assert len(rows) == expected_rows, len(rows)
assert len(ids) == expected_records, len(ids)
assert len(alphas) == expected_alphas, len(alphas)
assert all(len(grid) == expected_alphas for grid in grids.values())
assert endpoint_matches["0.0"] == expected_records, endpoint_matches
assert endpoint_matches["1.0"] == expected_records, endpoint_matches
assert manifest["selected_cases"] == expected_records, manifest["selected_cases"]
assert manifest["sampling"] == "all_records", manifest["sampling"]
assert manifest["per_family"] is None, manifest["per_family"]
assert manifest["baseline_metrics_sha256"] == "30cc2408d0d38cde651eb011fc8a5e319fe43cc9b09b5b96b1125cf8949e5b48"
assert manifest["analysis_worktree_clean"] is True
assert manifest["software_worktree_clean"] is True
print(
    f"Validated {len(ids)} records, {len(alphas)} alpha values, "
    f"{len(rows)} predictions, and both endpoints"
)
PY

sha256sum \
  "$OUTPUT_DIR/metrics.csv" \
  "$OUTPUT_DIR/run_manifest.json" \
  "$OUTPUT_DIR/evaluator.log" \
  "$OUTPUT_DIR/summary/summary.csv" \
  "$OUTPUT_DIR/summary/paired_summary.csv" \
  > "$OUTPUT_DIR/sha256sums.txt"

echo "Finished: $(date --iso-8601=seconds)"
echo "Outputs: $OUTPUT_DIR"
