# Mountain Centroid benchmark

This repository contains the analysis code, summary results, and figures for
the RNAStrAlign evaluation of Mountain Centroid. The reusable prediction
software is maintained separately in
[`MountainCentroid`](https://github.com/TakumiOtagaki/MountainCentroid) and is
pinned here as a submodule at the revision used for the reported benchmark.

## Contents

- `analysis/`: evaluation, summarization, and plotting programs
- `results/`: versioned summary results; full per-sequence metrics are ignored
- `figures/`: figures generated for the manuscript and Supplementary Information
- `provenance/`: revisions, checksums, software versions, and run information
- `scripts/`: dataset preparation and the full TSUBAME hybrid job

## Setup

Clone recursively and create the locked MountainCentroid environment:

```sh
git clone --recursive https://github.com/TakumiOtagaki/MountainCentroidBenchmark.git
cd MountainCentroidBenchmark
uv sync --project software/MountainCentroid --frozen --extra analysis --extra test
make -C software/MountainCentroid constrained
```

Prepare the external RNAStrAlign input:

```sh
uv run --project software/MountainCentroid \
  --with pandas --with pyarrow --with huggingface-hub \
  python scripts/prepare_rnastralign.py
```

See [REPRODUCING.md](REPRODUCING.md) for the benchmark commands. The checked-in
summary files and their hashes are described in
[provenance/README.md](provenance/README.md).

## Scope

The repository does not contain manuscript drafts, reviewer feedback, internal
notes, or the full raw per-sequence outputs. Those outputs can be regenerated
from the external dataset with the provided commands.
