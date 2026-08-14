#!/usr/bin/env python3
"""Download the RNAStrAlign table used by the benchmark and verify it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


EXPECTED_ROWS = 37149
EXPECTED_SHA256 = "ac45ba81e0cc7007fc8b07b2953b0c8a941d25f926bb9ac6fe737b51cb9c1eb3"
REQUIRED_COLUMNS = ("id", "sequence", "secondary_structure", "family", "subfamily")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rnastralign.csv"),
    )
    args = parser.parse_args()

    parquet = hf_hub_download(
        repo_id="multimolecule/rnastralign",
        repo_type="dataset",
        filename="train.parquet",
    )
    frame = pd.read_parquet(parquet)
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"unexpected columns: {tuple(frame.columns)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"unexpected row count: {len(frame)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    observed = sha256(args.output)
    if observed != EXPECTED_SHA256:
        raise ValueError(
            f"CSV checksum mismatch: expected {EXPECTED_SHA256}, observed {observed}"
        )
    print(f"Wrote {len(frame)} records to {args.output} ({observed})")


if __name__ == "__main__":
    main()
