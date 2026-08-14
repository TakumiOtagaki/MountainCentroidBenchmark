# Benchmark input

The benchmark uses the `train.parquet` table from the
[MultiMolecule RNAStrAlign dataset](https://huggingface.co/datasets/multimolecule/rnastralign),
which cites Tan et al., *Nucleic Acids Research* 45:11570--11581 (2017),
doi:10.1093/nar/gkx815. The dataset card reports the data license as
AGPL-3.0-or-later.

Run `scripts/prepare_rnastralign.py` to download the table, preserve its
extended dot-bracket reference structures, write the CSV, and verify its
SHA-256. The evaluation scripts then retain canonical RNA sequences with
reference structures containing round or square brackets and length at most
300 nt, yielding 21,254 benchmark records. Round and square brackets are
parsed independently, and base pairs are compared by their endpoints.

The downloaded CSV is intentionally excluded from this repository.
