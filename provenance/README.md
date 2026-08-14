# Benchmark provenance

## Inputs and software

- RNAStrAlign CSV: 37,149 source records; SHA-256
  `ac45ba81e0cc7007fc8b07b2953b0c8a941d25f926bb9ac6fe737b51cb9c1eb3`
- Analysis set: 21,254 records of at most 300 nt
- Deduplicated sensitivity set: 16,222 records
- MountainCentroid revision:
  `3557741fa39ce536755ea03a54dedc4310243555`
- R4RNA revision used for the case-study figure:
  `addfd23591461a45c086b6e9948c830cc84881a3`
- ViennaRNA 2.7.2; temperature 37 degrees Celsius

## Full runs

The four-method benchmark contains 21,254 complete cases for ViennaRNA MFE,
ViennaRNA centroid, the mountain-path relaxation, and Mountain Centroid. The
verified merged metrics SHA-256 is
`15399e6a2621c938c60d7a77d33bc303a82bc1fb2c77653018cd85ee65d2efbb`.

The hybrid run evaluated 13 alpha values for all 21,254 records, producing
276,302 predictions. The alpha-0 and alpha-1 structures matched the Mountain
Centroid and base-pair centroid endpoints for every record. The raw hybrid
metrics SHA-256 is
`74fb1e6505d98979d860c04e0ca793113a6b495fde0a5dbd6f61015b5eafb8ca`.
TSUBAME Grid Engine job 8397933 completed with exit status zero using 32
workers, Python 3.12.13, NumPy 2.0.2, and ViennaRNA 2.7.2.

The raw gamma-centroid metrics SHA-256 is
`801df3d050502cba376eea59a3e66b86fa7e77e6374e0c598e5fcec6b7d56227`.

These metric files preserve round and square reference brackets and compare
base pairs by their endpoints. Cached predictions were retained, and reference-based
metrics were recalculated with `analysis/recalculate_reference_metrics.py`.

## Versioned outputs

`checksums.sha256` covers all CSV, PDF, and PNG artifacts included in this
repository. Full raw metrics and resumable per-record JSON files are omitted
from Git because they can be regenerated with the documented commands.
