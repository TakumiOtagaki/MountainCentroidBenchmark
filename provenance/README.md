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
`30cc2408d0d38cde651eb011fc8a5e319fe43cc9b09b5b96b1125cf8949e5b48`.

The hybrid run evaluated 13 alpha values for all 21,254 records, producing
276,302 predictions. The alpha-0 and alpha-1 structures matched the Mountain
Centroid and base-pair centroid endpoints for every record. The raw hybrid
metrics SHA-256 is
`7c405a3d67d87f2c6b3bcf4331445efe5a96ea4c170d17652563839d9ad51295`.
TSUBAME Grid Engine job 8397933 completed with exit status zero using 32
workers, Python 3.12.13, NumPy 2.0.2, and ViennaRNA 2.7.2.

The raw gamma-centroid metrics SHA-256 is
`eb4416d2df2dc169efb3a2713d0373c9f44166654b7bfd2bb72b4b0f2b6b39db`.

## Versioned outputs

`checksums.sha256` covers all CSV, PDF, and PNG artifacts included in this
repository. Full raw metrics and resumable per-record JSON files are omitted
from Git because they can be regenerated with the documented commands.
