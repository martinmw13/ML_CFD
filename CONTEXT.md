# Context

Domain language for `mlcfd`, a package for dimensionality reduction of CFD
snapshot data. Use these terms (not synonyms) in issues, tests, and code.

## Glossary

### Snapshot matrix

A 2-D array of CFD snapshots. The **canonical** layout used throughout the
pipeline is `(n_free, n_snapshots)` — spatial points (free, i.e. cylinder-masked,
degrees of freedom) down the rows, snapshots across the columns. This is
**snapshots-as-columns** (see *snapshot orientation*).

### Snapshot orientation

The single answer to "is a snapshot a row or a column?", modelled by
`SnapshotOrientation` in `src/mlcfd/orientation.py` and applied at each consumer's
edge by `orient()`:

- **snapshots-as-columns** (`SNAPSHOTS_AS_COLUMNS`) — the canonical layout;
  `orient()` returns the matrix unchanged.
- **snapshots-as-rows** (`SNAPSHOTS_AS_ROWS`) — each snapshot is a row/sample (the
  scikit-learn convention); `orient()` returns the transpose.

`orient()` is its own inverse, so a consumer that orients on the way in and back
out round-trips to the canonical layout.

This replaces three earlier, independently-named mechanisms — `spatial_reduction`
(scaler), `transpose_flag`/`sklearn_layout` (manifold + Kernel PCA), and the
autoencoder `layout` — and removes the old "transpose when the flag is `False`"
inversion. Every prior default mapped to `SNAPSHOTS_AS_ROWS`, which stays the
default for the scaler, manifold, and KPCA configs; `ae_spatial` is
snapshots-as-rows and `ae_temporal` is snapshots-as-columns. These transpose
decisions are load-bearing for parity with the thesis notebooks and are pinned by
`tests/test_orientation_parity.py`.
