# Legacy method notebooks (thesis)

These are the **original per-method Jupyter workflows** (PCA/SVD, PCA sklearn, LLE, Isomap, KPCA, spatial/temporal autoencoders), moved from the top-level `AE/`, `PCA_SVD/`, etc. folders.

## What changed

- **Imports:** legacy `dataprocess` path hacks are replaced by `mlcfd.thesis` and `erase_cylinder_rows` (see the first code cell in each notebook).
- **Data paths:** `../data/` was updated to **`../../data/`** because these notebooks now live under `notebooks/legacy/`.
- **Reproducible runs:** for non-interactive execution, prefer the CLI and YAML under [`runs/examples/`](../../runs/examples/):

  ```bash
  mlcfd run runs/examples/pca_sklearn_re50.example.yaml
  ```

Keep using this directory for exploration, plots, and parameter sweeps that are not yet encoded in YAML.
