# Legacy `dataprocess` module

The Python modules in this directory (`dataprocess.py`, `domain_crop.py`) are the **original thesis-era helpers** used by the top-level method notebooks (PCA, LLE, autoencoders, etc.).

The maintained implementation lives in the **`mlcfd`** package under [`src/mlcfd/`](../src/mlcfd/): configuration validation, I/O, preprocessing, models, `mlcfd run`, and structured logging.

For new work:

- Prefer `from mlcfd...` imports and a YAML / CLI run.
- Keep using this folder only if you are reproducing a historical notebook one-to-one or until notebooks are updated.

The thesis Jupyter notebooks now live under [`notebooks/legacy/`](../notebooks/legacy/) and import `mlcfd` instead of this package.

When nothing else references this folder, it may be removed in favor of `mlcfd` only.
