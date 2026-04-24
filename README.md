# ML_CFD

This repository contains the code developed for the master’s thesis in Nuclear Engineering: **PCA**, **manifold learning**, **Kernel PCA**, and **autoencoders** for dimensionality reduction of **computational fluid dynamics (CFD)** snapshot data.

The thesis manuscript is in [`docs/MLCFD_Thesis-MADRID-WAGNER.pdf`](docs/MLCFD_Thesis-MADRID-WAGNER.pdf).

The CFD data itself is not stored here for size reasons. If you need the datasets, contact [martin.madrid@ib.edu.ar](mailto:martin.madrid@ib.edu.ar). Expected CSV layout and mesh parameters are documented under [`data/README.md`](data/README.md).

## Install (uv)

```bash
uv sync --extra dev
```

Editable install and scripts use the `mlcfd` package under [`src/mlcfd/`](src/mlcfd/).

## Run a reduction pipeline (CLI)

`RunConfig` is validated with **Pydantic**; `model_params` must set `profile` to `sweep`, `manifold`, `kpca`, or `autoencoder` (see [`src/mlcfd/config/schemas.py`](src/mlcfd/config/schemas.py)).

```bash
mlcfd run runs/examples/pca_sklearn_re50.example.yaml --log-level INFO
```

- **Sweep models** (PCA, LLE, Isomap, KPCA): write `*_error_rec.csv` and, if `output.save_plots` is true, a reconstruction error plot.
- **Autoencoders** (`ae_spatial`, `ae_temporal`): write `*_metrics.json` after training.

More examples: [`runs/README.md`](runs/README.md).

## Pre-commit (optional)

```bash
pre-commit install
pre-commit run --all-files
```

Hooks run **Ruff** (`ruff` = lint/check with `--fix`, `ruff-format` = format), `nbstripout` on notebooks, and small file hygiene checks.

## Package layout (high level)

| Path | Role |
|------|------|
| `mlcfd.config` | Pydantic schemas and `RunConfig` |
| `mlcfd.mesh` | Cylinder mask and grid metadata |
| `mlcfd.io` | CSV I/O for snapshot matrices |
| `mlcfd.preprocessing` | Load / split / mask / standardize |
| `mlcfd.models` | PCA, manifolds, KPCA, autoencoders, factory |
| `mlcfd.visualization` | Modes and reconstruction error plots |
| `mlcfd.pipeline` | YAML-driven orchestration |
| `mlcfd.cli` | `mlcfd run` and future commands |
| `mlcfd.logging_config` | `configure_logging`, `get_logger("…")` |
| `mlcfd.thesis` | Drop-in helpers for migrated thesis notebooks (`thesis_mesh`, `read_X_csv`, etc.) |

**Logger names** (under the `mlcfd` namespace) include: `io`, `preprocessing`, `models`, `pipeline`, `visualization`, and `cli`.

## Notebooks

Exploratory work lives under [`notebooks/`](notebooks/); the CLI is the reproducible path. Use **nbstripout** so large outputs are not committed.

The thesis notebooks (PCA, manifold methods, autoencoders) are under [`notebooks/legacy/`](notebooks/legacy/); they import `mlcfd` instead of the old `dataprocess` package. See [`notebooks/legacy/README.md`](notebooks/legacy/README.md).
