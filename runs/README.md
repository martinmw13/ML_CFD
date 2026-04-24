# Run configurations

This folder holds example **YAML** files consumed by the CLI:

```bash
mlcfd run runs/examples/pca_sklearn_re50.example.yaml
```

Copy an `.example.yaml` to a new file, adjust paths, and point `--log-level` as needed. Paths in YAML are usually **relative to the current working directory** when you launch `mlcfd`.

Required keys match [`RunConfig`](../src/mlcfd/config/schemas.py): `mesh`, `data`, `output`, `model_name`, and `model_params` (the latter must include a `profile` field for Pydantic’s discriminated union: `sweep`, `manifold`, `kpca`, or `autoencoder`).
