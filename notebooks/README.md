# Exploratory notebooks

These notebooks are for **interactive experimentation** (parameter tweaks, plots, debugging). Reproducible, scriptable runs should use the CLI:

```bash
mlcfd run runs/examples/pca_sklearn_re50.example.yaml
```

- Prefer **date-prefixed** notebook names, e.g. `2026-04-23_topic_exploration.ipynb`.
- Use **nbstripout** (see repo pre-commit config) to avoid committing large cell outputs.
- The template notebook shows how to call `run_from_config` with a validated `RunConfig`—uncomment the last cell only when your `data/` CSV is present.
