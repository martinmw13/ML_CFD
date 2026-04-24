#!/usr/bin/env python3
"""Patch moved thesis notebooks: replace ``dataprocess`` usage with ``mlcfd.thesis`` imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MIGRATION_MARK = "mlcfd: thesis notebook imports (replaces ``dataprocess``)"

IMPORTS_CELL = f"""# {MIGRATION_MARK}
from mlcfd.preprocessing.pipeline import erase_cylinder_rows as erase_cyl
from mlcfd.thesis import (
    plot_save_reconst,
    read_X_csv,
    save_thesis_modes,
    thesis_mesh,
)
"""


def _fix_data_path_line(line: str) -> str:
    """From ``notebooks/legacy/`` one more ``..`` is needed to reach project ``data/``."""
    line = line.replace('"../data/', '"../../data/')
    line = line.replace("'../data/", "'../../data/")
    line = line.replace("(\"../data/", "(\"../../data/")
    return line


def _patch_source(sources: list[str]) -> list[str]:
    new_lines: list[str] = []
    for line in sources:
        if "sys.path.append" in line and "dataprocess" in line:
            continue
        if "import dataprocess as dp" in line:
            continue
        line = line.replace("dp.Mesh(", "thesis_mesh(")
        line = line.replace("dp.read_X_csv", "read_X_csv")
        line = line.replace("dp.erase_cyl", "erase_cyl")
        line = line.replace("dp.save_modes", "save_thesis_modes")
        line = line.replace("dp.plot_save_reconst", "plot_save_reconst")
        line = _fix_data_path_line(line)
        new_lines.append(line)
    return new_lines


def _strip_stale_sys(sources: list[str]) -> list[str]:
    joined = "".join(sources)
    if "sys." in joined:
        return sources
    return [ln for ln in sources if not (ln.strip() == "import sys")]


def patch_notebook(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(MIGRATION_MARK in "".join(c.get("source", [])) for c in data.get("cells", [])):
        return
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]
        source = _strip_stale_sys(_patch_source(list(source)))
        cell["source"] = source
        cell["outputs"] = []
        cell["execution_count"] = None
    first_code_idx = next(
        (i for i, c in enumerate(data["cells"]) if c.get("cell_type") == "code"),
        0,
    )
    new_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in IMPORTS_CELL.strip().split("\n")],
    }
    data["cells"].insert(first_code_idx, new_cell)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(paths: list[Path]) -> None:
    for p in paths:
        patch_notebook(p)
        print("patched", p)


if __name__ == "__main__":
    main([Path(s).resolve() for s in sys.argv[1:]])
