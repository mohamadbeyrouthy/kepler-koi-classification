"""
make_final_notebook.py  (Member A: Hend, plan step P8)
======================================================
Merges every part into ONE notebook, in the order of the plan, for the Moodle submission:

    P1  notebooks/P1_data_split.ipynb            data audit, cleaning, split by star   (Hend)
    P2+P3 notebooks/B_models.ipynb               features, leakage, Level 1 models     (Mohamad)
    P4  notebooks/P4_neural_network.ipynb        Level 2 neural network                (Hend)
    P5  notebooks/B_comparison_test.ipynb        comparison, one-time test, splits     (Mohamad)
    P6  notebooks/P6_failure_analysis.ipynb      failure analysis, confidence          (Hend)
    P7  notebooks/B_interpretability_final.ipynb interpretability, complexity, model   (Mohamad)

Output: notebooks/final_notebook.ipynb (outputs cleared). Then open it, Run All, Ctrl+S, push.

Usage (from the repository root):
    python make_final_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT = NB_DIR / "final_notebook.ipynb"

PARTS = [
    ("P1", "Data audit, cleaning and split by star", "Hend", "P1_data_split.ipynb"),
    ("P2 + P3", "Features, leakage check and Level 1 models", "Mohamad", "B_models.ipynb"),
    ("P4", "Level 2: neural network", "Hend", "P4_neural_network.ipynb"),
    ("P5", "Model comparison, one-time test and split sensitivity", "Mohamad", "B_comparison_test.ipynb"),
    ("P6", "Failure analysis and trust in confidence", "Hend", "P6_failure_analysis.ipynb"),
    ("P7", "Interpretability, complexity and the final model", "Mohamad", "B_interpretability_final.ipynb"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def clean(cell):
    """Copy a cell without its outputs so the merged notebook is run fresh, top to bottom."""
    c = {"cell_type": cell["cell_type"], "metadata": {}, "source": cell["source"]}
    if cell["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    return c


missing = [f for *_, f in PARTS if not (NB_DIR / f).exists()]
if missing:
    raise SystemExit(f"Missing notebooks: {missing}. Run `git pull` first.")

toc = "\n".join(f"{i}. **{p}**: {title} ({who})" for i, (p, title, who, _) in enumerate(PARTS, 1))
cells = [md(
    "# Exoplanet candidate classification (NASA Kepler KOI)\n"
    "**CMPS 261 · Project 4.1 · Hend Albakour and Mohammad Hashem Al Beyrouthy**\n\n"
    "Can machine learning tell real exoplanets from false positives using only measurements available "
    "**before** scientists vetted them? This notebook contains all the code, in the order of the project plan:\n\n"
    f"{toc}\n\n"
    "Run it from the `notebooks/` folder of the repository (Run All). Every part reads its inputs from the "
    "repository (`data/`, `features.py`, `results/`, `models/`), so the parts can also be run on their own. "
    "`random_state = 42` is used everywhere."
)]

for part, title, who, fname in PARTS:
    nb = json.loads((NB_DIR / fname).read_text(encoding="utf-8"))
    cells.append(md(f"---\n# {part} · {title}\n*Owner: {who} · source notebook: `notebooks/{fname}`*"))
    cells.extend(clean(c) for c in nb["cells"])

merged = {
    "cells": cells,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(merged["cells"]):          # unique cell ids (nbformat 4.5)
    c["id"] = f"cell-{i:03d}"

OUT.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(c["cell_type"] == "code" for c in cells)
print(f"Wrote {OUT.relative_to(ROOT)}: {len(cells)} cells ({n_code} code) from {len(PARTS)} notebooks.")
print("Next: open it, Run All (about 10-15 minutes), Ctrl+S, then push.")
