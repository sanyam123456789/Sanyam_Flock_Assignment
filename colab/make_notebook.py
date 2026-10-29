"""Turn seq2point_refit.py (cells marked with '# %%') into seq2point_refit.ipynb."""
import json
from pathlib import Path

here = Path(__file__).parent
src = (here / "seq2point_refit.py").read_text(encoding="utf-8")
cells = []
for block in src.split("# %%")[1:]:
    head, _, body = block.partition("\n")
    if head.strip() == "[markdown]":
        lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip("\n").splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.strip("\n")})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
(here / "seq2point_refit.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {len(cells)} cells")
