"""Export DEVELOPMENT-house data for the Seq2Point notebook on Colab (GPU).

One file per development house: colab/data/house{N}.parquet with the 1-minute aggregate
(NaN where the aggregate is invalid - exactly the input LightGBM sees), the target, and the
same `evaluable` / `train_row` flags used by the LightGBM folds. Test houses are never
exported here.
"""
from __future__ import annotations

from src import features as F
from src.config import CFG, ROOT

OUT = ROOT / "colab" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev, test = CFG["dev_houses"], set(CFG["test_houses"])
    assert not (set(dev) & test)
    total = 0
    for h in dev:
        d = F.load(h, columns=["x", "y", "evaluable", "train_row", "house"]).rename(columns={"x": "agg"})
        p = OUT / f"house{h}.parquet"
        d.to_parquet(p, compression="zstd")
        total += p.stat().st_size
        print(f"house {h}: {len(d):,} minutes, {int(d.evaluable.sum()):,} evaluable -> {p.stat().st_size / 1e6:.1f} MB")
    print(f"total {total / 1e6:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
