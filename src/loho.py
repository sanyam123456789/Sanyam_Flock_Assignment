"""Leave-one-house-out (LOHO) cross-validation on the DEVELOPMENT houses, and the single
final run on the locked TEST houses.

  * Fold k: train on 15 development houses, predict every evaluable minute of the 16th.
  * Training uses every 5th evaluable minute (features are computed on the full series);
    evaluation uses ALL evaluable minutes.
  * Out-of-fold (OOF) predictions are saved to cache/oof/<name>.parquet (index: UTC minute;
    columns: house, y, one column per prediction variant), so every table can be rebuilt
    without retraining. The same format is used for predictions made elsewhere (e.g. Colab).
  * Leakage: no house is in both training and evaluation of a fold; features use only the
    aggregate; all thresholds come from development houses; test houses are scored once.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src import features as F
from src.config import CACHE_DIR, CFG, OUT_DIR
from src.metrics import score_frame, summarise

DEV, TEST = CFG["dev_houses"], CFG["test_houses"]
OOF_DIR = CACHE_DIR / "oof"
MET_DIR = OUT_DIR / "metrics"
OOF_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(houses: list[int], feats: list[str], which: str) -> pd.DataFrame:
    """which='train' -> every 5th evaluable minute; 'eval' -> all evaluable minutes."""
    col = {"train": "train_row", "eval": "evaluable"}[which]
    out = []
    for h in houses:
        d = F.load(h, columns=list(dict.fromkeys(feats + ["y", "house", "issues", col])))
        out.append(d[d[col]].drop(columns=[col]))
    return pd.concat(out)


def run(make_model, name: str, houses: list[int] = DEV, events: bool = True, verbose: bool = True) -> pd.DataFrame:
    """LOHO over `houses`; returns per-house metrics for every prediction variant."""
    probe = make_model()
    feats = list(probe.features)
    train_all = load_rows(houses, feats, "train")
    parts = []
    for h in houses:
        t0 = time.time()
        m = make_model().fit(train_all[train_all["house"] != h])
        ev = load_rows([h], feats, "eval")
        pr = m.predict(ev)
        pr.insert(0, "y", ev["y"].to_numpy())
        pr.insert(0, "house", h)
        parts.append(pr)
        if verbose:
            print(f"  {name}: fold house {h} done in {time.time() - t0:.0f}s", flush=True)
    oof = pd.concat(parts)
    oof.index.name = "time_utc"
    oof.to_parquet(OOF_DIR / f"{name}.parquet")
    return score_oof(name, events=events)


def score_oof(name: str, events: bool = True, split: str = "dev") -> pd.DataFrame:
    oof = pd.read_parquet(OOF_DIR / f"{name}.parquet")
    pred_cols = [c for c in oof.columns if c not in ("house", "y")]
    ph = score_frame(oof, pred_cols, events=events)
    ph["variant"] = name + ":" + ph["variant"]
    ph.to_csv(MET_DIR / f"{name}_{split}_per_house.csv", index=False)
    summarise(ph).to_csv(MET_DIR / f"{name}_{split}_summary.csv")
    return ph


def final_test(make_model, name: str, events: bool = True) -> pd.DataFrame:
    """THE single test run: train on all development houses, predict each test house once."""
    m = make_model()
    feats = list(m.features)
    m.fit(load_rows(DEV, feats, "train"))
    parts = []
    for h in TEST:
        ev = load_rows([h], feats, "eval")
        pr = m.predict(ev)
        pr.insert(0, "y", ev["y"].to_numpy())
        pr.insert(0, "house", h)
        parts.append(pr)
    oof = pd.concat(parts)
    oof.index.name = "time_utc"
    oof.to_parquet(OOF_DIR / f"TEST_{name}.parquet")
    return score_oof(f"TEST_{name}", events=events, split="test")


def compare(per_house: pd.DataFrame, metrics=("nde_day", "mae_w", "mae_active_w", "sae", "f1",
                                                 "precision", "recall", "event_recall", "event_precision")) -> pd.DataFrame:
    """Compact macro table: mean across houses (and min-max) for each variant."""
    g = per_house.groupby("variant")
    t = g[list(metrics)].mean().round(3)
    t.insert(0, "houses", g.size())
    t["nde_day_min"], t["nde_day_max"] = g["nde_day"].min().round(3), g["nde_day"].max().round(3)
    return t.sort_values("nde_day")


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))
