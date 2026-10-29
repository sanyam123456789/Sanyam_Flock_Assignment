"""P5: the single final evaluation on the locked TEST houses.

The model configuration is read from outputs/metrics/selection.json (decided on development
houses and committed before this script was first run). It is retrained on all 16
development houses and applied once to each test house. Both baselines are scored on the
same minutes. Results are reported as they are.

Uncertainty: day-block bootstrap (1,000 resamples of counted days, with replacement) per house,
and for the macro mean (days resampled within each house at the same time).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import features as F
from src import loho
from src.baselines import AlwaysOff, HourOfWeek
from src.config import CACHE_DIR, CFG
from src.metrics import daily, score_frame, summarise
from src.models import TwoStage, gate

TEST, DEV = CFG["test_houses"], CFG["dev_houses"]
MODEL_DIR = CACHE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OOF = loho.OOF_DIR / "TEST_final.parquet"
N_BOOT = 1000


def selected() -> dict:
    sel = json.loads((loho.MET_DIR / "selection.json").read_text())["final_choice"]
    v = sel["variant"].split(":")[1]                       # e.g. hard_r800_t0.4
    assert sel["config"] == "lgb_twostage" and v.startswith("hard"), sel
    return {"calendar": bool(sel.get("calendar")), "rounds": int(v.split("_r")[1].split("_")[0]),
            "t": float(v.split("_t")[1]), "variant": sel["variant"]}


def predict_test() -> pd.DataFrame:
    s = selected()
    for h in TEST:
        F.load(h, rebuild=True)                             # build test-house features
    m = TwoStage(calendar=s["calendar"], rounds=(s["rounds"],))
    train = loho.load_rows(DEV, m.features, "train")
    m.fit(train)
    m.clf.save_model(str(MODEL_DIR / "final_classifier.txt"))
    m.reg.save_model(str(MODEL_DIR / "final_regressor.txt"))
    how = HourOfWeek().fit(train)
    parts = []
    for h in TEST:
        ev = loho.load_rows([h], m.features, "eval")
        out = m.predict(ev)
        out["model"] = gate(out, s["rounds"], "hard", s["t"]).astype("float32")
        out["always_off"] = AlwaysOff().predict(ev)["pred"].to_numpy()
        out["hour_of_week"] = how.predict(ev)["pred"].to_numpy()
        out.insert(0, "y", ev["y"].to_numpy())
        out.insert(0, "house", h)
        parts.append(out)
    res = pd.concat(parts)
    res.index.name = "time_utc"
    res.to_parquet(FINAL_OOF)
    return res


def _day_arrays(y: pd.Series, p: pd.Series):
    d = daily(y, p.clip(lower=0))
    d = d[d.counted]
    return (d.est - d.act).abs().to_numpy(), d.act.to_numpy()


def bootstrap(res: pd.DataFrame, col: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    per = {h: _day_arrays(g["y"], g[col]) for h, g in res.groupby("house")}
    boots = {h: np.empty(N_BOOT) for h in per}
    for b in range(N_BOOT):
        for h, (ae, act) in per.items():
            i = rng.integers(0, len(ae), len(ae))
            boots[h][b] = ae[i].sum() / act[i].sum()
    macro = np.mean(np.vstack(list(boots.values())), axis=0)
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"per_house_nde_ci": {int(h): ci(a) for h, a in boots.items()}, "macro_nde_ci": ci(macro)}


def main() -> dict:
    res = predict_test() if not FINAL_OOF.exists() else pd.read_parquet(FINAL_OOF)
    cols = ["model", "always_off", "hour_of_week"]
    ph = score_frame(res, cols)
    ph.to_csv(loho.MET_DIR / "test_per_house.csv", index=False)
    summarise(ph).to_csv(loho.MET_DIR / "test_summary.csv")
    out = {"selected": selected(), "test_houses": TEST,
           "macro": ph.groupby("variant")[["nde_day", "daily_mae_kwh", "mae_w", "mae_active_w", "sae", "rmse_w",
                                           "precision", "recall", "f1", "event_recall", "event_precision"]]
                   .mean().round(4).to_dict("index"),
           "bootstrap": {c: bootstrap(res, c) for c in ("model", "hour_of_week")}}
    (loho.MET_DIR / "test_results.json").write_text(json.dumps(out, indent=2, default=str))
    return {"per_house": ph, "summary": out}


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(r["per_house"].round(3).to_string(index=False))
    print(json.dumps(r["summary"], indent=1, default=str))
