"""Smoke test: the whole Section 2-3 flow on a small committed sample, in a few minutes.

    python smoke_test.py            # or: python run_all.py --smoke
    python smoke_test.py --update   # rewrite smoke/expected.json (only after a deliberate change)

Data: smoke/data/CLEAN_House{5,7}.csv.gz are unmodified rows of the official cleaned REFIT
files (CC BY 4.0, Murray et al. 2017) for 18 Aug - 14 Sep 2014. Both are development houses,
so the locked test houses are never touched. The window contains UTC days, local-time days
and dropped switch days, so the time-base correction is exercised.

Flow: cleaned CSV -> time-base correction (committed labels/) -> 1-minute cache -> validity
rules -> features -> leave-one-house-out with always-off, hour-of-week and the two-stage
LightGBM (soft gate) -> metrics. Everything is written to smoke/out/, never to cache/,
outputs/ or figures/.

Smoke-only changes, because 2 houses x 4 weeks is tiny: min_data_in_leaf 500 -> 50 and
100 boosting rounds. The numbers are a check that the code runs and gives the same answer,
not results (one house of training data is far too little); they are compared with smoke/expected.json (tolerance 0.01).
"""
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / "smoke"
OUT = SMOKE / "out"
HOUSES = [5, 7]
ROUNDS = (50, 100)
TOL = 0.01

# Point every path at smoke/out/ BEFORE importing src (src.config reads these at import).
shutil.rmtree(OUT, ignore_errors=True)
for sub in ("DATA/clean", "cache", "outputs", "figures"):
    (OUT / sub).mkdir(parents=True)
os.environ.update(REFIT_DATA_DIR=str(OUT / "DATA"), REFIT_CACHE_DIR=str(OUT / "cache"),
                  REFIT_OUT_DIR=str(OUT / "outputs"), REFIT_FIG_DIR=str(OUT / "figures"))
os.environ.pop("REFIT_LABEL_DIR", None)  # always the committed labels/

sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from src import clean_data, features, loho, models  # noqa: E402
from src.baselines import AlwaysOff, HourOfWeek  # noqa: E402
from src.metrics import score_frame  # noqa: E402

models.BASE_PARAMS = models.BASE_PARAMS | {"min_data_in_leaf": 50}  # smoke only (see docstring)


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    t0 = time.time()
    print("[1/4] Unpack the sample", flush=True)
    for h in HOUSES:
        with gzip.open(SMOKE / "data" / f"CLEAN_House{h}.csv.gz", "rb") as src, \
                open(OUT / "DATA" / "clean" / f"CLEAN_House{h}.csv", "wb") as dst:
            shutil.copyfileobj(src, dst)

    print("[2/4] Cleaned CSV -> time-base correction -> 1-minute cache", flush=True)
    stats = {h: clean_data.build_house(h) for h in HOUSES}
    for h, s in stats.items():
        check(s["time_col_mismatch_vs_unix"] == 0, f"house {h}: Time column equals Unix")
        check(s["tb_days_local_converted"] == 3, f"house {h}: 3 local-time days converted to UTC")
        check(s["tb_days_dropped"] >= 1, f"house {h}: {s['tb_days_dropped']} switch day(s) dropped")
        m = clean_data.load_house(h)
        check(str(m.index.tz) == "UTC" and m.index.is_monotonic_increasing, f"house {h}: 1-minute index is sorted UTC")

    print("[3/4] Validity rules and features", flush=True)
    for h in HOUSES:
        f = features.load(h)
        check(f["evaluable"].mean() > 0.5, f"house {h}: {f['evaluable'].mean():.0%} of minutes evaluable")
        check(f.loc[f.evaluable, "x"].notna().all(), f"house {h}: aggregate present on every evaluable minute")

    print("[4/4] Leave-one-house-out: baselines and two-stage LightGBM", flush=True)
    loho.run(AlwaysOff, "always_off", houses=HOUSES, verbose=False)
    loho.run(HourOfWeek, "hour_of_week", houses=HOUSES, verbose=False)
    loho.run(lambda: models.TwoStage(rounds=ROUNDS), "lgb_twostage", houses=HOUSES, events=False, verbose=False)

    oof = {n: pd.read_parquet(loho.OOF_DIR / f"{n}.parquet")
           for n in ("always_off", "hour_of_week", "lgb_twostage")}
    tw = oof["lgb_twostage"]
    tw["soft"] = models.gate(tw, max(ROUNDS), "soft")
    res = {}
    for name, df, col in (("always_off", oof["always_off"], "pred"),
                          ("hour_of_week", oof["hour_of_week"], "pred"),
                          ("lgb_twostage_soft", tw, "soft")):
        ph = score_frame(df, [col], events=True)
        for _, r in ph.iterrows():
            res[f"{name}/house{int(r.house)}"] = {k: round(float(r[k]), 4) for k in ("nde_day", "mae_w", "f1")}
    for k, v in res.items():
        print(f"        {k:28s} NDE_day {v['nde_day']:.3f}  MAE {v['mae_w']:6.1f} W  F1 {v['f1']:.3f}")
    check(all(v == v for r in res.values() for v in r.values()), "all metrics are numbers (no NaN)")

    exp_path = SMOKE / "expected.json"
    if "--update" in sys.argv or not exp_path.exists():
        exp_path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  wrote {exp_path.relative_to(ROOT)}")
    else:
        exp = json.loads(exp_path.read_text())
        bad = [f"{k}.{m}: got {res[k][m]}, expected {exp[k][m]}" for k in exp for m in exp[k]
               if abs(res[k][m] - exp[k][m]) > TOL * max(1.0, abs(exp[k][m]))]
        for b in bad:
            print("        " + b)
        check(not bad, f"metrics match smoke/expected.json (tolerance {TOL:.0%})")
    print(f"Smoke test passed in {time.time() - t0:.0f} s (outputs in smoke/out/)")


if __name__ == "__main__":
    main()
