"""Features for washing-machine disaggregation, from the household AGGREGATE only.

Rules:
  * Only the aggregate is used. No appliance channel and no lagged washing-machine power
    (no target leakage).
  * Windows are CENTRED on minute t (they look 1-60 minutes into the past AND the future),
    the "midpoint" framing of Seq2Point (Zhang et al. 2018). This suits offline
    disaggregation of historical data; it is NOT a real-time model.
  * Statistics use valid aggregate minutes only (invalid minutes are NaN, never filled).
  * Each house is processed on its own, so windows never cross houses.
  * A minute is "evaluable" (used for training and scoring) only if its target is valid
    and at least WIN_VALID_MIN of the 61-minute window around it has valid aggregate data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, LOCAL_TZ
from src.quality import house_frame

STEP_W = 1000          # W: heater-sized step (dev-house EDA: heater-on step median 1.36 kW in 1 min)
WINDOWS = (5, 15, 31, 61)
LONG = 121
WIN_VALID_MIN = 0.9
TRAIN_EVERY = 5        # train on every 5th evaluable minute (features use the full series)

FEAT_DIR = CACHE_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

CALENDAR = ["hod_sin", "hod_cos", "dow_sin", "dow_cos"]


def _roll(s: pd.Series, w: int):
    return s.rolling(w, center=True, min_periods=1)


def build(house: int) -> pd.DataFrame:
    df, _ = house_frame(house)
    x = df["agg"].where(df["valid_agg"]).astype("float64")
    f = {}
    # point values
    f["x"] = x
    for k in (1, 2):
        f[f"x_m{k}"] = x.shift(k)
        f[f"x_p{k}"] = x.shift(-k)
    # changes
    d = x.diff()
    f["d_back"] = d                       # x(t) - x(t-1)
    f["d_fwd"] = x.shift(-1) - x          # x(t+1) - x(t)
    ad = d.abs()
    for w in (15, 61):
        f[f"absdiff_sum_{w}"] = _roll(ad, w).sum()
    # centred window statistics
    for w in WINDOWS:
        r = _roll(x, w)
        mx, mn = r.max(), r.min()
        f[f"mean_{w}"], f[f"std_{w}"], f[f"min_{w}"], f[f"max_{w}"] = r.mean(), r.std(), mn, mx
        f[f"range_{w}"] = mx - mn
    # heater-sized steps
    up = (d > STEP_W).astype(float).where(d.notna())
    down = (d < -STEP_W).astype(float).where(d.notna())
    for w in (31, 61):
        f[f"steps_up_{w}"] = _roll(up, w).sum()
        f[f"steps_down_{w}"] = _roll(down, w).sum()
    # long context
    r = _roll(x, LONG)
    f[f"mean_{LONG}"], f[f"std_{LONG}"] = r.mean(), r.std()
    # level above the home's baseload (rolling 24 h 5th percentile, centred)
    base = x.rolling(1440, center=True, min_periods=360).quantile(0.05)
    f["above_base"] = x - base
    # data validity of the 61-minute window
    f["valid_frac_61"] = _roll(df["valid_agg"].astype(float), 61).mean()
    # calendar (ABLATION ONLY): UK local time
    loc = df.index.tz_convert(LOCAL_TZ)
    hod = loc.hour + loc.minute / 60
    f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * loc.dayofweek / 7), np.cos(2 * np.pi * loc.dayofweek / 7)

    out = pd.DataFrame(f, index=df.index).astype("float32")
    out["y"] = df["wm"].astype("float32")
    out["issues"] = df["issues"].to_numpy()
    out["evaluable"] = (df["valid"] & df["valid_agg"]).to_numpy() & (out["valid_frac_61"] >= WIN_VALID_MIN).to_numpy()
    out["train_row"] = out["evaluable"] & (np.arange(len(out)) % TRAIN_EVERY == 0)
    out["house"] = np.int16(house)
    return out


FEATURES = [c for c in [
    "x", "x_m1", "x_p1", "x_m2", "x_p2", "d_back", "d_fwd", "absdiff_sum_15", "absdiff_sum_61",
    *[f"{s}_{w}" for w in WINDOWS for s in ("mean", "std", "min", "max", "range")],
    "steps_up_31", "steps_down_31", "steps_up_61", "steps_down_61",
    f"mean_{LONG}", f"std_{LONG}", "above_base", "valid_frac_61"]]


def path(house: int):
    return FEAT_DIR / f"house{house}.parquet"


def load(house: int, columns=None, rebuild: bool = False) -> pd.DataFrame:
    p = path(house)
    if rebuild or not p.exists():
        build(house).to_parquet(p)
    return pd.read_parquet(p, columns=columns)
