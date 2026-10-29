"""Check that the washing-machine channel in each house looks like a washing machine.

For every appliance channel we compute simple signature statistics on the 1-minute
cache. A washing machine should be OFF most of the time, have runs of roughly
1-3 hours, and reach heater power (~1.5-2.5 kW) in at least some runs.
We also check month by month that the WM channel keeps that signature
(the readme warns that monitors were sometimes moved between appliances).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean_data import load_house
from src.config import APPLIANCE_COLS, HOUSES_ALL, OUT_DIR, WD_CHANNEL, WM_CHANNELS

ON_W = 20          # provisional ON level used only for this check (final theta is set in P2)
MERGE_GAP = 10     # minutes; pauses up to this length are bridged when counting runs


def runs(on: np.ndarray, merge_gap: int = MERGE_GAP) -> np.ndarray:
    """Lengths (minutes) of ON runs after bridging OFF gaps <= merge_gap."""
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(idx) > merge_gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return ends - starts + 1


def channel_stats(x: pd.Series) -> dict:
    v = x.dropna().to_numpy()
    if v.size == 0:
        return {}
    on = v >= ON_W
    r = runs(on)
    days = len(v) / 1440
    return {
        "on_frac_%": 100 * on.mean(),
        "on_median_W": float(np.median(v[on])) if on.any() else 0.0,
        "max_W": float(v.max()) if v.size else np.nan,
        "min_ge1500W": int((v >= 1500).sum()),
        "runs_per_day": len(r) / days if days else np.nan,
        "run_median_min": float(np.median(r)) if r.size else 0.0,
    }


def verify(houses=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    houses = houses or HOUSES_ALL
    rows, monthly = [], []
    for h in houses:
        df = load_house(h)
        for c in APPLIANCE_COLS:
            role = "WM" if c in WM_CHANNELS.get(h, []) else ("WD" if WD_CHANNEL.get(h) == c else "")
            rows.append({"house": h, "channel": c, "role": role, **channel_stats(df[c])})
        for c in WM_CHANNELS.get(h, []):
            for per, x in df[c].groupby(df.index.tz_convert(None).to_period("M")):
                s = channel_stats(x)
                monthly.append({"house": h, "channel": c, "month": str(per),
                                "valid_days": x.notna().sum() / 1440, **s})
    return pd.DataFrame(rows), pd.DataFrame(monthly)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 20)
    allc, mon = verify()
    allc.round(2).to_csv(OUT_DIR / "channel_signatures.csv", index=False)
    mon.round(2).to_csv(OUT_DIR / "wm_monthly_signatures.csv", index=False)
    print(allc.round(2).to_string(index=False))
