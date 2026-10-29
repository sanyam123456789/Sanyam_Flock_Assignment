"""Validity rules for the official cleaned data (Sections 2 and 3).

All rules are data-quality rules, fixed BEFORE the test houses were drawn and applied
identically to every house (see DATA.md section 5):

  * no reading in the minute                      -> invalid (missing)
  * aggregate 1-minute mean exactly identical for >= FROZEN_MIN consecutive minutes
                                                  -> invalid (frozen sensor; the official
                                                     forward-fill hides outages this way)
  * outside the house's period (House 13: before 25 Mar 2015 only) -> not used
  * WM channel silent (no minute >= 20 W) for >= DEAD_WM_DAYS days while the household is
    clearly at home (median daily aggregate energy >= DEAD_WM_ACTIVE x its usual level)
                                                  -> WM channel treated as unavailable
  * a WM "cycle" (src/cycles.py rule) longer than 6 hours is not a washing programme
    (e.g. House 5: 13 h at a steady 2.1 kW) -> those minutes are invalid target data
    (rule added after the test lock, using development-house evidence only)
  * `Issues` (sum of IAMs > aggregate) is KEPT and flagged: it marks 11-46% of WM-ON minutes
    because the sensors are not synchronised, so dropping it would remove the events we study.
A valid day = UK-local calendar day with >= VALID_DAY_FRAC valid minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean_data import load_house
from src.config import LOCAL_TZ, PERIOD_END, WM_CHANNELS
from src.cycles import MAX_CYCLE, detect_cycles

FROZEN_MIN = 30
DEAD_WM_DAYS = 21
DEAD_WM_ACTIVE = 0.75
SILENT_W = 20          # "any activity" level for the dead-monitor rule only
VALID_DAY_FRAC = 0.90


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def frozen_mask(agg: pd.Series, min_len: int = FROZEN_MIN) -> np.ndarray:
    a = agg.to_numpy()
    same = np.r_[False, (a[1:] == a[:-1]) & ~np.isnan(a[1:])]
    # a run of identical values = first minute + following "same" minutes
    start_of_run = ~same
    run_id = np.cumsum(start_of_run)
    lens = np.bincount(run_id)[run_id]
    return (lens >= min_len) & ~np.isnan(a)


def dead_wm_stretches(df: pd.DataFrame, wm: pd.Series) -> list[tuple]:
    loc = df.index.tz_convert(LOCAL_TZ)
    day = pd.Index(loc.date)
    present = df["valid_agg"].groupby(day).mean() > 0.5
    active = (wm >= SILENT_W).groupby(day).sum()
    kwh = df["Aggregate"].where(df["valid_agg"]).groupby(day).mean() * 24 / 1000
    usual = kwh[present].median()
    days = active.index[present.reindex(active.index, fill_value=False).to_numpy()]
    silent = (active.loc[days] == 0).to_numpy()
    out = []
    for s, e in zip(*_runs(silent)):
        if e - s >= DEAD_WM_DAYS:
            win = days[s:e]
            if kwh.loc[win].median() >= DEAD_WM_ACTIVE * usual:
                out.append((win[0], win[-1], int(e - s), round(float(kwh.loc[win].median()), 1), round(float(usual), 1)))
    return out


def house_frame(house: int) -> tuple[pd.DataFrame, dict]:
    """1-minute frame with target `wm` (sum of the house's WM channels), validity flags
    and `issues`. Index: UTC minutes."""
    d = load_house(house)
    if house in PERIOD_END:
        d = d[d.index < pd.Timestamp(PERIOD_END[house], tz=LOCAL_TZ)]
    chans = WM_CHANNELS[house]
    wm = d[chans].sum(axis=1, min_count=len(chans))
    has = d["n"].gt(0) & d["Aggregate"].notna() & wm.notna()
    frz = frozen_mask(d["Aggregate"])
    d = d.assign(valid_agg=has & ~frz)
    dead = dead_wm_stretches(d, wm)
    dead_mask = np.zeros(len(d), bool)
    loc_date = pd.Index(d.index.tz_convert(LOCAL_TZ).date)
    for a, b, *_ in dead:
        dead_mask |= (loc_date >= a) & (loc_date <= b)
    valid = (d["valid_agg"] & ~dead_mask).to_numpy()
    # WM "cycles" longer than MAX_CYCLE are not washing programmes -> invalid target minutes
    cyc = detect_cycles(wm, pd.Series(valid, index=d.index))
    implausible = np.zeros(len(d), bool)
    for a, b in cyc.loc[cyc.minutes > MAX_CYCLE, ["i0", "i1"]].itertuples(index=False):
        implausible[a:b + 1] = True
    out = pd.DataFrame({
        "agg": d["Aggregate"], "wm": wm,
        # aggregate usable as model INPUT (features); independent of the WM channel
        "valid_agg": (d["n"].gt(0) & d["Aggregate"].notna()).to_numpy() & ~frz,
        # minute usable as a TARGET (aggregate and WM channel both trustworthy)
        "valid": valid & ~implausible,
        "issues": d["issues_frac"].gt(0).to_numpy(),
    }, index=d.index)
    for c in chans:  # keep separate channels (House 4 has two)
        out[c] = d[c]
    info = {
        "house": house, "wm_channels": "+".join(chans),
        "span_minutes": len(d),
        "minutes_no_reading": int((~has).sum()),
        "minutes_frozen": int((frz & has).sum()),
        "dead_wm_stretches": [(str(a), str(b), n) for a, b, n, *_ in dead],
        "minutes_dead_wm": int((dead_mask & d["valid_agg"]).sum()),
        "implausible_runs_gt6h": int((cyc.minutes > MAX_CYCLE).sum()),
        "minutes_implausible_runs": int(implausible.sum()),
        "minutes_valid": int(out["valid"].sum()),
        "issues_pct_valid": round(100 * float(out.loc[out.valid, "issues"].mean()), 2),
    }
    return out, info


def daily_validity(df: pd.DataFrame) -> pd.Series:
    """Share of valid minutes per UK-local day (days counted over 1440 minutes)."""
    loc = pd.Index(df.index.tz_convert(LOCAL_TZ).date)
    return df["valid"].groupby(loc).sum() / 1440
