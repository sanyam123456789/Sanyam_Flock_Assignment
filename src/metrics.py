"""Evaluation metrics. All are computed PER HOUSE on evaluable minutes, then summarised
across houses (macro mean, median, min, max). No pooled-only numbers.

Primary (model selection):
  NDE_day = sum over counted days |estimated daily kWh - actual daily kWh| / sum of actual daily kWh
  A day (UK local date) is counted if >= DAY_VALID of its minutes are evaluable; daily kWh is
  summed over the same evaluable minutes for actual and estimate. Our own definition, related
  to (not taken from) the energy metrics in Zhang et al. 2018 and Pereira & Nunes 2018.
  It matches how a utility would use the output (daily consumption), and errors cannot
  cancel across days.

Secondary / supplementary:
  daily_mae_kwh   mean |est - act| per counted day (NDE_day in kWh units)
  mae_w           minute-level MAE (always shown next to always-off: the "MAE trap")
  mae_active_w    MAE on minutes where the washing machine is ON (actual >= THETA)
  sae             |total estimated energy - total actual| / total actual (can hide cancelling errors)
  rmse_w
  precision/recall/f1   minute ON-state at THETA ("knows WHEN", not energy accuracy)
  event_recall/event_precision  cycles found by the EDA cycle rule on actual vs predicted power;
                  a cycle counts as matched if it overlaps any cycle on the other side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LOCAL_TZ
from src.cycles import THETA, detect_cycles

DAY_VALID = 0.9
METRICS = ["nde_day", "daily_mae_kwh", "mae_w", "mae_active_w", "sae", "rmse_w",
           "precision", "recall", "f1", "event_recall", "event_precision", "days", "act_kwh_per_day"]


def _expected_minutes(dates: pd.DatetimeIndex) -> np.ndarray:
    """Minutes in each UK-local day (1380 / 1440 / 1500 around clock changes)."""
    nxt = (dates + pd.Timedelta(hours=36)).normalize()  # +36 h is safe on 23 h and 25 h days
    return ((nxt - dates).total_seconds() / 60).to_numpy()


def daily(y: pd.Series, p: pd.Series) -> pd.DataFrame:
    """Daily actual/estimated kWh over evaluable minutes; `counted` marks valid days."""
    loc = y.index.tz_convert(LOCAL_TZ)
    key = loc.normalize()
    g = pd.DataFrame({"act": y.to_numpy(), "est": p.to_numpy()}, index=key).groupby(level=0)
    t = g.sum() / 60 / 1000
    t["n"] = g.size()
    t["counted"] = t["n"] >= DAY_VALID * _expected_minutes(t.index)
    return t


def _events(y: pd.Series, p: pd.Series) -> tuple[float, float]:
    minute = (y.index.asi8 // 60_000_000_000) if y.index.unit == "ns" else (
        y.index.as_unit("ns").asi8 // 60_000_000_000)
    pos = minute - minute.min()
    n = int(pos.max()) + 1
    idx = pd.date_range(y.index.min(), periods=n, freq="min")
    v = np.zeros(n, bool)
    v[pos] = True
    yy = np.zeros(n, "float32"); yy[pos] = y.to_numpy()
    pp = np.zeros(n, "float32"); pp[pos] = p.to_numpy()
    valid, yy, pp = pd.Series(v, index=idx), pd.Series(yy, index=idx), pd.Series(pp, index=idx)
    ca, cp = detect_cycles(yy, valid), detect_cycles(pp, valid)
    if len(ca) == 0:
        return np.nan, np.nan

    def overlaps(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
        if len(b) == 0:
            return np.zeros(len(a), bool)
        bs, be = b.i0.to_numpy(), b.i1.to_numpy()
        order = np.argsort(bs)
        bs, be = bs[order], be[order]
        run_max_end = np.maximum.accumulate(be)
        # b intervals with start <= a.end; overlap if the max end among them >= a.start
        k = np.searchsorted(bs, a.i1.to_numpy(), side="right") - 1
        ok = k >= 0
        out = np.zeros(len(a), bool)
        out[ok] = run_max_end[k[ok]] >= a.i0.to_numpy()[ok]
        return out

    rec = overlaps(ca, cp).mean()
    prec = overlaps(cp, ca).mean() if len(cp) else np.nan
    return float(rec), float(prec)


def house_metrics(y: pd.Series, p: pd.Series, events: bool = True) -> dict:
    """y, p: actual and predicted W on the evaluable minutes of ONE house (UTC index)."""
    p = p.clip(lower=0)
    d = daily(y, p)
    dc = d[d.counted]
    err = p - y
    on_a, on_p = y >= THETA, p >= THETA
    tp = int((on_a & on_p).sum())
    prec = tp / on_p.sum() if on_p.sum() else np.nan
    rec = tp / on_a.sum() if on_a.sum() else np.nan
    f1 = 0.0 if not tp else 2 * prec * rec / (prec + rec)
    er, ep = _events(y, p) if events else (np.nan, np.nan)
    return {
        "nde_day": float((dc.est - dc.act).abs().sum() / dc.act.sum()),
        "daily_mae_kwh": float((dc.est - dc.act).abs().mean()),
        "mae_w": float(err.abs().mean()),
        "mae_active_w": float(err[on_a].abs().mean()),
        "sae": float(abs(p.sum() - y.sum()) / y.sum()),
        "rmse_w": float(np.sqrt((err ** 2).mean())),
        "precision": prec, "recall": rec, "f1": f1,
        "event_recall": er, "event_precision": ep,
        "days": int(len(dc)), "act_kwh_per_day": float(dc.act.mean()),
    }


def score_frame(oof: pd.DataFrame, pred_cols: list[str], events: bool = True) -> pd.DataFrame:
    """Per-house metrics for each prediction column of an OOF frame
    (columns: house, y, <pred cols>; index: UTC minute)."""
    rows = []
    for h, g in oof.groupby("house"):
        for c in pred_cols:
            rows.append({"variant": c, "house": int(h), **house_metrics(g["y"], g[c], events)})
    return pd.DataFrame(rows)


def summarise(per_house: pd.DataFrame) -> pd.DataFrame:
    """Macro mean, median, min and max across houses for each variant."""
    m = [c for c in METRICS if c in per_house.columns]
    g = per_house.groupby("variant")[m]
    out = pd.concat({"mean": g.mean(), "median": g.median(), "min": g.min(), "max": g.max()}, axis=1)
    return out.swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def bootstrap_days(y: pd.Series, p: pd.Series, n_boot: int = 1000, seed: int = 42) -> dict:
    """Day-block bootstrap of NDE_day and minute MAE for one house: resample counted days
    with replacement (keeps within-day dependence). Returns 2.5% / 97.5% percentiles."""
    p = p.clip(lower=0)
    d = daily(y, p)
    dc = d[d.counted]
    loc_day = y.index.tz_convert(LOCAL_TZ).normalize()
    keep = loc_day.isin(dc.index)
    ae_day = pd.Series(np.abs(p - y).to_numpy()[keep], index=loc_day[keep]).groupby(level=0).agg(["sum", "size"])
    ae_day = ae_day.reindex(dc.index)
    rng = np.random.default_rng(seed)
    n = len(dc)
    absd = (dc.est - dc.act).abs().to_numpy()
    act = dc.act.to_numpy()
    s_ae, s_n = ae_day["sum"].to_numpy(), ae_day["size"].to_numpy()
    nde, mae = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        nde[b] = absd[i].sum() / act[i].sum()
        mae[b] = s_ae[i].sum() / s_n[i].sum()
    return {"nde_day_ci": [float(np.percentile(nde, 2.5)), float(np.percentile(nde, 97.5))],
            "mae_w_ci": [float(np.percentile(mae, 2.5)), float(np.percentile(mae, 97.5))]}
