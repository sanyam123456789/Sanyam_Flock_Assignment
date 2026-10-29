"""Official cleaned REFIT data -> 1-minute Parquet cache (one house at a time).

Resampling rule (Sections 2 and 3):
  * Timestamps come from the `Unix` column. The cleaned `Time` column is the same value
    written out; we check this and record mismatches.
  * Time-base correction (our finding, see src/timebase.py and DATA.md): during British
    Summer Time some periods of the cleaned files are UK local time, not UTC. Each day is
    labelled UTC / local / drop; local days are converted to UTC, dropped days (switch
    days that mix both) are removed, and the repeated 01:00 hour of the October clock
    change in local-time data is removed because it cannot be placed. All output
    timestamps are UTC.
  * Each reading goes to the UTC minute that contains it; a minute's value is the
    plain mean of its readings (about 4-8 readings per minute).
  * Minutes with no readings at all are kept as NaN (missing), never filled here.
  * `issues_frac` = share of readings in the minute with Issues == 1.
  * `n` = number of readings in the minute (0 = missing minute).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.config import APPLIANCE_COLS, CACHE_DIR, HOUSES_ALL, LOCAL_TZ, WORKERS, clean_file
from src.timebase import _bst_days, daily_labels, resolve_labels

POWER_COLS = ["Aggregate"] + APPLIANCE_COLS


def cache_path(house: int):
    return CACHE_DIR / f"house{house}_1min.parquet"


def stats_path(house: int):
    return CACHE_DIR / f"house{house}_clean_stats.json"


def correct_timebase(df: pd.DataFrame, house: int) -> tuple[pd.DataFrame, dict]:
    """Convert UK-local days to UTC, drop switch days and the unplaceable repeated hour."""
    t = daily_labels(house)
    day = pd.to_datetime(df["Unix"], unit="s").dt.strftime("%Y-%m-%d")
    all_days = pd.Index(sorted(day.unique()))
    t = t.reindex(all_days)
    t["bst"] = _bst_days(all_days)
    # Days with rows but no "distinctive" aggregate value at all: the aggregate never
    # changes all day (frozen sensor hidden by the official forward-fill). No time-base
    # evidence and no usable data -> dropped and counted separately.
    frozen = t["label"].isna()
    t["label"] = t["label"].where(~frozen, "frozen")
    lab = resolve_labels(t)
    lab[frozen.to_numpy()] = "frozen"
    row_lab = day.map(lab).to_numpy()

    drop_day = (row_lab == "drop") | (row_lab == "frozen")
    local = row_lab == "local"
    naive = pd.to_datetime(df.loc[local, "Unix"], unit="s")
    as_utc = naive.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    unplaceable = as_utc.isna()
    new_unix = df["Unix"].to_numpy().copy()
    new_unix[local] = np.where(unplaceable, -1, (as_utc.dt.tz_localize(None) - pd.Timestamp(0)).dt.total_seconds()).astype("int64")
    keep = ~drop_day
    keep[np.flatnonzero(local)[unplaceable.to_numpy()]] = False

    out = df.loc[keep].copy()
    out["Unix"] = new_unix[keep]
    n_before = len(out)
    out = out.sort_values("Unix", kind="stable").drop_duplicates("Unix", keep="first")
    info = {
        "tb_days_utc_or_winter": int(lab.isin(["UTC", "winter"]).sum()),
        "tb_days_local_converted": int((lab == "local").sum()),
        "tb_days_dropped": int((lab == "drop").sum()),
        "tb_dropped_days_list": ",".join(lab[lab == "drop"].index),
        "frozen_days_dropped": int((lab == "frozen").sum()),
        "frozen_days_list": ",".join(lab[lab == "frozen"].index),
        "tb_rows_dropped_switch_days": int((row_lab == "drop").sum()),
        "rows_dropped_frozen_days": int((row_lab == "frozen").sum()),
        "tb_rows_dropped_repeated_hour": int(unplaceable.sum()),
        "tb_rows_dropped_overlap_after_shift": int(n_before - len(out)),
    }
    return out, info


def build_house(house: int, force: bool = False) -> dict:
    out, sp = cache_path(house), stats_path(house)
    if out.exists() and sp.exists() and not force:
        return json.loads(sp.read_text())

    df = pd.read_csv(
        clean_file(house),
        engine="pyarrow",
        dtype={c: "float32" for c in POWER_COLS} | {"Unix": "int64", "Issues": "float32", "Time": "string"},
    )
    n_rows = len(df)
    # Time column vs Unix (UTC)
    t_parsed = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    t_unix = pd.to_datetime(df["Unix"], unit="s")
    time_mismatch = int((t_parsed != t_unix).sum())
    del t_parsed, t_unix
    df = df.drop(columns="Time")

    unix = df["Unix"].to_numpy()
    non_monotonic = int((np.diff(unix) < 0).sum())
    dup_ts = int(df["Unix"].duplicated().sum())
    dup_rows = int(df.duplicated().sum())
    neg_vals = int((df[POWER_COLS] < 0).sum().sum())
    nan_vals = int(df[POWER_COLS].isna().sum().sum())
    issues_rows = int((df["Issues"] == 1).sum())

    # --- time-base correction -> true UTC ---------------------------------------
    df, tb = correct_timebase(df, house)
    unix = df["Unix"].to_numpy()
    gaps = np.diff(unix)

    df["minute"] = df["Unix"] // 60
    g = df.groupby("minute", sort=True)
    m = g[POWER_COLS].mean()
    m["issues_frac"] = g["Issues"].mean()
    m["n"] = g.size().astype("int16")
    full = np.arange(m.index.min(), m.index.max() + 1)
    m = m.reindex(full)
    m["n"] = m["n"].fillna(0).astype("int16")
    m.index = pd.to_datetime(m.index * 60, unit="s", utc=True)
    m.index.name = "time_utc"
    m[POWER_COLS + ["issues_frac"]] = m[POWER_COLS + ["issues_frac"]].astype("float32")
    m.to_parquet(out)

    stats = {
        "house": house,
        "n_rows": n_rows,
        "first_utc": str(pd.to_datetime(unix.min(), unit="s")),
        "last_utc": str(pd.to_datetime(unix.max(), unit="s")),
        "time_col_mismatch_vs_unix": time_mismatch,
        "non_monotonic_steps": non_monotonic,
        "duplicate_timestamps": dup_ts,
        "duplicate_rows": dup_rows,
        "negative_values": neg_vals,
        "nan_values": nan_vals,
        "issues_rows": issues_rows,
        "issues_row_frac": issues_rows / n_rows,
        **tb,
        "interval_s_median": float(np.median(gaps)),
        "interval_s_p95": float(np.percentile(gaps, 95)),
        "interval_s_p99": float(np.percentile(gaps, 99)),
        "gaps_over_60s": int((gaps > 60).sum()),
        "gaps_over_1h": int((gaps > 3600).sum()),
        "n_minutes_span": int(len(m)),
        "n_minutes_with_data": int((m["n"] > 0).sum()),
    }
    sp.write_text(json.dumps(stats, indent=2))
    return stats


def load_house(house: int) -> pd.DataFrame:
    """1-minute frame for one house (builds the cache if missing)."""
    if not cache_path(house).exists():
        build_house(house)
    return pd.read_parquet(cache_path(house))


def build_all(houses=None, workers: int = WORKERS, force: bool = False) -> pd.DataFrame:
    houses = houses or HOUSES_ALL
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats = list(ex.map(build_house, houses, [force] * len(houses)))
    return pd.DataFrame(stats).set_index("house")


if __name__ == "__main__":
    import time

    t0 = time.time()
    s = build_all()
    pd.set_option("display.width", 250)
    print(s.T)
    print(f"done in {time.time() - t0:.0f}s")
