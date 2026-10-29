"""Which time base does the official cleaned file use, day by day?

Our finding (see DATA.md): raw REFIT timestamps are UK local clock time (the 01:00 hour is
repeated at the October clock change and missing at the March change). The official cleaned
files are UTC for some periods and UK local time for others.

Method: for every cleaned row we look for a raw row (Part1 or Part2) with an identical
Aggregate value either at the same timestamp or 1 hour later. (Only Aggregate is used:
the official cleaning moved some appliance columns, e.g. in Houses 12, 13 and 21, but
never the aggregate clamp.)
  * clean row == raw row 1 h later  -> the cleaned timestamp is UTC (raw is local, BST = UTC+1)
  * clean row == raw row same time  -> the cleaned timestamp is UK local time
This only tells the two apart during British Summer Time (BST). In winter (GMT) local == UTC,
so those days need no correction.

Day labels (BST days only):
  UTC      >= 20% of rows match at +1 h and < 5% at 0 h
  local    >= 20% of rows match at 0 h and < 5% at +1 h
  mixed    >= 10% at both                     (a switch day: dropped)
  unclear  anything else, e.g. too few matches (resolved from neighbours, see resolve_labels)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.config import HOUSES_ALL, LABEL_DIR, LOCAL_TZ, OUT_DIR, WORKERS, clean_file, raw_files

USE = ["Unix", "Aggregate"]
CLOCK_CHANGES_AUTUMN = ["2013-10-27", "2014-10-26"]
CLOCK_CHANGES_SPRING = ["2014-03-30", "2015-03-29"]


def _bst_days(days: pd.Index) -> np.ndarray:
    noon = (pd.to_datetime(days.astype(str)) + pd.Timedelta(hours=12)).tz_localize("UTC")
    local = noon.tz_convert(LOCAL_TZ)
    return np.asarray(local.tz_localize(None) != noon.tz_localize(None))  # BST: local clock != UTC clock


def label_path(house: int):
    return LABEL_DIR / f"house{house}_timebase_days.csv"


def daily_labels(house: int, recompute: bool = False) -> pd.DataFrame:
    """Per-day match shares and label. The labels are committed in labels/, so Sections 2-3
    need only the cleaned files; recompute=True re-derives them from the raw files."""
    path = label_path(house)
    if path.exists() and not recompute:
        return pd.read_csv(path, index_col=0)
    cl = pd.read_csv(clean_file(house), engine="pyarrow", usecols=USE)
    # Only "distinctive" rows: the aggregate differs from both neighbouring readings.
    # In long flat stretches a value also matches by coincidence 1 h away.
    a = cl["Aggregate"]
    cl = cl[(a != a.shift()) & (a != a.shift(-1))].reset_index(drop=True)
    day = pd.to_datetime(cl["Unix"], unit="s").dt.strftime("%Y-%m-%d").values
    hit0 = np.zeros(len(cl), bool)
    hit1 = np.zeros(len(cl), bool)
    for f in raw_files(house):
        r = pd.read_csv(f, engine="pyarrow", usecols=USE).dropna().drop_duplicates("Unix")
        for off, hit in ((0, hit0), (3600, hit1)):
            m = cl.merge(r.assign(Unix=r.Unix - off), on="Unix", how="left", suffixes=("", "_r"))
            hit |= (m.Aggregate == m.Aggregate_r).to_numpy()
    t = pd.DataFrame({"rows": 1, "match_0h": hit0 & ~hit1, "match_1h": hit1 & ~hit0}).groupby(day).agg(
        rows=("rows", "sum"), match_0h=("match_0h", "mean"), match_1h=("match_1h", "mean"))
    t["bst"] = _bst_days(t.index)
    lab = np.full(len(t), "unclear", dtype=object)
    lab[(t.match_1h >= 0.2) & (t.match_0h < 0.05)] = "UTC"
    lab[(t.match_0h >= 0.2) & (t.match_1h < 0.05)] = "local"
    lab[(t.match_0h >= 0.1) & (t.match_1h >= 0.1)] = "mixed"
    lab[~t.bst.to_numpy()] = "winter"
    t["label"] = lab
    t.to_csv(path)
    return t


def resolve_labels(t: pd.DataFrame) -> pd.Series:
    """Final per-day decision. 'unclear' BST days take the label of the nearest labelled
    BST days only if the previous and next labelled days agree; otherwise they are dropped."""
    lab = t["label"].copy()
    known = lab.where(lab.isin(["UTC", "local"]))  # 'frozen'/'winter' days give no evidence
    prev, nxt = known.ffill(), known.bfill()
    unclear = lab.eq("unclear")
    agree = unclear & prev.eq(nxt) & prev.notna()
    lab[agree] = prev[agree]
    lab[unclear & ~agree] = "drop"
    lab[lab.eq("mixed")] = "drop"
    # Clock-change days are only partly BST, so the day-level match is not informative.
    # They take the label of the neighbouring BST day (the day before in October,
    # the day after in March). A local-time October change day then loses its repeated
    # 01:00 hour during conversion (it cannot be placed).
    days = pd.to_datetime(pd.Index(lab.index).astype(str))
    for d in CLOCK_CHANGES_AUTUMN + CLOCK_CHANGES_SPRING:
        d = pd.Timestamp(d)
        if d not in days:
            continue
        nb = d - pd.Timedelta(days=1) if str(d.date()) in CLOCK_CHANGES_AUTUMN else d + pd.Timedelta(days=1)
        key, nkey = lab.index[days == d][0], str(nb.date())
        lab[key] = lab.get(nkey, "drop") if lab.get(nkey) in ("UTC", "local", "drop") else "drop"
    return lab


def summary(house: int) -> dict:
    t = daily_labels(house)
    lab = resolve_labels(t)
    bst = t[t.bst]
    # switch points: consecutive labelled BST days whose label changes
    k = bst["label"][bst["label"].isin(["UTC", "local"])]
    switches = [f"{a}->{b} ({k.index[i]})" for i, (a, b) in enumerate(zip(k.shift(), k)) if i and a != b]
    return {
        "house": house,
        "days_with_data": len(t),
        "winter_days": int((~t.bst).sum()),
        "bst_UTC": int((t.label == "UTC").sum()),
        "bst_local": int((t.label == "local").sum()),
        "bst_mixed": int((t.label == "mixed").sum()),
        "bst_unclear": int((t.label == "unclear").sum()),
        "unclear_resolved": int(((t.label == "unclear") & lab.isin(["UTC", "local"])).sum()),
        "days_dropped": int((lab == "drop").sum()),
        "last_UTC_day": k[k == "UTC"].index.max() if (k == "UTC").any() else None,
        "first_local_day": k[k == "local"].index.min() if (k == "local").any() else None,
        # evidence: median share of rows matching raw at +1 h / 0 h on days of each label
        "UTC_days_match_1h/0h": f"{t.loc[t.label == 'UTC', 'match_1h'].median():.2f}/{t.loc[t.label == 'UTC', 'match_0h'].median():.2f}",
        "local_days_match_0h/1h": f"{t.loc[t.label == 'local', 'match_0h'].median():.2f}/{t.loc[t.label == 'local', 'match_1h'].median():.2f}",
        "switches": "; ".join(switches),
    }


# ---- independent check: daily routine shift across clock changes ------------------
CHANGES = {"2014-03-30": "spring", "2014-10-26": "autumn", "2015-03-29": "spring"}


def _profile(s: pd.Series) -> np.ndarray:
    s = s.dropna()
    p = s.groupby(s.index.hour * 60 + s.index.minute).mean().reindex(range(1440))
    p = p.interpolate(limit_direction="both").to_numpy()
    return (p - p.mean()) / p.std()


def routine_shift(s: pd.Series, day: str, days: int = 21) -> float:
    """Lag (minutes) that best aligns the mean daily aggregate profile after the clock
    change with the one before. Timestamps taken at face value. True UTC data should give
    about -60 after a spring change and +60 after an autumn change; local time about 0."""
    t = pd.Timestamp(day)
    b = s[(s.index >= t - pd.Timedelta(days=days)) & (s.index < t - pd.Timedelta(days=1))]
    a = s[(s.index >= t + pd.Timedelta(days=1)) & (s.index < t + pd.Timedelta(days=days))]
    if b.notna().sum() < 5 * 1440 or a.notna().sum() < 5 * 1440:
        return np.nan
    pb, pa = _profile(b), _profile(a)
    lags = np.arange(-120, 121)
    return float(lags[np.argmax([np.corrcoef(pb, np.roll(pa, -L))[0, 1] for L in lags])])


def routine_shift_table() -> pd.DataFrame:
    """Before (official timestamps) vs after (our corrected UTC cache)."""
    from src.clean_data import load_house

    rows = []
    for h in HOUSES_ALL:
        raw = pd.read_csv(clean_file(h), engine="pyarrow", usecols=USE)
        before = raw.groupby(raw.Unix // 60)["Aggregate"].mean()
        before.index = pd.to_datetime(before.index * 60, unit="s")
        before = before.reindex(pd.date_range(before.index.min(), before.index.max(), freq="min"))
        after = load_house(h)["Aggregate"]
        after.index = after.index.tz_convert(None)
        r = {"house": h}
        for d, kind in CHANGES.items():
            r[f"{d} {kind} before"] = routine_shift(before, d)
            r[f"{d} {kind} after"] = routine_shift(after, d)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("house")
    t.loc["median"] = t.median()
    return t


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(daily_labels, HOUSES_ALL))
    s = pd.DataFrame([summary(h) for h in HOUSES_ALL]).set_index("house")
    s.to_csv(OUT_DIR / "clean_timebase_summary.csv")
    pd.set_option("display.width", 300, "display.max_colwidth", 200)
    print(s.drop(columns="switches").to_string())
    print()
    for h, r in s.iterrows():
        print(f"House {h}: {r.switches}")
    rs = routine_shift_table()
    rs.to_csv(OUT_DIR / "clean_timebase_routine_shift.csv")
    print(rs.to_string())
