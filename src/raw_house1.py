"""Section 1: inspection and cleaning of RAW House 1 -> 1-minute series.

Pipeline (each rule is stated in DATA.md):
  1. Parse   both parts with explicit dtypes; `Unix` (integer seconds) is the timestamp.
             `Time` is a text copy of the same instant, sometimes 1 s different; not used.
  2. Time    raw timestamps are UK LOCAL clock time (evidence: the 01:00 hour is repeated at
             the October clock change and missing at the March change). We convert local ->
             UTC. Readings in the repeated 01:00 hour cannot be placed and are dropped.
  3. Sort    by UTC timestamp.
  4. Dupes   (a) exact duplicate rows (same Unix + same values) -> keep one;
             (b) rows sharing a timestamp with different values -> one row per timestamp,
                 each channel = mean of its non-missing values at that timestamp
                 (in Part2 this merges partial rows from sensors reporting separately).
  5. Values  impossible values -> missing (NaN), never silently replaced by 0:
               * IAM channel > 4000 W   (13 A plug ~ 3 kW max; values are bit-error codes)
               * Aggregate > 20000 W    (above a typical domestic supply; includes the
                                         clamp's 24000 W ceiling)
               * Aggregate == 0 W       (Part1 writes 0 for an unavailable sensor; a home
                                         never draws exactly 0 W)
               * negative values        (none found)
  6. Resample each part to 1-minute means (UTC minute bins); minutes with no reading = NaN.
  7. Overlap  Part1 and Part2 overlap (27 Jun - 1 Oct 2014): per channel and minute, use
             Part1 if it has a value, else Part2.
  8. Gaps    per channel: gaps of <= FILL_MAX_MIN minutes are forward-filled (sensors log on
             change, so the last value most likely still held); longer gaps stay NaN and are
             flagged as outages. Flag per channel: 0 measured, 1 short-gap filled, 2 outage.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import APPLIANCE_COLS, CACHE_DIR, LOCAL_TZ, OUT_DIR, raw_files

COLS = ["Aggregate"] + APPLIANCE_COLS
IAM_MAX_W = 4000
AGG_MAX_W = 20000
FILL_MAX_MIN = 5
HOUSE = 1


def load_part(i: int) -> pd.DataFrame:
    f = raw_files(HOUSE)[i - 1]
    return pd.read_csv(f, engine="pyarrow",
                       dtype={"Time": "string", "Unix": "int64"} | {c: "float64" for c in COLS})


def inspect_part(d: pd.DataFrame) -> dict:
    t = pd.to_datetime(d["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    off = (t - pd.to_datetime(d["Unix"], unit="s")).dt.total_seconds()
    u = d["Unix"].to_numpy()
    vals = d[["Unix"] + COLS]
    exact_dup = int(vals.duplicated().sum())
    dedup = vals.drop_duplicates()
    conflict_ts = int(dedup["Unix"].duplicated().sum())
    ut = np.unique(u)
    gaps = np.diff(ut)
    agg = d["Aggregate"]
    iam = d[APPLIANCE_COLS]
    return {
        "rows": len(d),
        "first_time_raw": str(d["Time"].iloc[0]), "last_time_raw": str(d["Time"].iloc[-1]),
        "time_parse_failures": int(t.isna().sum()),
        "time_minus_unix_seconds": {str(k): int(v) for k, v in off.value_counts().items()},
        "out_of_order_steps": int((np.diff(u) < 0).sum()),
        "exact_duplicate_rows": exact_dup,
        "same_timestamp_conflicting_rows": conflict_ts,
        "unique_timestamps": int(ut.size),
        "interval_s_percentiles_5_25_50_75_95_99": [float(x) for x in np.percentile(gaps, [5, 25, 50, 75, 95, 99])],
        "gaps_gt_1min": int((gaps > 60).sum()), "gaps_gt_5min": int((gaps > 300).sum()),
        "gaps_gt_1h": int((gaps > 3600).sum()), "gaps_gt_1day": int((gaps > 86400).sum()),
        "largest_gaps_h": [(str(pd.to_datetime(ut[k], unit="s")), round(gaps[k] / 3600, 1))
                           for k in np.argsort(gaps)[-5:][::-1]],
        "nan_per_channel": {c: int(d[c].isna().sum()) for c in COLS},
        "rows_with_any_nan": int(d[COLS].isna().any(axis=1).sum()),
        "zero_share_pct": {c: round(100 * float((d[c] == 0).mean()), 2) for c in COLS},
        "aggregate_zero_rows": int((agg == 0).sum()),
        "negative_values": int((d[COLS] < 0).sum().sum()),
        "aggregate_gt_10kW": int((agg > 10000).sum()), "aggregate_gt_20kW": int((agg > AGG_MAX_W).sum()),
        "aggregate_max": float(agg.max()),
        "iam_gt_4kW_per_channel": {c: int((d[c] > IAM_MAX_W).sum()) for c in APPLIANCE_COLS},
        "iam_max": float(iam.max().max()),
        "rows_iam_sum_gt_aggregate": int((iam.sum(axis=1, min_count=1) > agg).sum()),
    }


def to_utc(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    local = pd.to_datetime(d["Unix"], unit="s").dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    bad = local.isna()
    out = d.loc[~bad, ["Unix"] + COLS].copy()
    out["Unix"] = ((local[~bad].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp(0))
                   .dt.total_seconds().astype("int64"))
    return out, {"rows_dropped_unplaceable_local_time": int(bad.sum())}


def dedupe(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = d.sort_values("Unix", kind="stable")
    n0 = len(d)
    d = d.drop_duplicates()
    n_exact = n0 - len(d)
    n_conf = int(d["Unix"].duplicated().sum())
    d = d.groupby("Unix", sort=True)[COLS].mean()  # mean of non-missing values per channel
    return d, {"exact_duplicates_removed": n_exact, "same_timestamp_rows_merged": n_conf, "rows_after": len(d)}


def value_rules(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = d.copy()
    info = {
        "iam_gt_4kW_set_nan": {c: int((d[c] > IAM_MAX_W).sum()) for c in APPLIANCE_COLS},
        "aggregate_gt_20kW_set_nan": int((d["Aggregate"] > AGG_MAX_W).sum()),
        "aggregate_zero_set_nan": int((d["Aggregate"] == 0).sum()),
        "negative_set_nan": int((d[COLS] < 0).sum().sum()),
    }
    for c in APPLIANCE_COLS:
        d.loc[d[c] > IAM_MAX_W, c] = np.nan
    d.loc[(d["Aggregate"] > AGG_MAX_W) | (d["Aggregate"] == 0), "Aggregate"] = np.nan
    d[d < 0] = np.nan
    return d, info


def resample_1min(d: pd.DataFrame) -> pd.DataFrame:
    m = d.groupby(d.index // 60)[COLS].mean()
    m.index = pd.to_datetime(m.index * 60, unit="s", utc=True)
    return m


def merge_parts(m1: pd.DataFrame, m2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    idx = pd.date_range(min(m1.index.min(), m2.index.min()), max(m1.index.max(), m2.index.max()), freq="min")
    a, b = m1.reindex(idx), m2.reindex(idx)
    both = a.notna() & b.notna()
    merged = a.combine_first(b)
    info = {
        "overlap_utc": [str(m2.index.min()), str(m1.index.max())],
        "minutes_with_both_parts": {c: int(both[c].sum()) for c in COLS},
        "minutes_filled_from_part2_in_overlap": {
            c: int((a[c].isna() & b[c].notna() & (idx <= m1.index.max()) & (idx >= m2.index.min())).sum()) for c in COLS},
        "aggregate_1min_median_abs_diff_W": float((a["Aggregate"] - b["Aggregate"])[both["Aggregate"]].abs().median()),
    }
    merged.index.name = "time_utc"
    return merged, info


def nan_runs(s: pd.Series) -> pd.DataFrame:
    isna = s.isna().to_numpy()
    if not isna.any():
        return pd.DataFrame(columns=["start", "end", "minutes"])
    edges = np.diff(np.r_[0, isna.astype(int), 0])
    st, en = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return pd.DataFrame({"start": s.index[st], "end": s.index[en - 1], "minutes": en - st})


def fill_and_flag(m: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    out = m.copy()
    info = {}
    for c in COLS:
        runs = nan_runs(m[c])
        short = runs[runs.minutes <= FILL_MAX_MIN]
        flag = np.zeros(len(m), dtype="int8")
        isna = m[c].isna().to_numpy()
        short_mask = np.zeros(len(m), bool)
        for s, n in zip(m.index.get_indexer(short.start), short.minutes):
            if s > 0:  # a gap at the very start has no previous value to carry
                short_mask[s:s + n] = True
        out[c] = m[c].ffill().where(short_mask | ~isna)
        flag[short_mask] = 1
        flag[isna & ~short_mask] = 2
        out[f"{c}_flag"] = flag
        long_runs = runs[runs.minutes > FILL_MAX_MIN]
        info[c] = {
            "missing_minutes": int(isna.sum()),
            "gaps": int(len(runs)), "short_gaps": int(len(short)),
            "minutes_filled": int(short_mask.sum()),
            "outages_flagged": int(len(long_runs)), "outage_minutes": int(long_runs.minutes.sum()),
        }
    outages = nan_runs(m["Aggregate"])
    outages = outages[outages.minutes > FILL_MAX_MIN].reset_index(drop=True)
    outages["hours"] = (outages.minutes / 60).round(2)
    return out, info, outages


def gap_length_table(m: pd.DataFrame) -> pd.DataFrame:
    runs = nan_runs(m["Aggregate"])
    bins = [0, 1, 2, 5, 10, 30, 60, 360, 1440, 10**9]
    labels = ["1", "2", "3-5", "6-10", "11-30", "31-60", "1-6h", "6-24h", ">1 day"]
    runs["len"] = pd.cut(runs.minutes, bins=bins, labels=labels)
    return runs.groupby("len", observed=False).agg(gaps=("minutes", "size"), minutes=("minutes", "sum"))


def run(save: bool = True) -> dict:
    report, frames = {}, {}
    for i in (1, 2):
        d = load_part(i)
        report[f"part{i}_inspection"] = inspect_part(d)
        d, r_t = to_utc(d)
        d, r_d = dedupe(d)
        d, r_v = value_rules(d)
        report[f"part{i}_cleaning"] = r_t | r_d | r_v
        frames[i] = resample_1min(d)
    merged, r_m = merge_parts(frames[1], frames[2])
    report["overlap_merge"] = r_m
    gl = gap_length_table(merged)
    clean, r_f, outages = fill_and_flag(merged)
    report["gaps"] = r_f
    report["aggregate_gap_lengths"] = gl.reset_index().astype({"len": str}).to_dict("records")
    report["span_utc"] = [str(clean.index.min()), str(clean.index.max())]
    report["minutes_total"] = len(clean)
    report["aggregate_valid_pct_after"] = round(100 * float(clean["Aggregate"].notna().mean()), 2)
    agg, iam = clean["Aggregate"], clean[APPLIANCE_COLS].sum(axis=1, min_count=9)
    report["minutes_iam_sum_gt_aggregate"] = int((iam > agg).sum())
    if save:
        clean.to_parquet(CACHE_DIR / "raw_house1_clean_1min.parquet")
        merged.to_parquet(CACHE_DIR / "raw_house1_unfilled_1min.parquet")
        outages.to_csv(OUT_DIR / "raw_house1_outages.csv", index=False)
        gl.to_csv(OUT_DIR / "raw_house1_gap_lengths.csv")
        (OUT_DIR / "raw_house1_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def compare_with_official() -> dict:
    """Optional sanity check: our cleaned raw House 1 vs the official cleaned House 1
    (1-minute, after our time-base correction), on minutes valid in both."""
    from src.clean_data import load_house

    ours = pd.read_parquet(CACHE_DIR / "raw_house1_unfilled_1min.parquet")
    off = load_house(HOUSE)
    out = {}
    for c in ("Aggregate", "Appliance5", "Appliance1"):
        j = pd.concat([ours[c], off[c]], axis=1, keys=["ours", "official"]).dropna()
        out[c] = {"minutes_compared": len(j),
                  "corr": round(float(j.ours.corr(j.official)), 4),
                  "median_abs_diff_W": round(float((j.ours - j.official).abs().median()), 2),
                  "mean_ours_W": round(float(j.ours.mean()), 1), "mean_official_W": round(float(j.official.mean()), 1)}
    v_ours = ours["Aggregate"].notna()
    v_off = off["Aggregate"].reindex(ours.index).notna()
    out["minutes_valid_ours_only"] = int((v_ours & ~v_off).sum())
    out["minutes_valid_official_only"] = int((~v_ours & v_off).sum())
    (OUT_DIR / "raw_house1_vs_official.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    from src import plots_section1

    run()
    plots_section1.main()


if __name__ == "__main__":
    import time

    t0 = time.time()
    rep = run()
    print(json.dumps(rep, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")
