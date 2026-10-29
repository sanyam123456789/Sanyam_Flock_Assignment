"""Section 2: exploratory analysis of washing-machine use (official cleaned data, 1 minute).

Cycle rule: see src/cycles.py (parameters chosen on development houses only).
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import style as S
from src.config import CFG, FIG_DIR, LOCAL_TZ, OUT_DIR, WM_CHANNELS
from src.quality import daily_validity, house_frame

from src.cycles import G, HEATER_W, L, MAX_CYCLE, THETA, detect_cycles

LONG_CYCLE = 240  # min: reported (long eco programmes are real; > MAX_CYCLE are removed as invalid)
DEV, TEST = CFG["dev_houses"], CFG["test_houses"]
HOUSES = sorted(WM_CHANNELS)


def runs_of(mask: np.ndarray):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def house_summary(h: int, df: pd.DataFrame, cyc: pd.DataFrame, info: dict) -> dict:
    v = df["valid"]
    dv = daily_validity(df)
    good_days = dv[dv >= 0.9]
    s, e = runs_of(~v.to_numpy())
    gaps = e - s
    w = df.loc[v, "wm"]
    n_days = len(good_days)
    loc_first = df.index[v.to_numpy()].min().tz_convert(LOCAL_TZ)
    loc_last = df.index[v.to_numpy()].max().tz_convert(LOCAL_TZ)
    return {
        "house": h, "role": "test" if h in TEST else "dev", "wm_channels": info["wm_channels"],
        "first_day": str(loc_first.date()), "last_day": str(loc_last.date()),
        "calendar_days": (loc_last.normalize() - loc_first.normalize()).days + 1,
        "valid_days": n_days,
        "valid_minutes_pct": round(100 * v.mean(), 1),
        "missing_intervals_gt5min": int((gaps > 5).sum()),
        "missing_intervals_gt1day": int((gaps > 1440).sum()),
        "frozen_minutes": info["minutes_frozen"],
        "dead_wm_days": sum(x[2] for x in info["dead_wm_stretches"]),
        "wm_exactly0_pct": round(100 * (w == 0).mean(), 1),
        "wm_off_pct": round(100 * (w < THETA).mean(), 1),
        "issues_pct": info["issues_pct_valid"],
        "cycles": len(cyc),
        "cycles_per_week": round(len(cyc) / max(v.sum() / 1440, 1) * 7, 2),
        "duration_median_min": float(cyc.minutes.median()),
        "energy_median_kwh": round(float(cyc.energy_kwh.median()), 3),
        "peak_median_w": round(float(cyc.peak_w.median())),
        "heater_cycles_pct": round(100 * cyc.heater.mean(), 1),
        "long_cycles_gt240": int((cyc.minutes > LONG_CYCLE).sum()),
        "implausible_runs_gt6h_removed": info["implausible_runs_gt6h"],
        "wm_kwh_per_valid_day": round(float(w.sum() / 60 / 1000 / max(v.sum() / 1440, 1)), 3),
        "agg_kwh_per_valid_day": round(float(df.loc[v, "agg"].sum() / 60 / 1000 / max(v.sum() / 1440, 1)), 2),
        "wm_share_of_aggregate_pct": round(float(100 * w.sum() / df.loc[v, "agg"].sum()), 1),
    }


def load_all():
    data, cycles, rows = {}, {}, []
    for h in HOUSES:
        df, info = house_frame(h)
        cyc = detect_cycles(df["wm"], df["valid"])
        cyc["house"] = h
        data[h], cycles[h] = df, cyc
        rows.append(house_summary(h, df, cyc, info))
    return data, cycles, pd.DataFrame(rows).set_index("house")


# ---------------------------------------------------------------- house 4 check
def house4_overlap(df: pd.DataFrame) -> dict:
    v = df["valid"]
    a4, a5 = (df["Appliance4"] >= THETA) & v, (df["Appliance5"] >= THETA) & v
    c4, c5 = detect_cycles(df["Appliance4"], v), detect_cycles(df["Appliance5"], v)
    return {"on_minutes_A4": int(a4.sum()), "on_minutes_A5": int(a5.sum()), "both_on": int((a4 & a5).sum()),
            "both_on_pct_of_any_on": round(100 * (a4 & a5).sum() / (a4 | a5).sum(), 2),
            "cycles_A4": len(c4), "cycles_A5": len(c5),
            "cycles_summed_target": len(detect_cycles(df["wm"], v))}


# ---------------------------------------------------------------- patterns
def patterns(cycles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of each house's cycle starts by local hour and by weekday."""
    hod, dow = {}, {}
    for h, c in cycles.items():
        t = pd.DatetimeIndex(c.start).tz_convert(LOCAL_TZ)
        hod[h] = pd.Series(t.hour).value_counts(normalize=True).reindex(range(24), fill_value=0)
        dow[h] = pd.Series(t.dayofweek).value_counts(normalize=True).reindex(range(7), fill_value=0)
    return pd.DataFrame(hod), pd.DataFrame(dow)


def weekday_exposure(data: dict) -> pd.DataFrame:
    """Valid days per weekday, to normalise weekday counts (outages are not uniform)."""
    out = {}
    for h, df in data.items():
        dv = daily_validity(df)
        out[h] = pd.Series(pd.DatetimeIndex(dv.index[dv >= 0.9].astype(str)).dayofweek).value_counts().reindex(range(7), fill_value=0)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- plots
def plot_patterns(hod: pd.DataFrame, dow: pd.DataFrame, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    q = hod * 100
    a.fill_between(q.index, q.quantile(0.25, axis=1), q.quantile(0.75, axis=1), color=S.BLUE, alpha=0.18, lw=0,
                   label="middle 50% of houses")
    a.plot(q.index, q.median(axis=1), color=S.BLUE, lw=2, label="median house")
    a.set_xticks(range(0, 24, 2))
    a.set_xlabel("hour of day the cycle starts (UK local time)")
    a.set_ylabel("% of the house's cycles")
    a.set_title("Cycle starts by hour of day (19 houses)")
    a.legend(loc="upper left")
    p = dow * 100
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    b.bar(range(7), p.median(axis=1), color=S.BLUE, width=0.6)
    b.errorbar(range(7), p.median(axis=1), yerr=[p.median(axis=1) - p.quantile(0.25, axis=1),
               p.quantile(0.75, axis=1) - p.median(axis=1)], fmt="none", ecolor=S.INK2, lw=1, capsize=3)
    b.axhline(100 / 7, color=S.GREY, lw=1, ls="--")
    b.text(6.4, 100 / 7, "even\n(14.3%)", va="center", ha="left", fontsize=8, color=S.INK2)
    b.set_xticks(range(7), days)
    b.set_ylabel("% of the house's cycles")
    b.set_title("Cycle starts by weekday (median, IQR)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_house(allc: pd.DataFrame, path):
    order = allc.groupby("house").energy_kwh.median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    specs = [("energy_kwh", "Energy per cycle (kWh)"), ("minutes", "Duration (minutes)"), ("peak_w", "Peak 1-minute power (W)")]
    for ax, (col, ttl) in zip(axes, specs):
        vals = [allc.loc[allc.house == h, col].to_numpy() for h in order]
        bp = ax.boxplot(vals, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops={"color": S.INK, "lw": 1.5}, whiskerprops={"color": S.INK2},
                        capprops={"color": S.INK2}, boxprops={"edgecolor": S.INK2})
        for patch, h in zip(bp["boxes"], order):
            patch.set_facecolor(S.ORANGE if h in TEST else S.BLUE)
            patch.set_alpha(0.55)
        ax.set_title(ttl)
        if col == "minutes":
            ax.set_xlim(0, 260)
    axes[0].set_yticks(range(1, len(order) + 1), [f"House {h}" + (" (test)" if h in TEST else "") for h in order])
    fig.suptitle("Washing-machine cycles compared across houses (boxes: middle 50% of cycles; whiskers: 1.5 IQR; "
                 "sorted by median energy)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pick_representative_day(h: int, df: pd.DataFrame, cyc: pd.DataFrame) -> pd.Timestamp:
    """A fully valid local day with exactly one cycle whose energy and duration are closest
    to the house's medians."""
    loc_day = pd.DatetimeIndex(cyc.start).tz_convert(LOCAL_TZ).normalize()
    per_day = pd.Series(loc_day).value_counts()
    dv = daily_validity(df)
    ok = {pd.Timestamp(str(d)).tz_localize(LOCAL_TZ) for d in dv.index[dv >= 0.99]}
    one = cyc[[d in ok and per_day[d] == 1 for d in loc_day]].copy()
    score = (one.energy_kwh / cyc.energy_kwh.median() - 1).abs() + (one.minutes / cyc.minutes.median() - 1).abs()
    return pd.Timestamp(one.loc[score.idxmin(), "start"]).tz_convert(LOCAL_TZ).normalize()


def plot_day(h: int, df: pd.DataFrame, cyc: pd.DataFrame, day: pd.Timestamp, path):
    t0, t1 = day, day + pd.Timedelta(days=1)
    w = df.loc[t0:t1]
    loc = w.index.tz_convert(LOCAL_TZ)
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a.plot(loc, w["agg"], color=S.GREY, lw=1)
    a.set_ylabel("Aggregate (W)")
    a.set_title(f"House {h}, {day:%a %d %b %Y}: whole-home aggregate (top) and washing machine (bottom)")
    b.plot(loc, w["wm"], color=S.BLUE, lw=1.3)
    b.axhline(THETA, color=S.ORANGE, lw=1, ls="--")
    b.text(loc[5], THETA, f" θ = {THETA} W", color=S.ORANGE, va="bottom", fontsize=8)
    for _, c in cyc[(cyc.start >= t0) & (cyc.start < t1)].iterrows():
        for ax in (a, b):
            ax.axvspan(c.start.tz_convert(LOCAL_TZ), c.end.tz_convert(LOCAL_TZ), color=S.BLUE, alpha=0.08, lw=0)
        b.annotate(f"cycle: {c.minutes} min, {c.energy_kwh:.2f} kWh, peak {c.peak_w:,.0f} W",
                   xy=(c.end.tz_convert(LOCAL_TZ), c.peak_w * 0.9), xytext=(8, 0), textcoords="offset points",
                   fontsize=8, color=S.INK)
    b.set_ylabel("Washing machine (W)")
    b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    b.set_xlabel("UK local time")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cycles(examples: list[tuple[int, pd.Series, pd.Series, pd.Series]], path):
    fig, axes = plt.subplots(1, len(examples), figsize=(4.2 * len(examples), 3.6), sharey=True)
    for ax, (h, w, agg, c) in zip(axes, examples):
        tmin = (w.index - c.start).total_seconds() / 60
        ax.plot(tmin, agg.to_numpy(), color=S.LIGHT, lw=1, label="aggregate")
        ax.plot(tmin, w.to_numpy(), color=S.BLUE, lw=1.4, label="washing machine")
        ax.axvspan(0, c.minutes - 1, color=S.BLUE, alpha=0.08, lw=0)
        ax.axhline(THETA, color=S.ORANGE, lw=1, ls="--")
        ax.set_title(f"House {h}: {c.minutes} min, {c.energy_kwh:.2f} kWh", fontsize=9)
        ax.set_xlabel("minutes from cycle start")
    axes[0].set_ylabel("power (W)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Detected cycles (shaded = cycle found by the rule; dashed = θ). Heater bursts ~2 kW, "
                 "motor/spin 100-500 W, pauses bridged", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    data, cycles, summ = load_all()
    allc = pd.concat(cycles.values(), ignore_index=True)
    summ.to_csv(OUT_DIR / "eda_house_summary.csv")
    allc.to_csv(OUT_DIR / "eda_cycles.csv", index=False)
    hod, dow = patterns(cycles)
    exp = weekday_exposure(data)
    dow_norm = (dow.mul(1, axis=0) / exp.div(exp.sum()))  # starts share / valid-day share
    hod.to_csv(OUT_DIR / "eda_hour_of_day.csv")
    dow.to_csv(OUT_DIR / "eda_day_of_week.csv")
    plot_patterns(hod, dow, FIG_DIR / "s2_usage_patterns.png")
    plot_cross_house(allc, FIG_DIR / "s2_cross_house_cycles.png")
    # (i) representative day: a development house
    h_day = 1
    day = pick_representative_day(h_day, data[h_day], cycles[h_day])
    plot_day(h_day, data[h_day], cycles[h_day], day, FIG_DIR / "s2_representative_day.png")
    # (ii) detected cycles: the median-energy cycle of three development houses
    ex = []
    for h in (1, 5, 19):
        c = cycles[h]
        c = c.iloc[(c.energy_kwh - c.energy_kwh.median()).abs().argsort().iloc[0]]
        sl = slice(c.start - pd.Timedelta(minutes=15), c.end + pd.Timedelta(minutes=15))
        ex.append((h, data[h].loc[sl, "wm"], data[h].loc[sl, "agg"], c))
    plot_cycles(ex, FIG_DIR / "s2_detected_cycles.png")
    extra = {"house4": house4_overlap(data[4]), "representative_day": [h_day, str(day.date())],
             "weekday_share_vs_exposure_median": dow_norm.median(axis=1).round(3).to_dict(),
             "params": {"theta_w": THETA, "g_min": G, "l_min": L, "heater_w": HEATER_W, "max_cycle_min": MAX_CYCLE}}
    (OUT_DIR / "eda_extra.json").write_text(json.dumps(extra, indent=2, default=str))
    return {"summary": summ, "extra": extra, "hod": hod, "dow": dow}


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(r["summary"].to_string())
    print(json.dumps(r["extra"], indent=1, default=str))
    print((r["hod"].median(axis=1) * 100).round(1).to_dict())
    print((r["dow"].median(axis=1) * 100).round(1).to_dict())
