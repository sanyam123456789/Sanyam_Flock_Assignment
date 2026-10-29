"""Section 1 figures: aggregate + two appliance channels, before vs after cleaning.

Before = raw readings exactly as recorded (both parts, raw timestamps = UK local clock).
After  = cleaned 1-minute series (stored in UTC, shown in UK local time so both columns
         share one time axis). Orange = short gap filled; grey band = flagged outage.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, FIG_DIR, LOCAL_TZ
from src.raw_house1 import COLS, IAM_MAX_W, load_part

CHANNELS = [("Aggregate", "Aggregate"), ("Appliance5", "Washing machine (Appliance5)"),
            ("Appliance1", "Fridge (Appliance1)")]
RAW_C, CLEAN_C, FILL_C, OUT_C = "#8a8986", "#2a78d6", "#eb6834", "#d9d8d4"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.6, "figure.dpi": 120,
})


def load_raw_local() -> pd.DataFrame:
    """Raw readings, both parts, index = raw timestamp (UK local clock, naive)."""
    parts = []
    for i in (1, 2):
        d = load_part(i)[["Unix"] + COLS]
        d["part"] = i
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d.index = pd.to_datetime(d.pop("Unix"), unit="s")
    return d.sort_index(kind="stable")


def load_clean_local() -> pd.DataFrame:
    c = pd.read_parquet(CACHE_DIR / "raw_house1_clean_1min.parquet")
    c.index = c.index.tz_convert(LOCAL_TZ).tz_localize(None)
    return c


def shade_outages(ax, flag: pd.Series):
    o = flag.eq(2).to_numpy()
    if not o.any():
        return
    e = np.diff(np.r_[0, o.astype(int), 0])
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        ax.axvspan(flag.index[s], flag.index[t - 1] + pd.Timedelta(minutes=1), color=OUT_C, lw=0, zorder=0)


def overview(raw: pd.DataFrame, clean: pd.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True)
    for r, (c, name) in enumerate(CHANNELS):
        before = raw[c].resample("D").max()
        after = clean[c].resample("D").max()
        valid = clean[f"{c}_flag"].ne(2).resample("D").mean()
        ymax = np.nanmax(after.to_numpy()) * 1.15
        for k, (ax, s, col, ttl) in enumerate(((axes[r, 0], before, RAW_C, "Before: raw readings"),
                                               (axes[r, 1], after, CLEAN_C, "After: cleaned 1-minute"))):
            ax.plot(s.index, s.to_numpy(), color=col, lw=1)
            if k == 1:
                bad = valid[valid < 0.5].index
                for d in bad:
                    ax.axvspan(d, d + pd.Timedelta(days=1), color=OUT_C, lw=0, zorder=0)
            if k == 0 and np.nanmax(s.to_numpy()) > ymax:
                n_off = int((s > ymax).sum())
                ax.set_ylim(0, ymax)
                ax.annotate(f"{n_off} days peak off-scale\n(max {np.nanmax(s.to_numpy()):,.0f} W)",
                            xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top", color=INK, fontsize=8)
            else:
                ax.set_ylim(0, ymax)
            if r == 0:
                ax.set_title(ttl, color=INK, fontsize=10, loc="left")
            if k == 0:
                ax.set_ylabel(f"{name}\ndaily max (W)")
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Raw House 1, whole period: daily maximum power before vs after cleaning "
                 "(grey band = day with < 50% valid minutes)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_overview_before_after.png", bbox_inches="tight")
    plt.close(fig)


def zoom(raw: pd.DataFrame, clean: pd.DataFrame, start: str, end: str, fname: str, title: str):
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    rw, cw = raw.loc[t0:t1], clean.loc[t0:t1]
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True)
    for r, (c, name) in enumerate(CHANNELS):
        after = cw[c]
        ymax = max(np.nanmax(after.to_numpy()) if after.notna().any() else 100, 100) * 1.15
        a0, a1 = axes[r]
        b = rw[c].dropna()
        # break the line where readings are > 60 s apart so gaps stay visible
        brk = b.index.to_series().diff().gt(pd.Timedelta(seconds=60)).to_numpy()
        yb = b.to_numpy().copy()
        yb[brk] = np.nan
        a0.plot(b.index, yb, color=RAW_C, lw=0.8)
        a0.scatter(b.index, b.to_numpy(), color=RAW_C, s=2)
        off = b[b > ymax]
        if len(off):
            a0.scatter(off.index, np.full(len(off), ymax * 0.97), marker="^", s=30, color=INK, zorder=5)
            a0.annotate(f"{len(off)} readings off-scale (max {off.max():,.0f} W)", xy=(0.99, 0.86),
                        xycoords="axes fraction", ha="right", va="top", fontsize=8, color=INK)
        shade_outages(a1, cw[f"{c}_flag"])
        a1.plot(after.index, after.to_numpy(), color=CLEAN_C, lw=1.2)
        filled = after[cw[f"{c}_flag"] == 1]
        if len(filled):
            a1.scatter(filled.index, filled.to_numpy(), s=12, color=FILL_C, zorder=5, label="short gap filled")
        for ax in (a0, a1):
            ax.set_ylim(0, ymax)
        a0.set_ylabel(f"{name}\npower (W)")
        if r == 0:
            a0.set_title("Before: raw readings (as recorded)", color=INK, fontsize=10, loc="left")
            a1.set_title("After: cleaned 1-minute (orange = short gap filled, grey = outage flagged)",
                         color=INK, fontsize=10, loc="left")
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.set_xlabel("UK local time")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def pick_spike_window(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.Timestamp:
    """Day with the most impossible IAM readings on the plotted appliances, preferring days
    when the washing machine actually ran (so the plot also shows a real cycle)."""
    bad = ((raw["Appliance5"] > IAM_MAX_W) | (raw["Appliance1"] > IAM_MAX_W)).resample("D").sum()
    wm_on = (clean["Appliance5"] > 1000).resample("D").sum()
    score = bad.where(wm_on.reindex(bad.index).fillna(0) > 5, 0)
    return score.idxmax()


def pick_gap_window(clean: pd.DataFrame) -> pd.Timestamp:
    """Centre a 4-hour window on a typical 10-30 min aggregate dropout that has the most
    filled short gaps (any channel) within +/- 2 h. Returns the window start."""
    from src.raw_house1 import nan_runs
    f = clean[[c for c in clean.columns if c.endswith("_flag")]].eq(1).any(axis=1).to_numpy()
    agg = clean["Aggregate"].where(clean["Aggregate_flag"] != 2)
    runs = nan_runs(agg)
    runs = runs[(runs.minutes >= 10) & (runs.minutes <= 30)]
    best, best_score = None, -1
    for st in runs.start:
        i = clean.index.get_loc(st)
        sc = f[max(i - 120, 0):i + 120].sum()
        if sc > best_score:
            best, best_score = i, sc
    return clean.index[max(best - 120, 0)]


def main():
    raw, clean = load_raw_local(), load_clean_local()
    overview(raw, clean)
    d = pick_spike_window(raw, clean)
    zoom(raw, clean, str(d), str(d + pd.Timedelta(days=1)), "s1_zoom_spikes_before_after.png",
         f"Raw House 1, {d:%d %b %Y}: impossible appliance readings (> 4 kW through a 13 A plug) removed")
    g = pick_gap_window(clean)
    zoom(raw, clean, str(g), str(g + pd.Timedelta(hours=4)), "s1_zoom_gaps_before_after.png",
         f"Raw House 1, {g:%d %b %Y %H:%M} + 4 h: short gaps filled (<= 5 min), longer gaps flagged as outages")
    print("figures:", "spike day", d.date(), "| gap window", g)


if __name__ == "__main__":
    main()
