"""Section 3 figures and failure analysis on the TEST houses (after the single test run).

Pre-declared selection rules (no cherry-picking):
  * actual-vs-predicted plot: test house with the median NDE_day; among its counted days with
    at least one actual cycle, the day with the median absolute daily error; window = that UK
    day +/- 12 h (48 h).
  * failure cases: the test day with the largest OVER-estimate and the test day with the largest
    UNDER-estimate of daily energy (all counted days of the 3 test houses).
  * failure causes are checked with data (other appliance channels, cycle type), not assumed.
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import style as S
from src.appliances import name as app_name
from src.clean_data import load_house
from src.config import APPLIANCE_COLS, FIG_DIR, LOCAL_TZ, WM_CHANNELS
from src.cycles import HEATER_W, THETA, detect_cycles
from src.final_test import FINAL_OOF
from src.loho import MET_DIR
from src.metrics import daily

OTHER_ON_W = 1000  # another appliance counts as "running a heating-size load" at >= 1 kW


def grid(g: pd.DataFrame) -> pd.DataFrame:
    """Evaluable rows of one house put back on a continuous minute grid (NaN elsewhere)."""
    idx = pd.date_range(g.index.min(), g.index.max(), freq="min")
    return g.reindex(idx)


def day_errors(g: pd.DataFrame) -> pd.DataFrame:
    d = daily(g["y"], g["model"])
    d = d[d.counted].copy()
    d["err"] = d.est - d.act
    return d


def _actual_vs_pred(ax, w):
    """Actual = filled grey area with a dark outline; predicted = orange line on top."""
    loc = w.index.tz_convert(LOCAL_TZ)
    ax.fill_between(loc, 0, w["y"].to_numpy(), color=S.LIGHT, lw=0, step="mid", label="actual washing machine")
    ax.plot(loc, w["y"].to_numpy(), color=S.INK2, lw=0.8, drawstyle="steps-mid")
    ax.plot(loc, w["model"].to_numpy(), color=S.ORANGE, lw=1.4, drawstyle="steps-mid", label="predicted")
    ax.set_ylabel("Washing machine (W)")
    ax.legend(loc="upper right")


def window_plot(h, g, agg, t0, t1, path, title, other=None, zoom=None):
    w, a = g.loc[t0:t1], agg.loc[t0:t1]
    nrow = 3 if other is not None else 2
    ratios = [1, 1.4] + ([1.2] if other is not None else [])
    if zoom is not None:
        fig = plt.figure(figsize=(12, 2.6 * (nrow + 1) + 0.9))
        gs = fig.add_gridspec(nrow + 1, 1, height_ratios=ratios + [1.6], hspace=0.35)
        axes = [fig.add_subplot(gs[0])]
        axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(1, nrow)]
        axz = fig.add_subplot(gs[nrow])
    else:
        fig, axes = plt.subplots(nrow, 1, figsize=(12, 2.6 * nrow + 0.6), sharex=True,
                                 gridspec_kw={"height_ratios": ratios})
    loc = lambda ix: ix.tz_convert(LOCAL_TZ)
    axes[0].plot(loc(a.index), a.to_numpy(), color=S.GREY, lw=0.9)
    axes[0].set_ylabel("Aggregate (W)")
    axes[0].set_title(title)
    _actual_vs_pred(axes[1], w)
    if other is not None:
        o = other.loc[t0:t1]
        cols = [c for c in o.columns if o[c].max() >= OTHER_ON_W][:4]
        palette = [S.BLUE, S.AQUA, S.VIOLET, S.MAGENTA]
        for c, col in zip(cols, palette):
            axes[2].plot(loc(o.index), o[c].to_numpy(), lw=1.1, color=col, label=app_name(h, c))
        axes[2].set_ylabel("Other appliances (W)")
        axes[2].legend(loc="upper right", fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=LOCAL_TZ))
    axes[-1].set_xlabel("UK local time")
    if zoom is not None:
        z0, z1 = zoom
        for ax in axes[:2]:
            ax.axvspan(loc(pd.DatetimeIndex([z0]))[0], loc(pd.DatetimeIndex([z1]))[0], color=S.BLUE, alpha=0.07, lw=0)
            plt.setp(ax.get_xticklabels(), visible=ax is axes[-1])
        wz = g.loc[z0:z1]
        _actual_vs_pred(axz, wz)
        axz.plot(loc(agg.loc[z0:z1].index), agg.loc[z0:z1].to_numpy(), color=S.GREY, lw=0.8, alpha=0.6,
                 label="aggregate (for reference)")
        axz.legend(loc="upper right")
        axz.set_title("Zoom on the wash (shaded window above)")
        axz.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
        axz.set_xlabel("UK local time")
        axz.set_ylabel("Power (W)")
    else:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(res: pd.DataFrame, path):
    houses = sorted(res.house.unique())
    fig, axes = plt.subplots(1, len(houses), figsize=(4.2 * len(houses), 4), sharex=True, sharey=True)
    hi = 0
    for ax, h in zip(axes, houses):
        d = day_errors(res[res.house == h])
        hi = max(hi, d.act.max(), d.est.max())
        ax.scatter(d.act, d.est, s=10, color=S.BLUE, alpha=0.55, lw=0)
        ax.set_title(f"Test house {h}: {len(d)} days")
        ax.set_xlabel("actual washing-machine kWh / day")
    for ax in axes:
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color=S.GREY, ls="--", lw=1)
    axes[0].set_ylabel("estimated kWh / day")
    fig.suptitle("Daily washing-machine energy, estimated vs actual (dashed = perfect)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def failure_stats(h: int, g: pd.DataFrame) -> dict:
    """Systematic checks of the suspected causes for one test house."""
    gg = grid(g)
    valid = gg["y"].notna()
    y, p = gg["y"].fillna(0), gg["model"].fillna(0)
    raw = load_house(h).reindex(gg.index)
    others = [c for c in APPLIANCE_COLS if c not in WM_CHANNELS[h]]
    fp = valid & (p >= THETA) & (y < THETA)               # predicted ON, machine OFF
    hot = raw[others].ge(OTHER_ON_W)
    fp_share = {app_name(h, c): round(100 * float((fp & hot[c]).sum() / max(fp.sum(), 1)), 1) for c in others}
    fp_share = dict(sorted(((k, v) for k, v in fp_share.items() if v >= 1), key=lambda kv: -kv[1]))
    fp_any = round(100 * float((fp & hot.any(axis=1)).sum() / max(fp.sum(), 1)), 1)
    # cycles with / without the heater: how many were found, how much energy was estimated
    cyc = detect_cycles(gg["y"], valid)
    pred_on = (p >= THETA).to_numpy()
    rows = []
    for _, c in cyc.iterrows():
        sl = slice(c.i0, c.i1 + 1)
        rows.append({"heater": bool(c.heater), "found": bool(pred_on[sl].any()),
                     "act_kwh": c.energy_kwh, "est_kwh": float(p.to_numpy()[sl].sum() / 60000)})
    cy = pd.DataFrame(rows)
    by = cy.groupby("heater").agg(cycles=("found", "size"), found_pct=("found", "mean"),
                                  act_kwh=("act_kwh", "sum"), est_kwh=("est_kwh", "sum"))
    by["found_pct"] = (100 * by.found_pct).round(1)
    by["energy_captured_pct"] = (100 * by.est_kwh / by.act_kwh).round(1)
    return {
        "total_actual_kwh": round(float(y[valid].sum() / 60000), 1),
        "total_estimated_kwh": round(float(p[valid].sum() / 60000), 1),
        "false_on_minutes": int(fp.sum()),
        "false_on_with_other_load_ge1kW_pct": fp_any,
        "false_on_by_appliance_pct": fp_share,
        "cycles_by_heater": {("heater" if k else "no heater"): v for k, v in by.round(2).to_dict("index").items()},
    }


def main():
    res = pd.read_parquet(FINAL_OOF)
    ph = pd.read_csv(MET_DIR / "test_per_house.csv")
    m = ph[ph.variant == "model"].set_index("house")["nde_day"].sort_values()
    h_med = int(m.index[len(m) // 2])
    out = {"median_house": h_med}

    # ---- actual vs predicted (median house, median-error washing day)
    g = res[res.house == h_med]
    gg = grid(g)
    cyc = detect_cycles(gg["y"], gg["y"].notna())
    wash_days = set(pd.DatetimeIndex(cyc.start).tz_convert(LOCAL_TZ).normalize())
    d = day_errors(g)
    d = d[d.index.isin(wash_days)]
    d["abs"] = d.err.abs()
    day = d.sort_values("abs").index[len(d) // 2]
    agg = load_house(h_med)["Aggregate"]
    # zoom: the day's actual cycles, padded by 30 minutes (display only; the window rule is unchanged)
    dc = cyc[(cyc.start >= day.tz_convert("UTC")) & (cyc.start < (day + pd.Timedelta(days=1)).tz_convert("UTC"))]
    zoom = (dc.start.min() - pd.Timedelta(minutes=30), dc.end.max() + pd.Timedelta(minutes=30))
    window_plot(h_med, gg, agg, day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=36),
                FIG_DIR / "s3_actual_vs_predicted.png",
                f"Test house {h_med}, {day:%a %d %b %Y} ± 12 h (median-error washing day of the median test house): "
                f"actual {d.loc[day, 'act']:.2f} kWh, estimated {d.loc[day, 'est']:.2f} kWh", zoom=zoom)
    out["actual_vs_predicted_day"] = str(day.date())

    # ---- daily scatter
    scatter_plot(res, FIG_DIR / "s3_daily_energy_scatter.png")

    # ---- failure cases: largest over- and under-estimate
    alld = pd.concat([day_errors(res[res.house == h]).assign(house=h) for h in sorted(res.house.unique())])
    alld = alld.rename_axis("day").reset_index()
    for kind, row in (("over", alld.loc[alld.err.idxmax()]), ("under", alld.loc[alld.err.idxmin()])):
        h, day = int(row.house), row.day
        gh = grid(res[res.house == h])
        raw = load_house(h)
        others = raw[[c for c in APPLIANCE_COLS if c not in WM_CHANNELS[h]]]
        window_plot(h, gh, raw["Aggregate"], day, day + pd.Timedelta(days=1),
                    FIG_DIR / f"s3_failure_{kind}estimate.png",
                    f"Failure case ({kind}-estimate): test house {h}, {day:%a %d %b %Y}: "
                    f"actual {row.act:.2f} kWh, estimated {row.est:.2f} kWh", other=others)
        out[f"failure_{kind}"] = {"house": h, "day": str(day.date()), "actual_kwh": round(float(row.act), 2),
                                  "estimated_kwh": round(float(row.est), 2)}

    # ---- systematic cause checks
    out["failure_stats"] = {int(h): failure_stats(int(h), res[res.house == h]) for h in sorted(res.house.unique())}
    (MET_DIR / "test_failure_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=str))
