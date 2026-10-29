"""Per-house NDE_day of the selected model vs both baselines, on development (LOHO) and test
houses. Reads only saved metric tables (outputs/metrics/); no model is run here.

    python -m src.plots_summary
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src import style as S
from src.config import FIG_DIR
from src.loho import MET_DIR

SELECTED_DEV = "lgb_twostage_cal:hard_r800_t0.4"


def _panel(ax, rows: pd.DataFrame, title: str, ci: dict | None = None):
    """rows: index = house label, columns model / hour_of_week (NDE_day)."""
    y = range(len(rows))
    ax.axvline(1.0, color=S.GREY, ls="--", lw=1)
    for i, (h, r) in enumerate(rows.iterrows()):
        ax.plot([r.model, r.hour_of_week], [i, i], color=S.LIGHT, lw=1.2, zorder=1)
    if ci:
        for i, h in enumerate(rows.index):
            for col, colour in (("model", S.BLUE), ("hour_of_week", S.ORANGE)):
                lo, hi = ci[col][h]
                ax.plot([lo, hi], [i, i], color=colour, lw=2.4, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.scatter(rows.hour_of_week, list(y), s=36, color=S.ORANGE, zorder=3, label="hour-of-week baseline",
               edgecolor="white", linewidth=1)
    ax.scatter(rows.model, list(y), s=36, color=S.BLUE, zorder=4, label="selected model",
               edgecolor="white", linewidth=1)
    ax.set_yticks(list(y), [f"House {h}" for h in rows.index])
    ax.set_xlabel("NDE_day (lower is better; dashed line = always-off baseline, 1.0)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)


def main(path=None):
    path = path or FIG_DIR / "s3_per_house_nde.png"
    dev = pd.read_csv(MET_DIR / "dev_all_variants_per_house.csv")
    d = dev[dev.variant.isin([SELECTED_DEV, "hour_of_week:pred"])].pivot(index="house", columns="variant", values="nde_day")
    d = d.rename(columns={SELECTED_DEV: "model", "hour_of_week:pred": "hour_of_week"}).sort_values("model", ascending=False)

    test = pd.read_csv(MET_DIR / "test_per_house.csv")
    t = test[test.variant.isin(["model", "hour_of_week"])].pivot(index="house", columns="variant", values="nde_day")
    t = t.sort_values("model", ascending=False)
    boot = json.loads((MET_DIR / "test_results.json").read_text())["bootstrap"]
    ci = {c: {int(h): v for h, v in boot[c]["per_house_nde_ci"].items()} for c in ("model", "hour_of_week")}

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.6), gridspec_kw={"height_ratios": [len(d), len(t) + 1.5], "hspace": 0.55})
    _panel(axes[0], d, f"Development houses (leave-one-house-out), n = {len(d)}\n"
                       f"macro mean: model {d.model.mean():.3f}, hour-of-week {d.hour_of_week.mean():.3f}")
    _panel(axes[1], t, f"Locked test houses (single run), n = {len(t)}\n"
                       f"macro mean: model {t.model.mean():.3f}, hour-of-week {t.hour_of_week.mean():.3f}", ci=ci)
    hi = max(d.max().max(), max(v[1] for v in ci["hour_of_week"].values())) * 1.05
    for ax in axes:
        ax.set_xlim(0, hi)
    axes[1].legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor="none")
    axes[1].text(0.0, -0.62, "Shaded bars: 95% day-block bootstrap interval (uncertainty within each house).",
                 transform=axes[1].transAxes, fontsize=8, color=S.INK2)
    fig.suptitle("Daily washing-machine energy error per house: selected model vs baselines",
                 x=0.01, y=0.995, ha="left", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
