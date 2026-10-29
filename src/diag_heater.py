"""EXP-01: heater-block diagnostic (post-hoc, diagnostic only; see experiments/EXP-01_heater_diagnostic.md).

Reads the saved leave-one-house-out predictions of the selected v1 configuration on the 16
DEVELOPMENT houses. Nothing is retrained and no prediction is changed.

For every minute inside a true heating block (actual WM >= 1500 W):
    y - pred = (y - w) + w * [p <= T]
    = regressor error + gating loss         (exact; the two parts sum to the total loss)
where w = regressor watts, p = classifier P(ON), T = 0.4 the v1 gate threshold.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features as F
from src import style as S
from src.clean_data import load_house
from src.config import APPLIANCE_COLS, CFG, FIG_DIR, OUT_DIR, WM_CHANNELS
from src.cycles import HEATER_W
from src.loho import OOF_DIR

T, R = 0.4, 800
MIN_LEN = 3
OTHER_W = 1000
STEP_W = 1000
NPOS = 30
DIAG_DIR = OUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
LEN_BINS = [(3, 5), (6, 10), (11, 20), (21, 10_000)]


def heater_blocks(y: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    hot = (y >= HEATER_W) & valid
    # bridge single-minute dips between two heater minutes (the dip minute must be valid)
    dip = np.zeros_like(hot)
    dip[1:-1] = ~hot[1:-1] & hot[:-2] & hot[2:] & valid[1:-1]
    m = hot | dip
    e = np.diff(np.r_[0, m.astype(np.int8), 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
    return [(a, b) for a, b in zip(st, en) if b - a + 1 >= MIN_LEN]


def house_blocks(h: int, oof: pd.DataFrame):
    g = oof[oof.house == h]
    idx = pd.date_range(g.index.min(), g.index.max(), freq="min")
    g = g.reindex(idx)
    valid = g["y"].notna().to_numpy()
    y = g["y"].fillna(0).to_numpy()
    p = g[f"p_on_r{R}"].fillna(0).to_numpy()
    w = g[f"power_on_r{R}"].fillna(0).to_numpy()
    pred = np.where(p > T, w, 0.0)
    x = F.load(h, columns=["x"])["x"].reindex(idx).to_numpy()
    raw = load_house(h).reindex(idx)
    others = [c for c in APPLIANCE_COLS if c not in WM_CHANNELS[h]]
    other_hot = raw[others].ge(OTHER_W).any(axis=1).to_numpy()

    rows, minutes = [], []
    for a, b in heater_blocks(y, valid):
        sl = slice(a, b + 1)
        before = x[max(a - 10, 0):a]
        step = np.nanmedian(x[sl]) - (np.nanmedian(before) if np.isfinite(before).any() else np.nan)
        yy, ww, pp, pr = y[sl], w[sl], p[sl], pred[sl]
        rows.append({
            "house": h, "start": idx[a], "minutes": b - a + 1,
            "act_wh": yy.sum() / 60, "captured_wh": pr.sum() / 60,
            "gating_loss_wh": (ww * (pp <= T)).sum() / 60, "regressor_err_wh": (yy - ww).sum() / 60,
            "gate_open_share": float((pp > T).mean()), "mean_p": float(pp.mean()),
            "other_load_overlap": bool(other_hot[sl].any()), "step_visible": bool(step >= STEP_W),
            "agg_step_w": float(step),
        })
        n = b - a + 1
        minutes.append(pd.DataFrame({"house": h, "pos_start": np.arange(n), "pos_end": np.arange(n) - (n - 1),
                                     "y": yy, "w": ww, "p": pp, "pred": pr}))
    return pd.DataFrame(rows), (pd.concat(minutes) if minutes else pd.DataFrame())


def split_table(blocks: pd.DataFrame, by) -> pd.DataFrame:
    g = blocks.groupby(by)
    t = g[["act_wh", "captured_wh", "gating_loss_wh", "regressor_err_wh"]].sum()
    out = pd.DataFrame({
        "blocks": g.size(),
        "actual_kwh": (t.act_wh / 1000).round(1),
        "captured_pct": (100 * t.captured_wh / t.act_wh).round(1),
        "gating_loss_pct": (100 * t.gating_loss_wh / t.act_wh).round(1),
        "regressor_err_pct": (100 * t.regressor_err_wh / t.act_wh).round(1),
        "gate_open_share": g.gate_open_share.mean().round(3),
    })
    return out


def plots(blocks: pd.DataFrame, mins: pd.DataFrame, per_house: pd.DataFrame):
    # 1. block-aligned profiles
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col", gridspec_kw={"height_ratios": [1.6, 1]})
    for j, (col, lab) in enumerate((("pos_start", "minutes since heating block started"),
                                    ("pos_end", "minutes before heating block ends"))):
        m = mins[mins[col].abs() < NPOS].groupby(col)
        a = m[["y", "w", "pred", "p"]].mean()
        n = m.size()
        a = a[n >= 50]
        ax[0, j].plot(a.index, a.y, color=S.INK, lw=2, label="actual WM power")
        ax[0, j].plot(a.index, a.w, color=S.BLUE, lw=1.8, label="regressor output (before gate)")
        ax[0, j].plot(a.index, a.pred, color=S.ORANGE, lw=1.8, label="final prediction (after gate)")
        ax[0, j].set_ylabel("mean power (W)")
        ax[1, j].plot(a.index, a.p, color=S.VIOLET, lw=1.8, label="mean P(ON)")
        open_share = m.apply(lambda d: (d.p > T).mean())[n >= 50]
        ax[1, j].plot(open_share.index, open_share, color=S.AQUA, lw=1.8, label="share of minutes with gate open")
        ax[1, j].axhline(T, color=S.GREY, ls="--", lw=1)
        ax[1, j].set_ylim(0, 1)
        ax[1, j].set_xlabel(lab)
    ax[0, 0].legend(loc="lower left", fontsize=8)
    ax[1, 0].legend(loc="lower left", fontsize=8)
    ax[0, 0].set_title("Aligned on block start")
    ax[0, 1].set_title("Aligned on block end")
    fig.suptitle(f"EXP-01: average over {len(blocks):,} true heating blocks in 16 development houses "
                 "(leave-one-house-out predictions of v1)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_heater_profile.png", bbox_inches="tight")
    plt.close(fig)

    # 2. per-house energy split
    ph = per_house.sort_values("captured_pct")
    fig, axh = plt.subplots(figsize=(9, 5.5))
    yv = np.arange(len(ph))
    cap, gate, reg = ph.captured_pct.to_numpy(), ph.gating_loss_pct.to_numpy(), ph.regressor_err_pct.to_numpy()
    axh.barh(yv, cap, color=S.BLUE, height=0.65, label="captured")
    axh.barh(yv, gate, left=cap, color=S.ORANGE, height=0.65, label="lost: gate closed")
    axh.barh(yv, np.clip(reg, 0, None), left=cap + gate, color=S.GREY, height=0.65,
             label="lost: regressor too low (net)")
    neg = reg < 0
    if neg.any():
        axh.barh(yv[neg], reg[neg], color=S.MAGENTA, height=0.65, label="regressor too high (net)")
    axh.axvline(100, color=S.INK, lw=1)
    axh.set_yticks(yv, [f"House {h}" for h in ph.index])
    axh.set_xlabel("% of actual heating-block energy")
    axh.set_title("Where heating-block energy goes, per development house")
    axh.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_heater_split.png", bbox_inches="tight")
    plt.close(fig)

    # 3. P(ON) histogram + regressor vs actual when gate open
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.hist(mins.p, bins=np.linspace(0, 1, 41), color=S.VIOLET, alpha=0.8)
    a1.axvline(T, color=S.INK, ls="--", lw=1)
    a1.text(T, a1.get_ylim()[1] * 0.95, f" gate threshold {T}", va="top", fontsize=8)
    a1.set_xlabel("classifier P(ON) during true heating minutes")
    a1.set_ylabel("minutes")
    a1.set_title(f"{100 * (mins.p <= T).mean():.0f}% of heating minutes have the gate closed")
    op = mins[mins.p > T]
    hb = a2.hexbin(op.y, op.w, gridsize=40, cmap="Blues", mincnt=1, bins="log")
    lim = max(op.y.max(), op.w.max())
    a2.plot([0, lim], [0, lim], color=S.GREY, ls="--", lw=1)
    a2.set_xlabel("actual WM power (W)")
    a2.set_ylabel("regressor output (W)")
    a2.set_title("Heating minutes with gate open: regressor vs actual")
    fig.colorbar(hb, ax=a2, label="minutes (log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_heater_p_and_power.png", bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    oof = pd.read_parquet(OOF_DIR / "lgb_twostage_cal.parquet",
                          columns=["house", "y", f"p_on_r{R}", f"power_on_r{R}"])
    assert set(oof.house.unique()) == set(CFG["dev_houses"]), "development houses only"
    bl, mn = [], []
    for h in CFG["dev_houses"]:
        b, m = house_blocks(h, oof)
        bl.append(b)
        mn.append(m)
    blocks, mins = pd.concat(bl, ignore_index=True), pd.concat(mn, ignore_index=True)
    blocks["len_bin"] = pd.cut(blocks.minutes, [a - 0.5 for a, _ in LEN_BINS] + [10_000.5],
                               labels=[f"{a}-{b}" if b < 10_000 else f">{a - 1}" for a, b in LEN_BINS])
    per_house = split_table(blocks, "house")
    blocks["all"] = "all"
    overall = split_table(blocks, "all")
    by_len = split_table(blocks, "len_bin")
    by_overlap = split_table(blocks, "other_load_overlap")
    by_step = split_table(blocks, "step_visible")
    blocks.to_csv(DIAG_DIR / "exp01_blocks.csv", index=False)
    for name, t in (("per_house", per_house), ("by_length", by_len), ("by_other_load", by_overlap),
                    ("by_step_visible", by_step)):
        t.to_csv(DIAG_DIR / f"exp01_{name}.csv")
    hm = mins
    pos = hm[hm.pos_start < 15].groupby("pos_start")[["y", "w", "pred", "p"]].mean().round(3)
    summary = {
        "overall": overall.reset_index(drop=True).iloc[0].to_dict(),
        "heating_minutes": int(len(hm)),
        "share_minutes_gate_closed": round(float((hm.p <= T).mean()), 3),
        "median_p_in_heating_minutes": round(float(hm.p.median()), 3),
        "regressor_when_gate_open": {
            "mean_actual_w": round(float(hm.loc[hm.p > T, "y"].mean()), 1),
            "mean_regressor_w": round(float(hm.loc[hm.p > T, "w"].mean()), 1)},
        "regressor_when_gate_closed": {
            "mean_actual_w": round(float(hm.loc[hm.p <= T, "y"].mean()), 1),
            "mean_regressor_w": round(float(hm.loc[hm.p <= T, "w"].mean()), 1)},
        "profile_first_15_min": pos.to_dict("index"),
    }
    (DIAG_DIR / "exp01_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plots(blocks, mins, per_house)
    return {"summary": summary, "per_house": per_house, "by_length": by_len,
            "by_other_load": by_overlap, "by_step_visible": by_step}


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in r["summary"].items() if k != "profile_first_15_min"}, indent=1, default=str))
    print(pd.DataFrame(r["summary"]["profile_first_15_min"]).T.to_string())
    for k in ("per_house", "by_length", "by_other_load", "by_step_visible"):
        print(f"\n== {k}\n{r[k].to_string()}")
