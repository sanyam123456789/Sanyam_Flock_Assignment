"""EXP-03: step-paired heater-block fill (post-hoc, exploratory; see experiments/EXP-03_block_fill.md).

Post-processing on the saved leave-one-house-out predictions of v1 for the 16 DEVELOPMENT houses.
Nothing is retrained; test houses are never read. Appliance channels only label filled blocks.

In plain words: find heater-sized ON/OFF step pairs in the aggregate; where v1's P(ON) over the
block is high enough, raise the prediction to the step size (never above the aggregate).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features as F
from src import style as S
from src.appliances import name as app_name
from src.clean_data import load_house
from src.config import APPLIANCE_COLS, CFG, FIG_DIR, WM_CHANNELS
from src.cycles import HEATER_W, THETA
from src.diag_heater import DIAG_DIR, R, T, heater_blocks
from src.diag_lookalike import app_type
from src.loho import OOF_DIR
from src.metrics import score_frame, summarise

EDGE_MIN_W = 200      # W   a minute belongs to an edge if the aggregate moves by more than this
EDGE_MAX_MIN = 3      # min an edge may spread over at most 3 minutes
STEP_W = F.STEP_W     # W   heater-sized edge (1000 W)
MAX_GAP = 40          # min fall must start within 40 min of the rise (dev heating blocks p99 = 39)
MATCH = 0.3           #     |rise - drop| <= 30% of the larger
MIN_LEN = 3           # min shortest block
F_MAX = 3000          # W   cap on the fill value
OTHER_W = 1000        # W   another monitored load "running" (labelling only)
P_PRIMARY = 0.3
P_GRID = [0.2, 0.3, 0.4, 0.5]
V1_DEV_MACRO_NDE = 0.658


def edges(x: np.ndarray, sign: int):
    """Edges in one direction: runs of consecutive minutes moving > EDGE_MIN_W that way."""
    d = np.r_[np.nan, np.diff(x)] * sign
    m = np.nan_to_num(d, nan=0.0) > EDGE_MIN_W
    e = np.diff(np.r_[0, m.astype(np.int8), 0])
    st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
    cs = np.r_[0, np.cumsum(np.nan_to_num(d))]
    size = cs[en + 1] - cs[st]
    keep = (en - st + 1 <= EDGE_MAX_MIN) & (size >= STEP_W)
    return st[keep], en[keep], size[keep]


def pair_blocks(x: np.ndarray):
    """Non-overlapping (start, end, F, base) heater blocks from matched rise/fall pairs."""
    r0, r1, up = edges(x, +1)
    f0, f1, dn = edges(x, -1)
    out, last_end = [], -1
    for a, a1, u in zip(r0, r1, up):
        if a <= last_end or a == 0:
            continue
        k = np.searchsorted(f0, a1 + 1)
        while k < len(f0) and f0[k] - a1 <= MAX_GAP:
            if abs(u - dn[k]) <= MATCH * max(u, dn[k]):
                break
            k += 1
        else:
            continue                        # no matching fall
        b = f1[k] - 1                       # last minute still (partly) at the heater level
        if b - a + 1 < MIN_LEN or f1[k] >= len(x):
            continue
        lo, hi = x[a - 1], x[f1[k]]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        out.append((a, b, min((u + dn[k]) / 2, F_MAX), (lo + hi) / 2))
        last_end = f1[k]
    return out


def label_block(sl, y, valid, raw, others, h) -> str:
    v = valid[sl]
    if v.any():
        yy = y[sl][v]
        if (yy >= HEATER_W).mean() >= 0.5:
            return "WM heating"
        if (yy >= THETA).mean() >= 0.5:
            return "WM on, not heating"
    best, best_e = "Unmonitored load", 0.0
    for c in others:
        col = np.nan_to_num(raw[c].to_numpy()[sl])
        if col.max(initial=0) >= OTHER_W and col.sum() > best_e:
            best, best_e = app_type(app_name(h, c)), col.sum()
    return f"WM off: {best}"


def house_run(h: int, oof: pd.DataFrame):
    g = oof[oof.house == h]
    idx = pd.date_range(g.index.min(), g.index.max(), freq="min")
    g = g.reindex(idx)
    valid = g["y"].notna().to_numpy()
    y = g["y"].fillna(0).to_numpy()
    p = g[f"p_on_r{R}"].to_numpy()
    w = g[f"power_on_r{R}"].fillna(0).to_numpy()
    v1 = np.where(np.nan_to_num(p) > T, w, 0.0)
    x = F.load(h, columns=["x"])["x"].reindex(idx).to_numpy("float64")
    raw = load_house(h).reindex(idx)
    others = [c for c in APPLIANCE_COLS if c not in WM_CHANNELS[h]]

    preds = {pm: v1.copy() for pm in P_GRID}
    rows = []
    for a, b, f, base in pair_blocks(x):
        sl = slice(a, b + 1)
        pv = p[sl][valid[sl]]
        if pv.size == 0:
            continue
        mean_p = float(pv.mean())
        fill = np.clip(np.minimum(f, x[sl] - base), 0, None)
        fill = np.nan_to_num(fill)
        new = np.maximum(v1[sl], fill)
        vv = valid[sl]
        e_old, e_new = np.abs(y[sl] - v1[sl])[vv], np.abs(y[sl] - new)[vv]
        rows.append({"house": h, "start": idx[a], "minutes": b - a + 1, "fill_w": f, "mean_p": mean_p,
                     "label": label_block(sl, y, valid, raw, others, h),
                     "added_wh": float((new - v1[sl])[vv].sum() / 60),
                     "err_reduced_wh": float(np.clip(e_old - e_new, 0, None).sum() / 60),
                     "err_added_wh": float(np.clip(e_new - e_old, 0, None).sum() / 60)})
        for pm in P_GRID:
            if mean_p >= pm:
                preds[pm][sl] = new
    blocks = pd.DataFrame(rows)

    frame = pd.DataFrame({"house": h, "y": y, "v1": v1, **{f"fill_p{pm}": preds[pm] for pm in P_GRID}},
                         index=idx)[valid]

    hb = heater_blocks(y, valid)
    hmask = np.zeros(len(y), bool)
    for a, b in hb:
        hmask[a:b + 1] = True
    hmask &= valid
    cap = {"heating_kwh": y[hmask].sum() / 60e3, "v1_kwh": v1[hmask].sum() / 60e3,
           **{f"fill_p{pm}_kwh": preds[pm][hmask].sum() / 60e3 for pm in P_GRID}}
    return frame, blocks, cap


def benefit_harm(frame: pd.DataFrame, col: str) -> dict:
    y, old, new = frame.y.to_numpy(), frame.v1.to_numpy(), frame[col].to_numpy()
    de = np.abs(y - new) - np.abs(y - old)
    add = new - old
    on = y >= THETA
    return {"err_reduced_kwh": float(np.clip(-de, 0, None).sum() / 60e3),
            "err_added_kwh": float(np.clip(de, 0, None).sum() / 60e3),
            "added_kwh_wm_on": float(add[on].sum() / 60e3),
            "added_kwh_wm_off": float(add[~on].sum() / 60e3)}


def plot(per_house: pd.DataFrame, by_label: pd.DataFrame):
    pc = f"fill_p{P_PRIMARY}"
    ph = per_house.pivot(index="house", columns="variant", values="nde_day").sort_values("v1")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    yv = np.arange(len(ph))
    for i, (o, n) in enumerate(zip(ph["v1"], ph[pc])):
        a1.plot([o, n], [i, i], color=S.BLUE if n < o else S.MAGENTA, lw=2.2)
    a1.scatter(ph["v1"], yv, color=S.GREY, s=28, zorder=3, label="v1")
    a1.scatter(ph[pc], yv, color=S.INK, s=28, zorder=3, label=f"v1 + block fill (P_MIN {P_PRIMARY})")
    a1.set_yticks(yv, [f"House {h}" for h in ph.index])
    a1.set_xlabel("NDE_day (lower is better)")
    a1.set_title("Per development house: blue = better, magenta = worse", fontsize=10)
    a1.legend(fontsize=8, loc="lower right")

    bl = by_label.sort_values("blocks")
    yl = np.arange(len(bl))
    a2.barh(yl - 0.18, bl.err_reduced_kwh, height=0.36, color=S.BLUE, label="error removed (benefit)")
    a2.barh(yl + 0.18, bl.err_added_kwh, height=0.36, color=S.ORANGE, label="error added (harm)")
    a2.set_yticks(yl, [f"{k}  (n={int(n):,})" for k, n in zip(bl.index, bl.blocks)])
    a2.set_xlabel("kWh, summed over 16 development houses (minute level)")
    a2.set_title("Accepted blocks by what was really running", fontsize=10)
    a2.legend(fontsize=8, loc="lower right")
    fig.suptitle("EXP-03: step-paired heater-block fill on v1 leave-one-house-out predictions "
                 "(development houses only)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp03_block_fill.png", bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    oof = pd.read_parquet(OOF_DIR / "lgb_twostage_cal.parquet",
                          columns=["house", "y", f"p_on_r{R}", f"power_on_r{R}"])
    assert set(oof.house.unique()) == set(CFG["dev_houses"]), "development houses only"
    frames, blocks, caps = [], [], []
    for h in CFG["dev_houses"]:
        fr, bl, cap = house_run(h, oof)
        frames.append(fr)
        blocks.append(bl)
        caps.append({"house": h, **cap})
        print(f"house {h}: {len(bl)} paired blocks", flush=True)
    frame = pd.concat(frames)
    blocks = pd.concat(blocks, ignore_index=True)
    cols = ["v1"] + [f"fill_p{pm}" for pm in P_GRID]
    per_house = score_frame(frame, cols)
    macro = summarise(per_house)
    v1_nde = per_house[per_house.variant == "v1"].nde_day.mean()
    assert abs(v1_nde - V1_DEV_MACRO_NDE) < 0.0015, f"v1 reproduction failed: {v1_nde:.4f}"

    wide = per_house.pivot(index="house", columns="variant", values="nde_day")
    variants = {}
    for c in cols[1:]:
        bh = benefit_harm(frame, c)
        pm = float(c.split("p")[-1])
        acc = blocks[blocks.mean_p >= pm]
        variants[c] = {
            "macro_nde_day": round(float(wide[c].mean()), 4),
            "houses_better": int((wide[c] < wide["v1"]).sum()),
            "houses_worse": int((wide[c] > wide["v1"]).sum()),
            "accepted_blocks": int(len(acc)),
            "accepted_blocks_wm_heating_share": round(float((acc.label == "WM heating").mean()), 3),
            **{k: round(v, 1) for k, v in bh.items()},
        }
    prim = variants[f"fill_p{P_PRIMARY}"]
    verdict = ("helps" if prim["macro_nde_day"] < V1_DEV_MACRO_NDE and prim["houses_better"] >= 8
               else "does not help")

    acc = blocks[blocks.mean_p >= P_PRIMARY]
    g = acc.groupby("label")
    by_label = pd.DataFrame({"blocks": g.size(), "added_kwh": g.added_wh.sum() / 1000,
                             "err_reduced_kwh": g.err_reduced_wh.sum() / 1000,
                             "err_added_kwh": g.err_added_wh.sum() / 1000}).round(1)
    by_label = by_label.sort_values("blocks", ascending=False)
    cap = pd.DataFrame(caps).set_index("house")
    capture = {"heating_kwh": round(cap.heating_kwh.sum(), 1),
               **{k.replace("_kwh", "_captured_pct"): round(100 * cap[k].sum() / cap.heating_kwh.sum(), 1)
                  for k in cap.columns if k != "heating_kwh"}}
    all_paired = blocks.groupby("label").size().sort_values(ascending=False).to_dict()

    per_house.to_csv(DIAG_DIR / "exp03_per_house.csv", index=False)
    macro.to_csv(DIAG_DIR / "exp03_macro.csv")
    blocks.to_csv(DIAG_DIR / "exp03_blocks.csv", index=False)
    by_label.to_csv(DIAG_DIR / "exp03_by_label_primary.csv")
    summary = {"v1_macro_nde_day": round(float(v1_nde), 4), "primary": f"fill_p{P_PRIMARY}",
               "verdict_primary": verdict, "variants": variants, "heating_block_capture": capture,
               "paired_blocks_all_by_label": all_paired}
    (DIAG_DIR / "exp03_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot(per_house, by_label)
    return {"summary": summary, "by_label": by_label, "wide": wide, "macro": macro}


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 220)
    print(json.dumps(r["summary"], indent=1, default=str))
    print("\n== accepted blocks (primary) by label\n", r["by_label"].to_string())
    print("\n== NDE_day per house\n", r["wide"].round(3).to_string())
    m = r["macro"]
    print("\n== macro mean\n", m.xs("mean", axis=1, level=1)[["nde_day", "sae", "mae_w", "f1",
                                                            "event_precision", "event_recall"]].round(3).to_string())
