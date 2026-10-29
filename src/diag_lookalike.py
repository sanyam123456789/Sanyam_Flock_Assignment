"""EXP-02: look-alike loads and regressor "mixing" check (post-hoc, diagnostic only;
see experiments/EXP-02_lookalike_check.md). Development houses only; nothing retrained.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import style as S
from src.appliances import name as app_name
from src.clean_data import load_house
from src.config import APPLIANCE_COLS, CFG, FIG_DIR, WM_CHANNELS
from src.cycles import HEATER_W, THETA
from src.diag_heater import DIAG_DIR, R, T, heater_blocks
from src.loho import OOF_DIR

TYPES = ["Dishwasher", "Tumble dryer", "Washer dryer", "Kettle", "Electric heater", "Toaster", "Microwave",
         "Dehumidifier"]
BANDS = [20, 100, 300, 800, HEATER_W, 10_000]


def app_type(n: str) -> str:
    for t in TYPES:
        if n.lower().startswith(t.lower()):
            return t
    return "Other"


def block_stats(kind, h, a, b, y, p, w, pred, label):
    sl = slice(a, b + 1)
    return {"kind": kind, "type": label, "house": h, "minutes": b - a + 1,
            "mean_p": float(p[sl].mean()), "max_p": float(p[sl].max()),
            "open_share": float((p[sl] > T).mean()), "mean_w": float(w[sl].mean()),
            "pred_wh": float(pred[sl].sum() / 60), "act_wm_wh": float(y[sl].sum() / 60)}


def main() -> dict:
    oof = pd.read_parquet(OOF_DIR / "lgb_twostage_cal.parquet",
                          columns=["house", "y", f"p_on_r{R}", f"power_on_r{R}"])
    assert set(oof.house.unique()) == set(CFG["dev_houses"])
    rows, mins_wm, mins_la, bands = [], [], [], []
    for h in CFG["dev_houses"]:
        g = oof[oof.house == h]
        idx = pd.date_range(g.index.min(), g.index.max(), freq="min")
        g = g.reindex(idx)
        valid = g["y"].notna().to_numpy()
        y = g["y"].fillna(0).to_numpy()
        p = g[f"p_on_r{R}"].fillna(0).to_numpy()
        w = g[f"power_on_r{R}"].fillna(0).to_numpy()
        pred = np.where(p > T, w, 0.0)
        for a, b in heater_blocks(y, valid):
            rows.append(block_stats("wm", h, a, b, y, p, w, pred, "WM heating"))
            mins_wm.append(p[a:b + 1])
        raw = load_house(h).reindex(idx)
        wm_off = valid & (y < THETA)
        for c in [c for c in APPLIANCE_COLS if c not in WM_CHANNELS[h]]:
            v = np.nan_to_num(raw[c].to_numpy())
            label = app_type(app_name(h, c))
            for a, b in heater_blocks(v, wm_off):
                rows.append(block_stats("lookalike", h, a, b, y, p, w, pred, label))
                mins_la.append(p[a:b + 1])
        on = valid & (y >= THETA)
        band = pd.cut(y[on], BANDS, right=False)
        bands.append(pd.DataFrame({"band": band, "y": y[on], "w": w[on], "pred": pred[on], "house": h}))
    blocks = pd.DataFrame(rows)
    blocks.to_csv(DIAG_DIR / "exp02_blocks.csv", index=False)

    # 1. per type
    g = blocks.groupby("type")
    by_type = pd.DataFrame({
        "blocks": g.size(), "houses": g.house.nunique(), "median_minutes": g.minutes.median(),
        "mean_p": g.mean_p.mean().round(3), "gate_open_share": g.open_share.mean().round(3),
        "mean_regressor_w": g.mean_w.mean().round(0),
        "predicted_kwh": (g.pred_wh.sum() / 1000).round(1),
    }).sort_values("blocks", ascending=False)
    by_type.to_csv(DIAG_DIR / "exp02_by_type.csv")

    # 2. separability (ROC AUC: WM heating = positive, look-alike = negative)
    pm_wm, pm_la = np.concatenate(mins_wm), np.concatenate(mins_la)
    auc = {"minute_p": roc_auc_score(np.r_[np.ones(len(pm_wm)), np.zeros(len(pm_la))], np.r_[pm_wm, pm_la])}
    for sub, d in (("all_blocks", blocks), ("blocks_ge6min", blocks[blocks.minutes >= 6])):
        lab = (d.kind == "wm").astype(int)
        auc[f"{sub}_mean_p"] = roc_auc_score(lab, d.mean_p)
        auc[f"{sub}_max_p"] = roc_auc_score(lab, d.max_p)
    for t in ("Dishwasher", "Tumble dryer", "Kettle"):
        d = blocks[(blocks.kind == "wm") | (blocks.type == t)]
        if (d.type == t).sum() >= 20:
            auc[f"vs_{t.lower().replace(' ', '_')}_block_mean_p"] = roc_auc_score((d.kind == "wm").astype(int), d.mean_p)
    auc = {k: round(float(v), 3) for k, v in auc.items()}

    # 3. regressor bias by actual-power band (ON minutes)
    bd = pd.concat(bands)
    gb = bd.groupby("band", observed=True)
    reg = pd.DataFrame({"minutes": gb.size(), "mean_actual_w": gb.y.mean().round(0),
                        "mean_regressor_w": gb.w.mean().round(0), "mean_final_pred_w": gb.pred.mean().round(0)})
    reg["regressor_bias_pct"] = (100 * (reg.mean_regressor_w / reg.mean_actual_w - 1)).round(1)
    reg.index = reg.index.astype(str)
    reg.to_csv(DIAG_DIR / "exp02_regressor_by_band.csv")
    all_on = {"mean_actual_w": round(float(bd.y.mean()), 1), "mean_regressor_w": round(float(bd.w.mean()), 1)}
    # ADDED AFTER REGISTRATION (diagnostic only): grouping by ACTUAL power always shows shrinkage,
    # even for a calibrated model. The calibration check groups by PREDICTED power instead:
    # among ON minutes, is the mean actual power equal to the mean regressor output?
    pb = pd.cut(bd.w, [0, 100, 300, 600, 1000, 1500, 2000, 10_000], right=False)
    gc = bd.groupby(pb, observed=True)
    calib = pd.DataFrame({"minutes": gc.size(), "mean_regressor_w": gc.w.mean().round(0),
                          "mean_actual_w": gc.y.mean().round(0)})
    calib["actual_over_predicted"] = (calib.mean_actual_w / calib.mean_regressor_w).round(2)
    calib.index = calib.index.astype(str)
    calib.to_csv(DIAG_DIR / "exp02_regressor_calibration_ADDED.csv")

    # false energy from look-alike blocks vs real energy captured in WM heating blocks
    wm_b, la_b = blocks[blocks.kind == "wm"], blocks[blocks.kind == "lookalike"]
    summary = {"auc": auc, "regressor_all_on_minutes": all_on,
               "wm_heating": {"blocks": len(wm_b), "mean_p": round(float(wm_b.mean_p.mean()), 3),
                              "gate_open_share": round(float(wm_b.open_share.mean()), 3),
                              "captured_kwh": round(float(wm_b.pred_wh.sum() / 1000), 1),
                              "actual_kwh": round(float(wm_b.act_wm_wh.sum() / 1000), 1)},
               "lookalike": {"blocks": len(la_b), "mean_p": round(float(la_b.mean_p.mean()), 3),
                             "gate_open_share": round(float(la_b.open_share.mean()), 3),
                             "false_kwh": round(float(la_b.pred_wh.sum() / 1000), 1)}}
    (DIAG_DIR / "exp02_summary.json").write_text(json.dumps(summary, indent=2))

    # figures
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    order = ["WM heating"] + [t for t in by_type.index if t != "WM heating" and by_type.loc[t, "blocks"] >= 20]
    data = [blocks.loc[blocks.type == t, "mean_p"].to_numpy() for t in order]
    bp = a1.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops={"color": S.INK, "lw": 1.5})
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(S.BLUE if t == "WM heating" else S.ORANGE)
        patch.set_alpha(0.55)
    a1.set_yticks(range(1, len(order) + 1), [f"{t} (n={int(by_type.loc[t, 'blocks'])})" for t in order])
    a1.axvline(T, color=S.GREY, ls="--", lw=1)
    a1.set_xlabel("block-average P(ON) from the v1 classifier")
    a1.set_title("Real WM heating blocks vs other appliances' heating blocks (WM off)")
    x = np.arange(len(reg))
    a2.bar(x - 0.2, reg.mean_actual_w, 0.4, color=S.INK, label="actual")
    a2.bar(x + 0.2, reg.mean_regressor_w, 0.4, color=S.BLUE, label="regressor output")
    a2.set_xticks(x, [b.replace(", ", "–").strip("[)") + " W" for b in reg.index], rotation=20)
    a2.set_ylabel("mean power (W)")
    a2.set_title("Regressor vs actual, by actual WM power (ON minutes)")
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_lookalike.png", bbox_inches="tight")
    plt.close(fig)
    return {"summary": summary, "by_type": by_type, "regressor": reg, "calibration_added": calib}


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 220)
    print(json.dumps(r["summary"], indent=1))
    print(r["by_type"].to_string())
    print(r["regressor"].to_string())
    print("\nADDED AFTER REGISTRATION - calibration by predicted power (ON minutes):")
    print(r["calibration_added"].to_string())
