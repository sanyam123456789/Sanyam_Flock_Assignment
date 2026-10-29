"""Model selection on development out-of-fold (OOF) predictions, following the procedure
pre-declared in the README. Reads cache/oof/*.parquet only (no training here).

Steps (primary metric: NDE_day, macro mean over the 16 development houses):
  1. single-stage L2 vs Tweedie x rounds {100, 200, 400, 800}
  2. zeroing threshold tau in {0, 5, 10, 20, 50} W on the step-1 winner
  3. two-stage challenger: soft gating, hard gating with P(ON) threshold {0.3..0.7};
     promoted only under the promotion rule
  4. calendar ablation of the current winner (promotion rule)
  5. Seq2Point CNN challenger (promotion rule) - if its OOF file exists
Promotion rule: the more complex candidate must have a lower macro NDE_day AND a lower
NDE_day in at least half of the development houses.
Two-stage calibration: reliability table, Brier score and expected calibration error (ECE,
10 equal-width bins). "Clearly miscalibrated" was fixed in advance as ECE > 0.05; only then
is isotonic calibration applied (fitted, for each held-out house, on the OOF predictions of
the other 15 development houses, so the held-out house is never used).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import loho
from src.config import FIG_DIR
from src.metrics import score_frame
from src.models import ROUNDS, gate

TAUS = (0, 5, 10, 20, 50)
GATE_T = (0.3, 0.4, 0.5, 0.6, 0.7)
ECE_LIMIT = 0.05
SEL_PATH = loho.MET_DIR / "selection.json"


def oof(name: str) -> pd.DataFrame:
    return pd.read_parquet(loho.OOF_DIR / f"{name}.parquet")


def score(frame: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    ph = score_frame(frame, cols)
    ph["variant"] = prefix + ":" + ph["variant"]
    return ph


def macro(ph: pd.DataFrame) -> pd.Series:
    return ph.groupby("variant")["nde_day"].mean().sort_values()


def promote(challenger: pd.DataFrame, incumbent: pd.DataFrame) -> dict:
    """Both: per-house rows of ONE variant. Returns the promotion-rule decision."""
    a = challenger.set_index("house")["nde_day"]
    b = incumbent.set_index("house")["nde_day"].reindex(a.index)
    wins = int((a < b).sum())
    return {"challenger_macro": float(a.mean()), "incumbent_macro": float(b.mean()),
            "houses_better": wins, "houses": int(len(a)),
            "promoted": bool(a.mean() < b.mean() and wins >= len(a) / 2)}


def with_variant(frame: pd.DataFrame, series: dict[str, pd.Series]) -> pd.DataFrame:
    out = frame[["house", "y"]].copy()
    for k, v in series.items():
        out[k] = v.to_numpy()
    return out


def calibration(p: pd.Series, on: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    b = np.minimum((p.to_numpy() * bins).astype(int), bins - 1)
    t = pd.DataFrame({"bin": b, "p": p.to_numpy(), "on": on.to_numpy().astype(float)}).groupby("bin").agg(
        n=("p", "size"), mean_pred=("p", "mean"), observed=("on", "mean"))
    ece = float((t.n * (t.mean_pred - t.observed).abs()).sum() / t.n.sum())
    brier = float(((p - on.astype(float)) ** 2).mean())
    return t, ece, brier


def plot_reliability(t: pd.DataFrame, ece: float, brier: float, path):
    import matplotlib.pyplot as plt
    from src import style as S
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=S.GREY, ls="--", lw=1, label="perfect calibration")
    ax.plot(t.mean_pred, t.observed, color=S.BLUE, marker="o", ms=5, lw=2, label="two-stage classifier")
    ax.set_xlabel("predicted P(ON)")
    ax.set_ylabel("observed share of ON minutes")
    ax.set_title(f"Reliability on development OOF\nECE {ece:.3f}, Brier {brier:.4f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def isotonic_loho(frame: pd.DataFrame, col: str) -> pd.Series:
    from sklearn.isotonic import IsotonicRegression
    out = pd.Series(np.nan, index=frame.index, dtype="float32")
    on = (frame["y"] >= 20).astype(float)
    for h in frame["house"].unique():
        tr = frame["house"] != h
        sub = frame[tr].iloc[::10]  # thinning for speed; same houses only
        ir = IsotonicRegression(out_of_bounds="clip").fit(sub[col], on[tr].iloc[::10])
        out[~tr] = ir.predict(frame.loc[~tr, col]).astype("float32")
    return out


def main() -> dict:
    sel = {}
    # ---------------- baselines (for the comparison table)
    base = pd.concat([score(oof("always_off"), ["pred"], "always_off"),
                      score(oof("hour_of_week"), ["pred"], "hour_of_week")])
    # ---------------- step 1: loss x rounds
    s1 = pd.concat([score(oof("lgb_l2"), [f"r{r}" for r in ROUNDS], "lgb_l2"),
                    score(oof("lgb_tweedie"), [f"r{r}" for r in ROUNDS], "lgb_tweedie")])
    m1 = macro(s1)
    w1 = m1.index[0]
    sel["step1_macro_nde"] = m1.round(4).to_dict()
    sel["step1_winner"] = w1
    cfg1, col1 = w1.split(":")
    # ---------------- step 2: tau
    f1 = oof(cfg1)
    s2 = score(with_variant(f1, {f"tau{t}": f1[col1].where(f1[col1] >= t, 0) for t in TAUS}),
               [f"tau{t}" for t in TAUS], f"{cfg1}_{col1}")
    m2 = macro(s2)
    w2 = m2.index[0]
    sel["step2_macro_nde"] = m2.round(4).to_dict()
    sel["step2_winner"] = w2
    tau = int(w2.split("tau")[-1])
    incumbent = s2[s2.variant == w2]
    winner = {"config": cfg1, "column": col1, "tau": tau, "variant": w2, "two_stage": False}
    # ---------------- step 3: two-stage challenger
    if (loho.OOF_DIR / "lgb_twostage.parquet").exists():
        f3 = oof("lgb_twostage")
        var = {}
        for r in ROUNDS:
            var[f"soft_r{r}"] = gate(f3, r, "soft")
            for t in GATE_T:
                var[f"hard_r{r}_t{t}"] = gate(f3, r, "hard", t)
        s3 = score(with_variant(f3, var), list(var), "lgb_twostage")
        m3 = macro(s3)
        w3 = m3.index[0]
        sel["step3_macro_nde_top10"] = m3.head(10).round(4).to_dict()
        sel["step3_best"] = w3
        # calibration of the classifier at the best variant's rounds
        r_best = int(w3.split("_r")[1].split("_")[0])
        t_cal, ece, brier = calibration(f3[f"p_on_r{r_best}"], f3["y"] >= 20)
        sel["step3_calibration"] = {"rounds": r_best, "ece": round(ece, 4), "brier": round(brier, 5),
                                    "clearly_miscalibrated": ece > ECE_LIMIT,
                                    "table": t_cal.round(4).reset_index().to_dict("records")}
        plot_reliability(t_cal, ece, brier, FIG_DIR / "s3_two_stage_reliability.png")
        if ece > ECE_LIMIT:  # isotonic, fitted without the held-out house
            p_iso = isotonic_loho(f3, f"p_on_r{r_best}")
            f3c = f3.copy()
            f3c[f"p_on_r{r_best}"] = p_iso
            var_c = {f"iso_soft_r{r_best}": gate(f3c, r_best, "soft")}
            for t in GATE_T:
                var_c[f"iso_hard_r{r_best}_t{t}"] = gate(f3c, r_best, "hard", t)
            s3c = score(with_variant(f3c, var_c), list(var_c), "lgb_twostage")
            s3 = pd.concat([s3, s3c])
            m3 = macro(s3)
            w3 = m3.index[0]
            sel["step3_best_after_calibration"] = w3
        decision = promote(s3[s3.variant == w3], incumbent)
        sel["step3_promotion"] = decision | {"challenger": w3, "incumbent": w2}
        if decision["promoted"]:
            incumbent = s3[s3.variant == w3]
            winner = {"config": "lgb_twostage", "variant": w3, "two_stage": True}
    else:
        s3 = pd.DataFrame()
    # ---------------- step 4: calendar ablation (needs <config>_cal OOF)
    cal_name = f"{winner['config']}_cal"
    if (loho.OOF_DIR / f"{cal_name}.parquet").exists():
        f4 = oof(cal_name)
        if winner["two_stage"]:
            v = winner["variant"].split(":")[1]
            r = int(v.split("_r")[1].split("_")[0])
            mode = "soft" if "soft" in v else "hard"
            t = float(v.split("_t")[1]) if mode == "hard" else 0.5
            if v.startswith("iso"):
                f4 = f4.copy()
                f4[f"p_on_r{r}"] = isotonic_loho(f4, f"p_on_r{r}")
            s4 = score(with_variant(f4, {v: gate(f4, r, mode, t)}), [v], cal_name)
        else:
            c, tau_ = winner["column"], winner["tau"]
            s4 = score(with_variant(f4, {f"{c}_tau{tau_}": f4[c].where(f4[c] >= tau_, 0)}),
                       [f"{c}_tau{tau_}"], cal_name)
        decision = promote(s4, incumbent)
        sel["step4_calendar"] = decision | {"challenger": s4.variant.iloc[0], "incumbent": incumbent.variant.iloc[0]}
        if decision["promoted"]:
            incumbent = s4
            winner = winner | {"calendar": True, "variant": s4.variant.iloc[0]}
    else:
        s4 = pd.DataFrame()
    # ---------------- step 5: Seq2Point (OOF produced on Colab)
    if (loho.OOF_DIR / "seq2point.parquet").exists():
        s5 = score(oof("seq2point"), ["pred"], "seq2point")
        decision = promote(s5, incumbent)
        sel["step5_seq2point"] = decision | {"incumbent": incumbent.variant.iloc[0]}
        if decision["promoted"]:
            incumbent = s5
            winner = {"config": "seq2point", "variant": "seq2point:pred"}
    else:
        s5 = pd.DataFrame()
    sel["final_choice"] = winner
    # ---------------- save tables
    allph = pd.concat([base, s1, s2, s3, s4, s5])
    allph.to_csv(loho.MET_DIR / "dev_all_variants_per_house.csv", index=False)
    loho.compare(allph).to_csv(loho.MET_DIR / "dev_all_variants_macro.csv")
    SEL_PATH.write_text(json.dumps(sel, indent=2, default=str))
    return sel


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=str))
