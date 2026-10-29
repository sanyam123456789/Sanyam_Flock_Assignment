# EXP-01 — Heater-block diagnostic (post-hoc, diagnostic only)

**Registered:** 28 Oct 2026, before running. **Status of v1:** frozen (`outputs/frozen_v1/`,
git tag `v1-final`, commit `a70dc3c`). This experiment changes **no** model, threshold or
prediction; it only explains the main failure found in the v1 test analysis.

## Question
On the final test houses the selected model found ~91% of heater washes but captured only
22–51% of their energy. Where is the energy lost inside true heating blocks?
- **A. Gating:** P(ON) ≤ 0.4, so the gate sets the prediction to 0 although the regressor
  predicted power.
- **B. Regressor:** the gate is open but the predicted watts are too low.
- **C. Both.**
No cause is assumed in advance.

## Data (development houses only)
- `cache/oof/lgb_twostage_cal.parquet` — leave-one-house-out predictions of the selected
  configuration on the 16 development houses: `y` (actual WM watts), `p_on_r800` (classifier
  P(WM ≥ 20 W)), `power_on_r800` (regressor watts, trained on ON minutes, clipped at 0).
  Selected prediction = `power_on_r800` if `p_on_r800 > 0.4`, else 0.
- `cache/features/house{h}.parquet` (aggregate `x`) and the 1-minute cleaned cache (other
  appliance channels) — used **only to describe** blocks, never as model input.
- Test houses (2, 15, 16) and `TEST_final.parquet` are **not** used.

## Method
- **True heating block** (from the actual WM reading only): run of minutes with `y ≥ 1500 W`
  (existing `HEATER_W`), bridging single-minute dips, at least 3 minutes long, all minutes evaluable.
- **Exact loss split** for every block minute: `y − pred = (y − w) + w·[p ≤ 0.4]`
  → regressor error + gating loss (they sum to the total loss).
- Outputs: per-house and overall split; block-aligned average profiles of y, w, p and prediction
  (from block start and towards block end); histogram of p in heater minutes; w vs y when the
  gate is open; splits by block length, by another monitored load ≥ 1 kW running at the same
  time, and by whether the ~2 kW step is visible in the aggregate.

## How results will be read (no decision rule — diagnostic only)
- Gating loss clearly larger → continuity/hysteresis or block-fill post-processing is the
  candidate fix (to be registered as EXP-02).
- Regressor shortfall clearly larger → regressor-side fixes (power calibration, heater-aware
  regressor or step features) are the candidates.
- Losses concentrated where other loads overlap → a confuser problem; fixes will help less.
Any fix will be registered separately and evaluated on the 16 development folds first. The
v1 test result (macro NDE_day 0.669) stays the headline whatever follows.
