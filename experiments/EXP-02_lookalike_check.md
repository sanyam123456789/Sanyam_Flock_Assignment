# EXP-02 — Look-alike loads and regressor "mixing" check (post-hoc, diagnostic only)

**Registered:** 28 Oct 2026, before running. Follows EXP-01 (heater energy lost: 25% gating,
31% regressor). v1 stays frozen (`v1-final`); no model, threshold or prediction is changed.

## Questions
1. **Look-alikes:** when another monitored heating load runs (dishwasher, tumble dryer,
   kettle, heaters, washer-dryer, ...) and the washing machine is OFF, what P(ON) and regressor
   power does v1 give? If they look like real washing-machine heating blocks, a gate-continuity
   fix would add false washes about as fast as it recovers real energy.
2. **Block-level separability:** can block-level summaries of P(ON) (mean, max) separate true
   washing-machine heating blocks from look-alike blocks better than single minutes? (Measured
   as ROC AUC.) High separability → continuity/block-level decisions are promising; low → only
   new features would help.
3. **Regressor mixing:** EXP-01 found the regressor ~22% too low in heating minutes. If it
   "averages" heater and motor minutes, it should be too HIGH on motor minutes (20–1500 W).

## Data (development houses only)
Same as EXP-01: `cache/oof/lgb_twostage_cal.parquet` (`y`, `p_on_r800`, `power_on_r800`) for the
16 development houses; appliance channels from the 1-minute cleaned cache, used only to label
blocks (never as model input). Test houses are not used.

## Definitions
- **WM heating block:** as EXP-01 (y ≥ 1500 W, single-minute dips bridged, ≥ 3 min).
- **Look-alike block:** a run of ≥ 3 evaluable minutes where one other monitored channel is
  ≥ 1500 W and the washing machine is OFF (y < 20 W) throughout. Labelled by appliance name.
  Unmonitored loads cannot be labelled and are not included.
- Gate threshold 0.4 and rounds 800, as in v1.

## Outputs and reading
- Per appliance type: blocks, mean P(ON), share of minutes with gate open, false energy
  predicted (kWh) — compared with WM heating blocks.
- ROC AUC (WM heating vs look-alike) for minute-level P(ON), block mean P(ON), block max P(ON),
  overall and for blocks ≥ 6 min.
- Regressor bias (mean predicted − mean actual) by actual-power band among ON minutes.
No decision rule: diagnostic only. Any fix will be registered separately (EXP-03).

## Results (appended after running; numbers in `outputs/diagnostics/exp02_*`)
- Look-alike blocks (18,121, WM off) get much lower P(ON) than real WM heating blocks
  (mean 0.16 vs 0.46; gate open 13% vs 55% of minutes). Separability (ROC AUC): minute 0.81,
  block mean 0.83 (≥ 6 min: 0.83); vs kettle 0.89, dishwasher 0.82, tumble dryer 0.74.
  Monitored look-alikes already receive 459 kWh of false energy (dishwasher 287, tumble dryer
  120) vs 997 kWh captured in real heating blocks; they are 3.3× more numerous than WM blocks.
- Regressor by ACTUAL band: +241% (20–100 W), +52% (100–300 W), −18%, −29%, −31% (≥ 1500 W);
  overall on ON minutes −4% (439 vs 459 W).
- **Added after registration:** calibration by PREDICTED band (ON minutes): actual/predicted
  = 0.95–1.12 between 100 and 2000 W (1.95 below 100 W, 0.89 above 2000 W). The regressor is
  roughly calibrated; its low heater predictions reflect uncertainty, not a fixable bias.
