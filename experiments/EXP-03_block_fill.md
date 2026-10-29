# EXP-03 — Step-paired heater-block fill (post-hoc, exploratory)

**Registered:** 29 Oct 2026, before running. v1 stays frozen (`v1-final`) and stays the
submitted model and the headline result (test macro NDE_day 0.669), whatever this shows.
**Test houses (2, 15, 16) are not used.** This is a post-processing step on saved predictions;
nothing is retrained.

## Why
EXP-01: v1 captures only 44% of the energy in true heating blocks (25% lost because the gate
is closed, 31% because the regressor predicts too few watts). In 84% of heating blocks the
~2 kW heater step is visible in the aggregate. EXP-02: look-alike heating loads (dishwasher,
tumble dryer, kettle, ...) get much lower P(ON) than real washing-machine heating blocks
(block mean 0.16 vs 0.46, AUC 0.83), but are 3.3× more numerous.

**Idea:** find heater-sized ON/OFF step pairs in the aggregate; where v1 already says "washing
machine likely" around that block, fill the block with the step size. **Benefit:** recovers
heater energy. **Risk:** adds false energy when the pair belongs to another appliance.

## Data (development houses only)
`cache/oof/lgb_twostage_cal.parquet` (leave-one-house-out `y`, `p_on_r800`, `power_on_r800`
of the 16 development houses) and the aggregate `x` from `cache/features/`. Appliance channels
are used **only to label** filled blocks afterwards (never to decide a fill).

## Method (all parameters fixed here, from existing development-house facts)
1. **Edges:** consecutive minutes where the aggregate changes by more than 200 W in the same
   direction form one edge (at most 3 minutes; 1-minute averages split a step over two
   minutes). Edge size = total change. A heater-sized edge has |size| ≥ 1000 W (existing
   `STEP_W`).
2. **Pairing:** each rising edge is paired with the first later falling edge that starts within
   40 minutes (development heating blocks: 99th percentile 39 min) and whose size matches
   within 30% of the larger of the two. Pairs do not overlap. A block runs from the first
   rising minute to the minute before the fall ends, at least 3 minutes, and must have valid
   aggregate data at both ends.
3. **Fill value per minute:** `min(F, x(t) − base)`, clipped at 0, where F = mean of rise and
   drop sizes, capped at 3000 W, and base = mean of the aggregate just before the rise and
   just after the fall. The fill can never exceed the aggregate.
4. **Accept** a block if its mean P(ON) (evaluable minutes) ≥ **P_MIN = 0.3** (primary; between
   the WM-heating mean 0.46 and the look-alike mean 0.16 from EXP-02).
5. **New prediction** in accepted blocks = max(v1 prediction, fill); unchanged elsewhere.

**Sensitivity (reported in full, not used to pick a winner):** P_MIN ∈ {0.2, 0.3, 0.4, 0.5}.
Any P_MIN other than 0.3 would be chosen in-sample on the same 16 houses and is labelled so.

## Outputs
- Per house and macro: NDE_day, SAE, MAE, F1 for v1 vs block fill (same metric code).
- **Benefit vs harm:** minute-level absolute-error reduction (kWh) vs increase (kWh); added
  energy split into WM-ON minutes and WM-OFF minutes (false energy).
- Accepted blocks labelled as: WM heating (actual WM ≥ 1500 W in ≥ half the minutes),
  WM on but not heating, or WM off → by another monitored channel ≥ 1000 W (dishwasher,
  tumble dryer, kettle, ...) or "unmonitored".
- Share of EXP-01 heating-block energy captured before / after.

## Decision rule (written before running)
The primary variant (P_MIN = 0.3) **helps** only if it lowers the macro development NDE_day
below v1 (0.658) **and** lowers it in at least 8 of 16 development houses (the same shape as
the project's promotion rule). Otherwise it **does not help**.
- Helps → described in the Recommendation as a candidate next step that still needs checking
  on houses it has never seen. It is **not** applied to the test houses here and does not
  change any reported test number.
- Does not help → reported as tried and not useful; v1 unchanged.

## Results (appended after running; numbers in `outputs/diagnostics/exp03_*`, figure `figures/exp03_block_fill.png`)
**Verdict (primary, P_MIN = 0.3): does not help.** Macro development NDE_day **0.881 vs 0.658**
for v1 (v1 reproduced exactly); better in only **3 of 16** houses (4, 8, 17), worse in 13.
- **Benefit:** share of true heating-block energy captured rises from 44.4% to 59.6%; error
  removed 272 kWh (242 kWh of it in real WM heating blocks).
- **Harm:** error added 1,150 kWh, about 4× the benefit. Only 27% of the 10,070 accepted blocks
  are real WM heating. Biggest harm sources: other loads switching **during a wash**
  ("WM on, not heating": 2,689 blocks, +330 kWh error) — the gate's P(ON) is high for the whole
  cycle, so it cannot tell whose step it is; dishwasher (+276 kWh), unmonitored loads (+200),
  tumble dryer (+163), kettle (+63).
- **Sensitivity (in-sample, not used to choose):** P_MIN 0.2 / 0.4 / 0.5 → macro NDE_day
  1.108 / 0.772 / 0.716. A stricter gate removes harm faster than benefit but never beats v1.
- Across all houses, heater-sized step pairs from other loads outnumber WM heating pairs about
  16 to 1 (61,000+ vs 4,040), so step pairing plus v1's P(ON) is not specific enough.

**Reading:** the missing heater energy cannot be recovered by post-processing v1. A fix would
need a model that sees the step *and* the washing-machine pattern around it, at the minute level
(e.g. step-aware features or a sequence model trained on heater blocks). v1 unchanged.
