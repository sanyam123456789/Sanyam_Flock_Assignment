# Washing-machine disaggregation on REFIT

Take-home assignment for Flock Energy (Data Science). A reproducible Python pipeline that
cleans raw REFIT smart-meter data, analyses washing-machine use in 19 UK homes, and estimates
washing-machine power from the whole-home signal at 1-minute resolution.

| Document | What is in it |
|---|---|
| [DATA.md](DATA.md) | Section 1 (raw House 1 inspection and cleaning) and all data-quality decisions |
| [RESULTS.md](RESULTS.md) | Section 2 (washing-machine EDA) and Section 3 (model, validation, results, failure analysis) |
| [Recommendation.md](Recommendation.md) | Section 4: one-page, utility-facing summary |

**Contents:**
[Problem](#the-problem) ·
[Dataset](#dataset) ·
[Approach](#approach) ·
[Results](#results-in-short) ·
[Setup](#setup) ·
[Running the pipeline](#running-the-pipeline) ·
[Validation](#validation-design-and-leakage-prevention) ·
[Assumptions and trade-offs](#assumptions-and-trade-offs) ·
[Limitations](#known-limitations) ·
[Research used](#research-used) ·
[AI tools](#ai-tools-used) ·
[Improvements](#possible-improvements) ·
[Repository layout](#repository-layout)

## The problem

Utilities usually only see a home's total electricity use. Knowing when large appliances run
helps with customer insight and with shifting load out of peak hours, but plug-level metering
is expensive. Non-intrusive load monitoring (NILM), also called disaggregation, estimates one
appliance's consumption from the total signal alone.

This project does that for the **washing machine**: given a home's 1-minute aggregate power,
estimate the washing machine's power for every minute, and from that its daily energy and its
wash events.

## Dataset

[REFIT Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
(University of Strathclyde, CC BY 4.0): 20 homes near Loughborough, UK, October 2013 to July
2015. Each home has an aggregate meter and 9 plug monitors, logging active power in watts about
every 8 seconds (irregular in practice).

- **Section 1** uses the raw files for House 1.
- **Sections 2 and 3** use the official cleaned files for the 19 homes that have a
  washing-machine channel (House 12 has none).

Two findings shaped everything downstream (details in DATA.md): the raw timestamps are UK local
time, and the official cleaned files are partly in UTC and partly in UK local time. We corrected
both, and our cleaned House 1 matches the official one closely (correlation 0.999).

## Approach

**Section 1: raw House 1 cleaning** (`src/raw_house1.py`). Convert timestamps to UTC, sort,
merge duplicate timestamps (mean of non-missing values), set impossible values to missing
(plug readings above 4 kW, aggregate above 20 kW), resample to 1-minute means, forward-fill gaps
of up to 5 minutes and flag longer outages without filling them.

**Section 2: washing-machine EDA** (`src/eda.py`). Resample the official cleaned data to
1 minute, apply validity rules (frozen readings, dead monitors), detect wash cycles with a simple
rule (≥ 20 W, pauses ≤ 10 min bridged, ≥ 10 min long), and summarise availability, timing and
cycle characteristics for all 19 homes.

**Section 3: disaggregation model** (`src/features.py`, `src/models.py`, `src/loho.py`,
`src/select_model.py`, `src/final_test.py`).
- *Input:* 41 features computed only from the aggregate: centred-window statistics over 5 to
  121 minutes, power changes, counts of heater-sized steps, level above baseload, and
  hour/day as sine and cosine. No appliance channel is used as input.
- *Selected model:* a two-stage LightGBM model. A classifier estimates the probability that
  the machine is ON; a regressor, trained only on ON minutes, estimates its power. Power is
  reported only when P(ON) > 0.4.
- *Candidates compared* (all decided before training, chosen on development homes only): two
  baselines (always-off, hour-of-week average), single-stage LightGBM with L2 or Tweedie loss,
  a zeroing threshold, soft and hard gating, calendar features, and a Seq2Point CNN trained on
  Google Colab.
- *Why this model:* each added piece of complexity had to beat the simpler model on average
  **and** in at least half of the 16 development homes. The two-stage model and calendar
  features passed; the CNN did not (0.802 vs 0.658).

## Results in short

The primary metric is **NDE_day**: the sum of daily washing-energy errors divided by the actual
washing energy. **Lower is better**; predicting "always off" scores exactly 1.0.

| | NDE_day |
|---|---|
| Always-off baseline | 1.000 |
| Hour-of-week baseline (test houses) | 1.052 |
| Selected model, development estimate (16 homes, leave-one-house-out) | 0.658 |
| **Selected model, test (3 unseen homes, single run)** | **0.669** (95% CI 0.64–0.70) |
| Per test home: House 2 / 15 / 16 | 0.835 / 0.495 / 0.678 |

- The model beats both baselines on daily energy in all three test homes.
- It finds about 70% of real washes, and about 70% of the washes it reports are real.
- **It under-estimates washing energy:** its totals are 39–69% of the actual totals, mainly
  because it misses part of the 2 kW heating phase. On minute-level error it is worse than
  always-off in two of the three test homes.
- The confidence interval covers day-to-day variation within these three homes, not variation
  between homes. Development results ranged from 0.36 to 1.17 across homes.

Full tables, plots and failure analysis: [RESULTS.md](RESULTS.md).

## Setup

Tested with Python 3.13.7 on Windows 11 (12-thread laptop). Houses are processed one at a time
to keep memory use down.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data** (not in the repository, about 15 GB of CSV files once extracted). Download both archives and extract them
with 7-Zip so the CSV files sit directly in these folders:

| Archive | Extract to | Files used |
|---|---|---|
| [REFIT_RAW_081116.7z](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z) | `DATA/raw/` | all `RAW_House*_Part1/2.csv` files: House 1 for Section 1, and all houses for the time-base correction of the cleaned data (DATA.md §4.2) |
| [CLEAN_REFIT_081116.7z](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z) | `DATA/clean/` | `CLEAN_House{N}.csv` for all 20 houses |

The readme files (`CLEAN_READ_ME_081116.txt`, `RAW_READ_ME_081116.txt`) are on the
[dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/).
Paths are set in `config.yaml`.

## Running the pipeline

**Everything, from the CSVs to every table and figure:**

```bash
python run_all.py
```

This runs, in order: Section 1 cleaning and figures → official data to 1-minute cache and the
time-base correction → channel checks → test-house lock check → Section 2 EDA → Section 3
leave-one-house-out training of baselines and LightGBM candidates → model selection → single
test run → test figures and failure analysis. Steps whose output already exists in `cache/` are
skipped, so an interrupted run can be restarted.

Other options:

```bash
python run_all.py --sections 1 2     # cleaning and EDA only, no model training
python run_all.py --diagnostics      # also re-run the post-hoc experiments EXP-01..03
python run_all.py --rebuild          # rebuild the 1-minute and feature caches from DATA/
```

**Running time.** LightGBM development training took about 2 h 15 min on the author's laptop
(`outputs/metrics/p4_train.log`); the data steps add to this.

**Seq2Point (optional, needs a GPU).** The CNN challenger runs on Google Colab with
development-house data only; steps are in [colab/README.md](colab/README.md). Its predictions go
into `cache/oof/seq2point.parquet`. Without that file, model selection skips the CNN step and
prints a note. The CNN was not selected, so the final model and test result do not depend on it.

**Individual steps** (each is a module that can be run on its own):

| Step | Command | Main outputs |
|---|---|---|
| Section 1 | `python -m src.raw_house1` | `outputs/raw_house1_*`, `figures/s1_*` |
| Cleaned data cache | `python -m src.clean_data` | `cache/house{N}_1min.parquet` |
| Section 2 | `python -m src.eda` | `outputs/eda_*`, `figures/s2_*` |
| Baselines (LOHO) | `python -m src.train_dev baselines` | `cache/oof/always_off.parquet`, `cache/oof/hour_of_week.parquet` |
| LightGBM candidates (LOHO) | `python -m src.train_dev` | `cache/oof/lgb_*.parquet` |
| Calendar ablation | `python -m src.train_dev calendar lgb_twostage` | `cache/oof/lgb_twostage_cal.parquet` |
| Model selection | `python -m src.select_model` | `outputs/metrics/selection.json`, `dev_all_variants_*.csv` |
| Single test run | `python -m src.final_test` | `cache/oof/TEST_final.parquet`, `outputs/metrics/test_*` |
| Test figures | `python -m src.plots_section3` and `python -m src.plots_summary` | `figures/s3_*`, `outputs/metrics/test_failure_analysis.json` |

**What a fresh clone contains.** All metric tables (`outputs/`) and figures (`figures/`) are in
the repository. The data, the 1-minute caches, the saved predictions and the trained models
(`cache/`, about 2 GB) are not, so on a fresh clone the full pipeline has to be run once.
`outputs/frozen_v1/` is a snapshot of the results at the time of the test run, with checksums
of the prediction files (`MANIFEST.json`).

**Reproducibility notes.** Seeds are fixed (42), LightGBM runs with `deterministic=True`, and
package versions are pinned. Before submission the pipeline was re-run with `python run_all.py`
(29 Oct 2026): Section 1 was rebuilt from the raw CSVs, and every table, metric and figure was
regenerated from the saved models and predictions. All 19 metric files came out byte-identical
to the frozen results in `outputs/frozen_v1/`. Retraining on different hardware or a different
thread count can change results in the last decimal.

## Validation design and leakage prevention

Written down before any model was trained:

- **Split by household.** Three test homes (2, 15, 16) were drawn at random with seed 42 from
  the 19 eligible homes and locked before any model code existed (`outputs/test_lock.json`).
  The pipeline re-runs the draw and stops if it does not match.
- **Development:** leave-one-house-out cross-validation on the other 16 homes. Every choice
  (thresholds, loss, number of boosting rounds, model type, features) was made here.
- **Test:** the selected configuration was retrained on all 16 development homes and scored
  once on the test homes. The result is reported as it came out.
- **No leakage:** no home is in both training and evaluation; features use only the aggregate;
  the cycle-rule thresholds and the feature step size come from development homes; feature
  windows never cross homes; the hour-of-week baseline is learned from training homes only.
- **Promotion rule:** a more complex model replaces a simpler one only if it lowers the
  macro-average NDE_day **and** is better in at least 8 of the 16 development homes.

A household split matches the real use (a model applied to new homes). A random split of
minutes would put nearly identical neighbouring minutes of the same wash on both sides and
overstate accuracy.

## Assumptions and trade-offs

- **Washing machine only, not washer-dryers.** Separate washer-dryer channels (Houses 1, 8, 9,
  18) stay in the aggregate as another load.
- **House 4** has two washing machines; the target is their sum. **House 13** is used only
  before its machine was replaced (25 Mar 2015).
- **Missing data is not filled** in Sections 2–3. Minutes that fail the validity rules are left
  out of training and scoring rather than guessed.
- **`Issues` minutes are kept.** They are 11–46% of the minutes when the washing machine is ON
  (unsynchronised sensors), so dropping them would remove much of what we want to study.
- **Centred windows** use up to 60 minutes of future data. Good for analysing historical data,
  but the model is not real-time.
- **Training on every 5th minute** keeps training time manageable; evaluation uses every
  minute.
- **NDE_day as the selection metric** matches daily energy, which is what a utility would use.
  It favours a cautious model, which partly explains why energy is under-estimated.
- **Calendar features were kept** because they helped on unseen development homes (11 of 16),
  even though they may encode UK routines.

## Known limitations

- Washing energy is under-estimated: the model's totals are 39–69% of the actual totals on the
  test homes. The model is better at finding washes than at measuring them.
- Results vary strongly between homes (development 0.36–1.17), and three test homes give only a
  rough estimate for new homes.
- Dishwashers, tumble dryers and unmonitored heating loads cause false washes; cold washes
  without the heater are mostly missed.
- The time-base correction of the official files is inferred from the data (three independent
  checks agree) but is not confirmed by the dataset authors.
- The model was trained on 16 UK homes from 2013–2015 and has not been tested elsewhere.

## Research used

| Paper | Idea used here |
|---|---|
| Murray, Stankovic & Stankovic (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122. [doi:10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122) | Dataset description, channel map, known data issues |
| Zhang, Zhong, Wang, Goddard & Sutton (2018). *Sequence-to-point learning with neural networks for non-intrusive load monitoring.* AAAI. [doi:10.1609/aaai.v32i1.11873](https://doi.org/10.1609/aaai.v32i1.11873) | Predicting the appliance at the **midpoint of a centred window** (our LightGBM features); SAE metric; our own Seq2Point-style CNN challenger. The paper used UK-DALE and REDD, not REFIT |
| Shin, Joo, Yim, Lee, Moon & Rhee (2019). *Subtask gated networks for non-intrusive load monitoring.* AAAI. [doi:10.1609/aaai.v33i01.33011150](https://doi.org/10.1609/aaai.v33i01.33011150) | Splitting the task into "is it on" × "how much" (our two-stage model). Ours is two separate LightGBM models, not a reproduction of their network |
| Pereira & Nunes (2018). *Performance evaluation in non-intrusive load monitoring: datasets, metrics, and tools — a review.* WIREs Data Mining and Knowledge Discovery. doi:10.1002/widm.1265 | Reporting both detection metrics and energy metrics, with explicit definitions, per house |

Murray et al. (2015), *A data management platform for personalised real-time energy feedback*
(EEDAL 2015), is also cited as the dataset readmes request.

## AI tools used

**Claude (Anthropic)** was used as a coding assistant throughout the project, in chat and as an
agent with access to the project folder: to discuss the approach and validation design, write
and debug code, run analyses, and draft and edit the documentation. **Google Colab** (GPU
runtime) was used to train the Seq2Point CNN.

The author made the project decisions (for example the House 4 and House 13 handling and the
pre-declared selection rules), ran the pipeline on his own laptop and on Colab, and checked
every output. All numbers in the documents come from the code and outputs in this repository.

## Possible improvements

In rough order of expected value:

1. **Recover heater energy.** The main error source. Post-processing does not fix it (EXP-03),
   so it needs model changes: step-aware features that describe a 2 kW block and its
   surroundings, or a sequence model trained with more weight on heating blocks. Evaluate on
   development homes first.
2. **More test homes.** Rotate the test set (nested cross-validation by household) to get a
   better estimate of performance on new homes than three homes allow.
3. **Extend the boosting-rounds grid.** 800 rounds, the top of the pre-declared grid, was best;
   larger values were not tried to avoid tuning after seeing results.
4. **Measure the effect of coarser data** by re-running the model on 15- and 30-minute
   aggregates (Recommendation.md discusses the expected effect).
5. **A causal (real-time) variant** using only past data, if live use is ever needed.
6. **Validate outside the UK** on sub-metered homes before any use in another country.

## Repository layout

```
run_all.py            end-to-end pipeline (Sections 1-3)
config.yaml           paths, channel map, test houses, seed
requirements.txt      pinned package versions
src/                  all pipeline code (one module per step, see table above)
colab/                Seq2Point CNN notebook and instructions (Google Colab, GPU)
experiments/          write-ups of post-hoc diagnostics EXP-01..03 (after the test run)
figures/              all figures used in the documents
outputs/              metric tables and JSON summaries
  metrics/            Section 3 development and test metrics
  frozen_v1/          snapshot at the time of the test run
  diagnostics/        EXP-01..03 outputs
DATA/                 REFIT CSVs (not in git)
cache/                1-minute caches, features, predictions, models (not in git)
```
