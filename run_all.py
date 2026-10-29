"""Run the whole pipeline end to end, from the REFIT CSVs in DATA/ to every table and figure.

    python run_all.py                    # Sections 1, 2 and 3 (reuses whatever is already in cache/)
    python run_all.py --sections 1 2     # data cleaning and EDA only (no model training)
    python run_all.py --diagnostics      # also re-run the post-hoc experiments EXP-01..03
    python run_all.py --rebuild          # delete the data caches (1-minute files, features) first
    python run_all.py --relabel          # re-derive the time-base labels in labels/ from the raw files
    python run_all.py --smoke            # smoke test (under a minute) on a committed 4-week sample of 2 houses

Data needed: Section 1 reads the raw files (DATA/raw/); Sections 2-3 read only the official
cleaned files (DATA/clean/), because the per-day time-base labels (which compare the cleaned
files with the raw ones) are committed in labels/. Set REFIT_DATA_DIR to use data stored elsewhere.

Section 3 trains every pre-declared configuration with leave-one-house-out cross-validation
on the 16 development houses (about 2.5 hours of CPU time on a 12-thread laptop), selects the
model with the pre-declared procedure, then scores the locked test houses once. Steps whose
output already exists in cache/oof/ are skipped. The Seq2Point CNN challenger runs on Google
Colab (colab/README.md); without its predictions in cache/oof/, model selection skips that
step and prints a note (the CNN was not selected, so the final choice does not change).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

import pandas as pd

from src.config import CACHE_DIR, OUT_DIR, WORKERS


def section1_and_2(sections: set[str], relabel: bool = False) -> None:
    from src import clean_data, eda, raw_house1, split, timebase, verify_channels

    if "1" in sections:
        print("[S1] Raw House 1: inspection, cleaning, before/after figures", flush=True)
        raw_house1.main()

    if relabel:
        from concurrent.futures import ProcessPoolExecutor
        print("[S2] Re-deriving the per-day time-base labels from the raw files -> labels/", flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(timebase.daily_labels, timebase.HOUSES_ALL, [True] * len(timebase.HOUSES_ALL)))

    print("[S2] Official cleaned data -> time-base check -> 1-minute UTC cache", flush=True)
    clean_data.build_all().to_csv(OUT_DIR / "clean_file_stats.csv")
    pd.DataFrame([timebase.summary(h) for h in timebase.HOUSES_ALL]).to_csv(
        OUT_DIR / "clean_timebase_summary.csv", index=False)
    timebase.routine_shift_table().to_csv(OUT_DIR / "clean_timebase_routine_shift.csv")

    print("[S2] Washing-machine channel verification", flush=True)
    allc, mon = verify_channels.verify()
    allc.round(2).to_csv(OUT_DIR / "channel_signatures.csv", index=False)
    mon.round(2).to_csv(OUT_DIR / "wm_monthly_signatures.csv", index=False)

    if "1" in sections:
        print("[S1] Our cleaned raw House 1 vs the official cleaned House 1", flush=True)
        raw_house1.compare_with_official()

    print("[S2] Eligibility + check that the locked test houses reproduce", flush=True)
    el = split.eligibility()
    el.to_csv(OUT_DIR / "eligibility.csv")
    split.check_lock(el.index[el.eligible].tolist())

    if "2" in sections:
        print("[S2] Washing-machine EDA: tables and figures", flush=True)
        eda.main()


def section3(diagnostics: bool) -> None:
    from src import final_test, loho, plots_section3, plots_summary, select_model, train_dev

    print("[S3] Development LOHO: baselines (always-off, hour-of-week)", flush=True)
    train_dev.train_baselines()

    print("[S3] Development LOHO: LightGBM L2, Tweedie, two-stage", flush=True)
    for name in train_dev.CONFIGS:
        train_dev.train(name)

    print("[S3] Model selection, steps 1-3 (loss and rounds, zeroing threshold, two-stage)", flush=True)
    sel = select_model.main()
    winner = sel["final_choice"]["config"]

    print(f"[S3] Calendar ablation of the current winner ({winner})", flush=True)
    train_dev.train(winner, calendar=True)

    print("[S3] Model selection, all steps (calendar ablation, Seq2Point if available)", flush=True)
    sel = select_model.main()
    if not (loho.OOF_DIR / "seq2point.parquet").exists():
        print("     note: cache/oof/seq2point.parquet not found, so the Seq2Point step was skipped "
              "(see colab/README.md). The frozen record with it is outputs/frozen_v1/.", flush=True)
    print("     selected:", json.dumps(sel["final_choice"]), flush=True)

    print("[S3] Single final run on the locked test houses", flush=True)
    final_test.main()

    print("[S3] Test figures and failure analysis", flush=True)
    plots_section3.main()
    plots_summary.main()

    if diagnostics:
        from src import diag_heater, diag_lookalike, exp03_block_fill
        print("[S3] Post-hoc diagnostics EXP-01..03 (development houses only)", flush=True)
        diag_heater.main()
        diag_lookalike.main()
        exp03_block_fill.main()


def check_data(sections: set[str], relabel: bool) -> None:
    """Stop before any work if a needed CSV is missing, and say what to put where."""
    from src.config import CLEAN_DIR, DATA_DIR, HOUSES_ALL, RAW_DIR, clean_file, raw_files

    need = [clean_file(h) for h in HOUSES_ALL]                  # every section uses the cleaned files
    if relabel:
        need += [f for h in HOUSES_ALL for f in raw_files(h)]  # labels are re-derived from all raw files
    elif "1" in sections:
        need += raw_files(1)                                    # Section 1: raw House 1 only
    missing = [f for f in need if not f.exists()]
    if missing:
        print(f"Missing {len(missing)} data file(s) under {DATA_DIR}:")
        for f in missing[:10]:
            print(f"  {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print(f"Cleaned files go in {CLEAN_DIR}, raw files in {RAW_DIR}.")
        print("Run `python get_data.py` to download and extract them (README, Data),"
              " or set REFIT_DATA_DIR to where the data already is.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"],
                    help="which assignment sections to run (default: all)")
    ap.add_argument("--diagnostics", action="store_true", help="also run the post-hoc experiments EXP-01..03")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the 1-minute and feature caches first (model predictions in cache/oof are kept)")
    ap.add_argument("--relabel", action="store_true",
                    help="re-derive the committed time-base labels from the raw files (implies --rebuild)")
    ap.add_argument("--smoke", action="store_true",
                    help="only run the smoke test on the committed sample (smoke_test.py) and exit")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(subprocess.call([sys.executable, "smoke_test.py"]))
    sections = set(args.sections)
    check_data(sections, args.relabel)
    if args.rebuild or args.relabel:
        for p in list(CACHE_DIR.glob("*.parquet")) + list(CACHE_DIR.glob("*.json")) + list(CACHE_DIR.glob("*.csv")):
            p.unlink()
        shutil.rmtree(CACHE_DIR / "features", ignore_errors=True)
    t0 = time.time()
    section1_and_2(sections, args.relabel)
    if "3" in sections:
        section3(args.diagnostics)
    print(f"Done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
