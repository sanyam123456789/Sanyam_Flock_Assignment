"""Train every pre-declared configuration with LOHO on the development houses and save the
out-of-fold predictions (cache/oof/<name>.parquet). Skips configurations already done.

    python -m src.train_dev                 # single-stage L2, Tweedie, two-stage
    python -m src.train_dev baselines       # always-off and hour-of-week baselines
    python -m src.train_dev calendar <name> # calendar ablation of a chosen config
"""
from __future__ import annotations

import sys
import time

from src import loho
from src.baselines import AlwaysOff, HourOfWeek
from src.models import SingleStage, TwoStage

CONFIGS = {
    "lgb_l2": lambda cal=False: SingleStage("l2", calendar=cal),
    "lgb_tweedie": lambda cal=False: SingleStage("tweedie", calendar=cal),
    "lgb_twostage": lambda cal=False: TwoStage(calendar=cal),
}


def train(name: str, calendar: bool = False):
    out = f"{name}_cal" if calendar else name
    if (loho.OOF_DIR / f"{out}.parquet").exists():
        print(f"{out}: already done", flush=True)
        return
    t0 = time.time()
    # events=False here: the full scoring (with event metrics) is done in src/select_model.py
    loho.run(lambda: CONFIGS[name](calendar), out, events=False)
    print(f"{out}: finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def train_baselines():
    """Both baselines through the same LOHO folds: the hour-of-week profile is learned from the
    training rows of the 15 other development houses and applied to the held-out house."""
    for cls in (AlwaysOff, HourOfWeek):
        if (loho.OOF_DIR / f"{cls.name}.parquet").exists():
            print(f"{cls.name}: already done", flush=True)
            continue
        loho.run(cls, cls.name)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "calendar":
        train(sys.argv[2], calendar=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "baselines":
        train_baselines()
    else:
        for n in CONFIGS:
            train(n)
