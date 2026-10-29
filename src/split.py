"""Eligibility rule and the locked, seeded random draw of test houses (Section 3).

Pre-stated (written before any model code):
  * eligible = house with a washing-machine channel, >= MIN_VALID_DAYS valid days and
    >= MIN_CYCLES cycles under a PROVISIONAL rule (>= 20 W, pauses <= 10 min bridged,
    >= 10 min long). The provisional rule is used only here; the final cycle rule (theta,
    G, L) is chosen later from development houses only.
  * test houses: N_TEST (3 if >= 10 eligible houses, else 2) drawn at random with SEED.
  * everything else: development houses (leave-one-house-out CV).
The result is written to config.yaml (`test_houses`) and the README; `check_lock()`
re-runs the draw and stops the pipeline if it does not match config.yaml.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.config import CFG, OUT_DIR, SEED, WM_CHANNELS
from src.quality import daily_validity, house_frame

MIN_VALID_DAYS = 90
MIN_CYCLES = 30


def provisional_cycles(wm: pd.Series, valid: pd.Series, th=20, gap=10, min_len=10) -> int:
    on = ((wm >= th) & valid).to_numpy()
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return 0
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    st, en = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    return int(((en - st + 1) >= min_len).sum())


def _house_row(h: int) -> dict:
    df, info = house_frame(h)
    dv = daily_validity(df)
    return info | {"valid_days": int((dv >= 0.9).sum()),
                   "provisional_cycles": provisional_cycles(df["wm"], df["valid"])}


def eligibility() -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(_house_row, sorted(WM_CHANNELS)))
    t = pd.DataFrame(rows).set_index("house")
    t["eligible"] = (t.valid_days >= MIN_VALID_DAYS) & (t.provisional_cycles >= MIN_CYCLES)
    return t


def draw(eligible: list[int]) -> list[int]:
    n = 3 if len(eligible) >= 10 else 2
    rng = np.random.default_rng(SEED)
    return sorted(int(h) for h in rng.choice(sorted(eligible), size=n, replace=False))


def check_lock(eligible: list[int]) -> None:
    locked = CFG.get("test_houses")
    if locked is None:
        raise RuntimeError("test_houses not set in config.yaml")
    if sorted(locked) != draw(eligible):
        raise RuntimeError(f"config test_houses {locked} != reproduced draw {draw(eligible)}")


if __name__ == "__main__":
    t = eligibility()
    t.to_csv(OUT_DIR / "eligibility.csv")
    elig = t.index[t.eligible].tolist()
    test = draw(elig)
    lock = {"seed": SEED, "eligible": elig, "test_houses": test,
            "dev_houses": [h for h in elig if h not in test],
            "rule": f">= {MIN_VALID_DAYS} valid days and >= {MIN_CYCLES} provisional cycles"}
    (OUT_DIR / "test_lock.json").write_text(json.dumps(lock, indent=2))
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(t.to_string())
    print(json.dumps(lock, indent=1))
