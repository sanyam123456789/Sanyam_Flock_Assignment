"""LightGBM models for washing-machine power (Section 3).

SingleStage  one regressor: aggregate-window features -> WM power (W); L2 or Tweedie loss.
             Predictions at several boosting-round counts come from one fit (num_iteration).
TwoStage     "hurdle" model inspired by the on/off x power decomposition of Subtask Gated
             Networks (Shin et al. 2019) - NOT a reproduction of it: two separately trained
             LightGBM models, a classifier for P(ON) (ON = power >= THETA) and a regressor for
             power trained on ON minutes only. No class re-weighting (it would distort P(ON)).
             Saved outputs: p_on and power_on per rounds count; gating is applied when scoring.
All hyperparameters except the rounds grid are fixed in advance (README).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import SEED
from src.cycles import THETA
from src.features import CALENDAR, FEATURES

ROUNDS = (100, 200, 400, 800)
BASE_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": 12, "verbosity": -1,
}


def staged_raw(booster: lgb.Booster, X: pd.DataFrame, rounds) -> dict[int, np.ndarray]:
    """Raw scores after each rounds count, accumulated tree block by tree block
    (each tree is evaluated once instead of once per rounds count)."""
    out, acc, prev = {}, None, 0
    for r in sorted(rounds):
        part = booster.predict(X, start_iteration=prev, num_iteration=r - prev, raw_score=True)
        acc = part if acc is None else acc + part
        out[r], prev = acc.copy(), r
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class SingleStage:
    def __init__(self, objective: str = "l2", calendar: bool = False, rounds=ROUNDS):
        self.objective, self.rounds = objective, rounds
        self.features = FEATURES + (CALENDAR if calendar else [])
        self.params = BASE_PARAMS | ({"objective": "tweedie", "tweedie_variance_power": 1.5}
                                     if objective == "tweedie" else {"objective": "regression"})

    def fit(self, train: pd.DataFrame):
        ds = lgb.Dataset(train[self.features], label=train["y"], free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=max(self.rounds))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = staged_raw(self.booster, X[self.features], self.rounds)
        link = np.exp if self.objective == "tweedie" else (lambda z: z)
        return pd.DataFrame({f"r{r}": np.clip(link(raw[r]), 0, None).astype("float32") for r in self.rounds},
                            index=X.index)


class TwoStage:
    def __init__(self, calendar: bool = False, rounds=ROUNDS):
        self.rounds = rounds
        self.features = FEATURES + (CALENDAR if calendar else [])

    def fit(self, train: pd.DataFrame):
        X, y = train[self.features], train["y"]
        on = (y >= THETA).astype(int)
        self.clf = lgb.train(BASE_PARAMS | {"objective": "binary"}, lgb.Dataset(X, label=on),
                             num_boost_round=max(self.rounds))
        m = on.to_numpy().astype(bool)
        self.reg = lgb.train(BASE_PARAMS | {"objective": "regression"}, lgb.Dataset(X[m], label=y[m]),
                             num_boost_round=max(self.rounds))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        rc, rr = staged_raw(self.clf, Xf, self.rounds), staged_raw(self.reg, Xf, self.rounds)
        out = {}
        for r in self.rounds:
            out[f"p_on_r{r}"] = _sigmoid(rc[r]).astype("float32")
            out[f"power_on_r{r}"] = np.clip(rr[r], 0, None).astype("float32")
        return pd.DataFrame(out, index=X.index)


def gate(oof: pd.DataFrame, r: int, mode: str, t: float = 0.5) -> pd.Series:
    """Combine two-stage outputs: soft = P(ON) x power; hard = power if P(ON) > t else 0."""
    p, w = oof[f"p_on_r{r}"], oof[f"power_on_r{r}"]
    return (p * w) if mode == "soft" else w.where(p > t, 0.0)
