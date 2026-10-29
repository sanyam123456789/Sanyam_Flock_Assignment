"""Simple baselines for Section 3.

AlwaysOff     predicts 0 W at every minute. Floor for energy metrics (NDE_day = SAE = 100%,
              F1 = 0) and exposes the "MAE trap": for an appliance that is off 90-99% of the
              time, predicting nothing already gives a low minute-level MAE.
HourOfWeek    mean washing-machine power for each of the 168 hour-of-week slots (UK local
              time), learned from the TRAINING houses only (each house weighted equally) and
              applied to the held-out house. If a model cannot beat it, it is learning
              household habits, not electrical signatures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LOCAL_TZ


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    loc = index.tz_convert(LOCAL_TZ)
    return (loc.dayofweek * 24 + loc.hour).to_numpy()


class AlwaysOff:
    name = "always_off"
    features: list[str] = []

    def fit(self, train: pd.DataFrame):
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"pred": np.zeros(len(X), dtype="float32")}, index=X.index)


class HourOfWeek:
    name = "hour_of_week"
    features: list[str] = []

    def fit(self, train: pd.DataFrame):
        t = pd.DataFrame({"house": train["house"].to_numpy(), "how": hour_of_week(train.index),
                          "y": train["y"].to_numpy()})
        per_house = t.groupby(["house", "how"])["y"].mean().unstack("how")
        self.profile = per_house.reindex(columns=range(168)).mean(axis=0).fillna(0).to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"pred": self.profile[hour_of_week(X.index)].astype("float32")}, index=X.index)
