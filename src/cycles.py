"""Washing-machine cycle rule (parameters chosen on DEVELOPMENT houses only).

In plain words: a cycle starts when the washing machine draws at least THETA watts,
continues through pauses of up to G minutes, and counts only if it lasts at least
L minutes. A "cycle" longer than MAX_CYCLE minutes is not a washing programme (no
domestic programme lasts 6 hours); its minutes are treated as invalid target data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THETA = 20        # W   standby sits at 0-5 W; 5-30 W is sparse; real operation >= ~50 W
G = 10            # min within-cycle pauses: most < 5 min; bridging beyond 10 min changes little
L = 10            # min 1-minute blips vs real cycles (mode 30-120 min); runs >= 10 min hold 99% of ON minutes
HEATER_W = 1500   # W   a cycle "used the heater" if its peak minute reaches this
MAX_CYCLE = 360   # min longer runs are not washing programmes


def detect_cycles(w: pd.Series, valid: pd.Series, theta=THETA, g=G, l=L) -> pd.DataFrame:
    """Cycles in a 1-minute power series. Invalid minutes count as OFF and a pause that
    contains an invalid minute is never bridged."""
    v = valid.to_numpy()
    on = (np.nan_to_num(w.to_numpy()) >= theta) & v
    idx = np.flatnonzero(on)
    cols = ["start", "end", "i0", "i1", "minutes", "energy_kwh", "peak_w", "heater"]
    if idx.size == 0:
        return pd.DataFrame(columns=cols)
    gap = np.diff(idx) - 1
    inv_cum = np.cumsum(~v)
    has_invalid = (inv_cum[idx[1:] - 1] - inv_cum[idx[:-1]]) > 0
    br = np.flatnonzero((gap > g) | has_invalid)
    st, en = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    keep = (en - st + 1) >= l
    st, en = st[keep], en[keep]
    x = np.nan_to_num(w.to_numpy())
    cs = np.r_[0, np.cumsum(x)]
    peak = np.array([x[a:b + 1].max() for a, b in zip(st, en)]) if len(st) else np.array([])
    return pd.DataFrame({
        "start": w.index[st], "end": w.index[en], "i0": st, "i1": en, "minutes": en - st + 1,
        "energy_kwh": (cs[en + 1] - cs[st]) / 60 / 1000, "peak_w": peak, "heater": peak >= HEATER_W,
    })
