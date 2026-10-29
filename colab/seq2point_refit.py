# %% [markdown]
# # Seq2Point challenger — REFIT washing machine (run on Google Colab with a GPU)
#
# Part of the Flock Energy take-home. Model **inspired by** Seq2Point (Zhang et al. 2018,
# AAAI): a CNN maps a window of the household aggregate to the washing-machine power at the
# window's **midpoint**. Adapted here (not a reproduction): 1-minute data, a 121-minute window
# (the same context as the LightGBM features), and a second input channel marking missing minutes.
#
# **Same validation as LightGBM:** leave-one-house-out over the 16 development houses, the same
# training minutes (`train_row`), the same evaluable minutes (`evaluable`), seed 42. Test houses
# are **not** in this data. The output `seq2point.parquet` is scored on the laptop by the same
# metric code as LightGBM.
#
# **Steps:** (1) Runtime → Change runtime type → **T4 GPU** → Save. (2) Upload the 16
# `house*.parquet` files to Google Drive folder `MyDrive/refit_seq2point/`. (3) Runtime → Run all.
# (4) Download `MyDrive/refit_seq2point/oof/seq2point.parquet`.
# If Colab disconnects, just Run all again: finished folds are skipped.

# %%
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
if IN_COLAB and not os.environ.get("S2P_DATA_DIR"):
    from google.colab import drive
    drive.mount("/content/drive")

DATA_DIR = os.environ.get("S2P_DATA_DIR", "/content/drive/MyDrive/refit_seq2point")
OUT_DIR = os.path.join(DATA_DIR, "oof")
SMOKE_TEST = os.environ.get("S2P_SMOKE", "0") == "1"   # tiny run for checking the code

DEV = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 18, 19, 20, 21]   # development houses only
WIN, HALF = 121, 60          # centred 121-minute window
SEED = 42
BATCH, LR = 1024, 1e-3
MAX_EPOCHS, PATIENCE = 10, 2
N_VAL_HOUSES = 2             # early stopping on 2 of the 15 training houses of each fold
VAL_MAX = 200_000
Y_SCALE = 1000.0             # target in kW inside the network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
print("device:", DEVICE, "| torch", torch.__version__, "| smoke test:", SMOKE_TEST)
if SMOKE_TEST:
    DEV_RUN, MAX_EPOCHS, MAX_BATCHES = DEV[:1], 2, 20
else:
    DEV_RUN, MAX_BATCHES = DEV, None
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# Load all development houses into one padded series (padding = missing, so windows never
# cross from one house into the next).
frames, offsets, agg_parts, y_parts, pos = {}, {}, [], [], 0
for h in DEV:
    d = pd.read_parquet(os.path.join(DATA_DIR, f"house{h}.parquet"))
    assert int(d["house"].iloc[0]) == h
    frames[h] = d
    pad = np.full(HALF, np.nan, np.float32)
    agg_parts.append(np.concatenate([pad, d["agg"].to_numpy(np.float32), pad]))
    y_parts.append(np.concatenate([pad, d["y"].to_numpy(np.float32), pad]))
    offsets[h] = pos + HALF
    pos += len(d) + 2 * HALF
AGG = np.concatenate(agg_parts)
YG = np.concatenate(y_parts)
TRAIN_IDX = {h: offsets[h] + np.flatnonzero(frames[h]["train_row"].to_numpy()) for h in DEV}
EVAL_IDX = {h: offsets[h] + np.flatnonzero(frames[h]["evaluable"].to_numpy()) for h in DEV}
print(f"{len(DEV)} houses, {len(AGG):,} padded minutes, "
      f"{sum(len(v) for v in TRAIN_IDX.values()):,} training minutes")

AGG_T = torch.tensor(AGG, device=DEVICE)
Y_T = torch.tensor(np.nan_to_num(YG), device=DEVICE)
OFFS = torch.arange(-HALF, HALF + 1, device=DEVICE)


def make_inputs(mu: float, sd: float) -> torch.Tensor:
    """2 x N tensor: standardised aggregate (missing -> 0) and validity mask."""
    valid = ~torch.isnan(AGG_T)
    x = torch.where(valid, (AGG_T - mu) / sd, torch.zeros_like(AGG_T))
    return torch.stack([x, valid.float()])


def windows(X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Batch of centred windows: (B, 2, WIN)."""
    return X[:, idx[:, None] + OFFS].permute(1, 0, 2)


# %%
class Seq2Point(nn.Module):
    """Conv layers as in Zhang et al. (2018); 2 input channels; 121-sample window."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 30, 10, padding="same"), nn.ReLU(),
            nn.Conv1d(30, 30, 8, padding="same"), nn.ReLU(),
            nn.Conv1d(30, 40, 6, padding="same"), nn.ReLU(),
            nn.Conv1d(40, 50, 5, padding="same"), nn.ReLU(),
            nn.Conv1d(50, 50, 5, padding="same"), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(50 * WIN, 1024), nn.ReLU(), nn.Linear(1024, 1))

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


def predict(model, X, idx_np, bs=8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=AMP):
        for i in range(0, len(idx_np), bs):
            idx = torch.tensor(idx_np[i:i + bs], device=DEVICE)
            out.append(model(windows(X, idx)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([], np.float32)


# %%
log = {}
for h in DEV_RUN:
    fold_file = os.path.join(OUT_DIR, f"fold_house{h}.parquet")
    if os.path.exists(fold_file) and not SMOKE_TEST:
        print(f"house {h}: already done, skipped")
        continue
    t0 = time.time()
    torch.manual_seed(SEED + h)
    rng = np.random.default_rng(SEED + h)
    train_h = [x for x in DEV if x != h]
    val_h = sorted(int(v) for v in rng.choice(train_h, N_VAL_HOUSES, replace=False))
    fit_h = [x for x in train_h if x not in val_h]
    fit_idx = np.concatenate([TRAIN_IDX[x] for x in fit_h])
    val_idx = np.concatenate([TRAIN_IDX[x] for x in val_h])
    if len(val_idx) > VAL_MAX:
        val_idx = rng.choice(val_idx, VAL_MAX, replace=False)
    mu, sd = float(np.nanmean(AGG[fit_idx])), float(np.nanstd(AGG[fit_idx]))
    X = make_inputs(mu, sd)

    model = Seq2Point().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=AMP)
    loss_fn = nn.MSELoss()
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(fit_idx)
        nb = 0
        for i in range(0, len(perm), BATCH):
            idx = torch.tensor(perm[i:i + BATCH], device=DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                loss = loss_fn(model(windows(X, idx)).float(), Y_T[idx] / Y_SCALE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            nb += 1
            if MAX_BATCHES and nb >= MAX_BATCHES:
                break
        vp = predict(model, X, val_idx if not SMOKE_TEST else val_idx[:5000])
        vy = YG[val_idx if not SMOKE_TEST else val_idx[:5000]] / Y_SCALE
        vl = float(np.mean((vp - vy) ** 2))
        hist.append(vl)
        print(f"house {h} epoch {ep + 1}: val MSE {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    ev = EVAL_IDX[h] if not SMOKE_TEST else EVAL_IDX[h][:20000]
    pred = np.clip(predict(model, X, ev) * Y_SCALE, 0, None).astype(np.float32)
    d = frames[h]
    ev_mask = d["evaluable"].to_numpy()
    out = pd.DataFrame({"house": np.int16(h), "y": d["y"].to_numpy(np.float32)[ev_mask][:len(pred)],
                        "pred": pred}, index=d.index[ev_mask][:len(pred)])
    out.index.name = "time_utc"
    if not SMOKE_TEST:
        out.to_parquet(fold_file)
    log[h] = {"val_houses": val_h, "epochs": len(hist), "best_val_mse": best, "val_hist": hist,
              "minutes": len(out), "seconds": round(time.time() - t0)}
    print(f"house {h}: done in {time.time() - t0:.0f}s, best epoch {int(np.argmin(hist)) + 1}")
    del X
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# %%
# Combine the folds into the file to bring back to the laptop.
if not SMOKE_TEST:
    parts = [pd.read_parquet(os.path.join(OUT_DIR, f"fold_house{h}.parquet")) for h in DEV]
    oof = pd.concat(parts)
    oof.to_parquet(os.path.join(OUT_DIR, "seq2point.parquet"))
    with open(os.path.join(OUT_DIR, "seq2point_log.json"), "a") as f:
        f.write(json.dumps(log, default=str) + "\n")
    print(oof.groupby("house").size())
    print("Saved:", os.path.join(OUT_DIR, "seq2point.parquet"), "- download it and put it in cache/oof/")
else:
    print("smoke test finished:", json.dumps(log, default=str))
