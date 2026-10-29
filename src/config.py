"""Load config.yaml and expose resolved paths."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

_p = CFG["paths"]


def _path(env: str, default: str) -> Path:
    """config.yaml path, unless overridden by an environment variable (used by the smoke
    test, or to point at data stored elsewhere, e.g. REFIT_DATA_DIR=/data/refit)."""
    return Path(os.environ[env]).resolve() if os.environ.get(env) else ROOT / default


DATA_DIR = _path("REFIT_DATA_DIR", _p["data_dir"])
RAW_DIR = DATA_DIR / _p["raw_subdir"]
CLEAN_DIR = DATA_DIR / _p["clean_subdir"]
CACHE_DIR = _path("REFIT_CACHE_DIR", _p["cache_dir"])
LABEL_DIR = _path("REFIT_LABEL_DIR", _p["labels_dir"])
FIG_DIR = _path("REFIT_FIG_DIR", _p["figures_dir"])
OUT_DIR = _path("REFIT_OUT_DIR", _p["outputs_dir"])

LOCAL_TZ = CFG["local_tz"]
SEED = CFG["seed"]
HOUSES_ALL = CFG["houses_all"]
WM_CHANNELS = {int(k): list(v) for k, v in CFG["wm_channel"].items()}  # target = sum of these
WD_CHANNEL = {int(k): v for k, v in CFG["washer_dryer_channel"].items()}
PERIOD_END = {int(k): v for k, v in (CFG.get("period_end") or {}).items()}

# Houses processed in parallel by the data steps. Each worker holds one full cleaned CSV in
# memory; 4 workers ran out of memory on a 32 GB laptop, so the default is 2. Set
# REFIT_WORKERS=1 if a run still fails with a MemoryError. Results do not depend on it.
WORKERS = int(os.environ.get("REFIT_WORKERS", 2))

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]

for d in (CACHE_DIR, LABEL_DIR, FIG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def raw_files(house: int) -> list[Path]:
    """Raw file names have an extra underscore for houses 13-21."""
    sep = "_" if house >= 13 else ""
    return [RAW_DIR / f"RAW_House{sep}{house}_Part{p}.csv" for p in (1, 2)]


def clean_file(house: int) -> Path:
    return CLEAN_DIR / f"CLEAN_House{house}.csv"
