"""Load config.yaml and expose resolved paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

_p = CFG["paths"]
DATA_DIR = ROOT / _p["data_dir"]
RAW_DIR = DATA_DIR / _p["raw_subdir"]
CLEAN_DIR = DATA_DIR / _p["clean_subdir"]
CACHE_DIR = ROOT / _p["cache_dir"]
FIG_DIR = ROOT / _p["figures_dir"]
OUT_DIR = ROOT / _p["outputs_dir"]

LOCAL_TZ = CFG["local_tz"]
SEED = CFG["seed"]
HOUSES_ALL = CFG["houses_all"]
WM_CHANNELS = {int(k): list(v) for k, v in CFG["wm_channel"].items()}  # target = sum of these
WD_CHANNEL = {int(k): v for k, v in CFG["washer_dryer_channel"].items()}
PERIOD_END = {int(k): v for k, v in (CFG.get("period_end") or {}).items()}

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]

for d in (CACHE_DIR, FIG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def raw_files(house: int) -> list[Path]:
    """Raw file names have an extra underscore for houses 13-21."""
    sep = "_" if house >= 13 else ""
    return [RAW_DIR / f"RAW_House{sep}{house}_Part{p}.csv" for p in (1, 2)]


def clean_file(house: int) -> Path:
    return CLEAN_DIR / f"CLEAN_House{house}.csv"
