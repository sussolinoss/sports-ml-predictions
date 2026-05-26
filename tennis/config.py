"""
Central project configuration.
Edit years, ELO parameters, and temporal splits here.
"""

import os
from datetime import date
from pathlib import Path

# Tour: ATP (default) or WTA. Select with  TENNIS_TOUR=wta python ...
TOUR = os.environ.get("TENNIS_TOUR", "atp").lower()
assert TOUR in ("atp", "wta"), "TENNIS_TOUR must be 'atp' or 'wta'"

# Paths: per-tour artifacts; raw CSVs are shared but use distinct names.
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # atp_matches_*.csv / wta_matches_*.csv (no collision)
TOUR_DIR = DATA_DIR / TOUR           # per-tour processed/model/odds/output
PROCESSED_DIR = TOUR_DIR / "processed"
MODEL_PATH = TOUR_DIR / "model.json"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Sackmann data (tennis_atp / tennis_wta repos)
SACKMANN_BASE_URL = (
    f"https://raw.githubusercontent.com/JeffSackmann/tennis_{TOUR}/master/"
    f"{TOUR}_matches_{{year}}.csv"
)

# Years to download: from 2005 to the CURRENT year (so 2026/2027/... are picked up
# automatically). The Sackmann GitHub repo updates in-season (lag ~1 week).
CURRENT_YEAR = date.today().year
YEARS = list(range(2005, CURRENT_YEAR + 1))

# How many of the most recent years to always re-download (updated in-season)
REFRESH_RECENT_YEARS = 2

# Years treated as burn-in (NOT used for training/test, only to warm up ELO)
BURNIN_END_YEAR = 2009

# Temporal split (inclusive end). Rolls forward: the last 2 seasons stay as
# HOLDOUT (test) for the betting decision; the rest is used for training.
TRAIN_END_YEAR = CURRENT_YEAR - 3   # e.g. 2026 -> train up to 2023
VAL_END_YEAR = CURRENT_YEAR - 2     # e.g. 2024 (early stopping + calibration)
TEST_END_YEAR = CURRENT_YEAR        # e.g. 2025-2026 held out

ELO_INITIAL = 1500.0
ELO_K_BASE = 32.0  # base K-factor
# K decays with matches played: K = K_BASE / (1 + matches/SCALE)^EXP
ELO_K_DECAY_SCALE = 100.0
ELO_K_DECAY_EXP = 0.4
# K for the surface-specific version (slightly higher because less data)
ELO_K_SURFACE = 36.0

RECENT_FORM_WINDOW = 10  # last N matches for "form"
SERVE_WINDOW = 20        # last N matches for rolling serve/return statistics
H2H_MIN_MATCHES = 1       # minimum count to use H2H as a feature

# Seed for the p1/p2 side randomization
RANDOM_SEED = 42

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda": 1.5,
    "reg_alpha": 0.5,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
}

EARLY_STOPPING_ROUNDS = 80
