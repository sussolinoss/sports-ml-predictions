"""
Configurazione centrale del progetto.
Modifica qui anni, parametri ELO, split temporali.
"""

import os
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Tour: ATP (default) o WTA. Scegli con  TENNIS_TOUR=wta python ...
# ---------------------------------------------------------------------------
TOUR = os.environ.get("TENNIS_TOUR", "atp").lower()
assert TOUR in ("atp", "wta"), "TENNIS_TOUR deve essere 'atp' o 'wta'"

# ---------------------------------------------------------------------------
# Percorsi (artefatti separati per tour; CSV grezzi condivisi: nomi distinti)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # atp_matches_*.csv / wta_matches_*.csv (non collidono)
TOUR_DIR = DATA_DIR / TOUR           # processed/model/odds/output per-tour
PROCESSED_DIR = TOUR_DIR / "processed"
MODEL_PATH = TOUR_DIR / "model.json"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dati Sackmann (repo tennis_atp / tennis_wta)
# ---------------------------------------------------------------------------
SACKMANN_BASE_URL = (
    f"https://raw.githubusercontent.com/JeffSackmann/tennis_{TOUR}/master/"
    f"{TOUR}_matches_{{year}}.csv"
)

# Anni da scaricare: dal 2005 all'anno CORRENTE (cosi' il 2026/2027/... entrano
# in automatico). Il repo Sackmann su GitHub si aggiorna in stagione (lag ~1 sett.).
CURRENT_YEAR = date.today().year
YEARS = list(range(2005, CURRENT_YEAR + 1))

# Quanti degli ultimi anni ri-scaricare sempre (in-season vengono aggiornati)
REFRESH_RECENT_YEARS = 2

# Anni considerati burn-in (NON usati per training/test, solo per scaldare ELO)
BURNIN_END_YEAR = 2009

# Split temporale (chiusura inclusiva). Rolla in avanti: le ultime 2 stagioni
# restano HOLDOUT (test) per decidere se bettare; il resto allena.
TRAIN_END_YEAR = CURRENT_YEAR - 3   # es. 2026 -> train fino a 2023
VAL_END_YEAR = CURRENT_YEAR - 2     # es. 2024 (early stopping + calibrazione)
TEST_END_YEAR = CURRENT_YEAR        # es. 2025-2026 held out

# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------
ELO_INITIAL = 1500.0
ELO_K_BASE = 32.0  # K-factor base
# K decresce con il numero di match giocati: K = K_BASE / (1 + matches/SCALE)^EXP
ELO_K_DECAY_SCALE = 100.0
ELO_K_DECAY_EXP = 0.4
# K per la versione surface-specific (un po' più alto perché meno dati)
ELO_K_SURFACE = 36.0

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RECENT_FORM_WINDOW = 10  # ultimi N match per "forma"
SERVE_WINDOW = 20        # ultimi N match per le statistiche rolling di servizio/risposta
H2H_MIN_MATCHES = 1       # numero minimo per usare H2H come feature

# Seed per la randomizzazione dei lati p1/p2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
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
