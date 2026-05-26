"""
Scarica e carica i match ATP dal repository pubblico di Jeff Sackmann.
https://github.com/JeffSackmann/tennis_atp

Salva i CSV grezzi in data/raw/ e produce un unico DataFrame ordinato cronologicamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Permetti l'esecuzione sia come modulo (python -m src.data_loader)
# sia come script (python src/data_loader.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, REFRESH_RECENT_YEARS, SACKMANN_BASE_URL, YEARS


# Colonne minime che ci servono dai CSV Sackmann
USEFUL_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "best_of", "round", "minutes",
    "winner_id", "winner_name", "winner_hand", "winner_ht",
    "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht",
    "loser_age", "loser_rank", "loser_rank_points",
    "score",
    # Statistiche match-by-match (servizio/risposta) per le feature rolling
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]

# Colonne statistiche da convertire a numerico
STAT_COLUMNS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]


def download_year(year: int, overwrite: bool = False) -> Path:
    """Scarica il CSV di un singolo anno se non già presente."""
    target = RAW_DIR / f"atp_matches_{year}.csv"
    if target.exists() and not overwrite:
        return target

    url = SACKMANN_BASE_URL.format(year=year)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Download fallito per anno {year} (HTTP {response.status_code}). "
            f"Controlla che l'anno esista nel repo."
        )
    target.write_bytes(response.content)
    return target


def download_all(years: list[int] | None = None,
                 refresh_recent: int = REFRESH_RECENT_YEARS) -> list[Path]:
    """Scarica tutti gli anni configurati. Gli ultimi `refresh_recent` anni vengono
    SEMPRE ri-scaricati (overwrite) perche' in stagione il repo li aggiorna."""
    years = years or YEARS
    refresh_set = set(years[-refresh_recent:]) if refresh_recent > 0 else set()
    paths = []
    for year in tqdm(years, desc="Download CSV Sackmann"):
        try:
            paths.append(download_year(year, overwrite=(year in refresh_set)))
        except RuntimeError as e:
            print(f"  ! {e}")
    return paths


def load_matches(years: list[int] | None = None) -> pd.DataFrame:
    """
    Carica tutti i CSV scaricati in un unico DataFrame ordinato per data.

    Restituisce un DataFrame con:
      - tourney_date convertito a pandas datetime
      - solo le colonne in USEFUL_COLUMNS
      - rimossi i match senza vincitore o senza data
      - ordinato per (tourney_date, match_num)
    """
    years = years or YEARS
    frames = []
    for year in years:
        path = RAW_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            print(f"  ! File mancante: {path.name}, lo salto")
            continue
        df = pd.read_csv(path, low_memory=False)
        # Tieni solo le colonne che esistono davvero (la struttura cambia un po' nel tempo)
        cols = [c for c in USEFUL_COLUMNS if c in df.columns]
        frames.append(df[cols])

    if not frames:
        raise RuntimeError("Nessun CSV trovato. Esegui prima download_all().")

    df = pd.concat(frames, ignore_index=True)

    # Parsing data
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    # Conversioni
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["best_of"] = df["best_of"].fillna(3).astype(int)
    df["surface"] = df["surface"].fillna("Unknown")

    # Numeric coercion per le feature che useremo
    for col in ["winner_ht", "winner_age", "winner_rank", "winner_rank_points",
                "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
                "minutes"] + STAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordinamento cronologico stretto: serve per evitare leakage in feature engineering
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"Scarico {len(YEARS)} anni di dati ATP...")
    download_all()
    print("\nCarico e ordino...")
    df = load_matches()
    print(f"  {len(df):,} match caricati dal {df.tourney_date.min().date()} "
          f"al {df.tourney_date.max().date()}")
    print(f"  Superfici: {df.surface.value_counts().to_dict()}")
