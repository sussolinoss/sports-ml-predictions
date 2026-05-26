"""
Download and load ATP matches from Jeff Sackmann's public repository.
https://github.com/JeffSackmann/tennis_atp

Saves the raw CSVs in data/raw/ and produces a single chronologically ordered DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Allow running both as a module (python -m src.data_loader)
# and as a script (python src/data_loader.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, REFRESH_RECENT_YEARS, SACKMANN_BASE_URL, YEARS


# Minimum columns we need from the Sackmann CSVs
USEFUL_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "best_of", "round", "minutes",
    "winner_id", "winner_name", "winner_hand", "winner_ht",
    "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht",
    "loser_age", "loser_rank", "loser_rank_points",
    "score",
    # Match-by-match serve/return statistics for the rolling features
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]

# Statistic columns to coerce to numeric
STAT_COLUMNS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]


def download_year(year: int, overwrite: bool = False) -> Path:
    """Download the CSV for a single year if not already present."""
    target = RAW_DIR / f"atp_matches_{year}.csv"
    if target.exists() and not overwrite:
        return target

    url = SACKMANN_BASE_URL.format(year=year)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed for year {year} (HTTP {response.status_code}). "
            f"Check that the year exists in the repo."
        )
    target.write_bytes(response.content)
    return target


def download_all(years: list[int] | None = None,
                 refresh_recent: int = REFRESH_RECENT_YEARS) -> list[Path]:
    """Download all configured years. The last `refresh_recent` years are
    ALWAYS re-downloaded (overwrite) because the repo updates them in-season."""
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
    Load all downloaded CSVs into a single DataFrame ordered by date.

    Returns a DataFrame with:
      - tourney_date converted to pandas datetime
      - only the columns in USEFUL_COLUMNS
      - matches without a winner or without a date removed
      - sorted by (tourney_date, match_num)
    """
    years = years or YEARS
    frames = []
    for year in years:
        path = RAW_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            print(f"  ! File mancante: {path.name}, lo salto")
            continue
        df = pd.read_csv(path, low_memory=False)
        # Keep only the columns that actually exist (the schema shifts over time)
        cols = [c for c in USEFUL_COLUMNS if c in df.columns]
        frames.append(df[cols])

    if not frames:
        raise RuntimeError("Nessun CSV trovato. Esegui prima download_all().")

    df = pd.concat(frames, ignore_index=True)

    # Parse date
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    # Type conversions
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["best_of"] = df["best_of"].fillna(3).astype(int)
    df["surface"] = df["surface"].fillna("Unknown")

    # Numeric coercion for the features we will use
    for col in ["winner_ht", "winner_age", "winner_rank", "winner_rank_points",
                "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
                "minutes"] + STAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strict chronological ordering: required to avoid leakage in feature engineering
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
