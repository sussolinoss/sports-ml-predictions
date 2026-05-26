"""
Bookmaker odds loader from tennis-data.co.uk (free, CSV/xlsx).

SOLE USE: ROI evaluation in the backtest. Odds do NOT enter the model as
features (the model stays "pure" on ELO/form/rank/...). Here we only download
historical closing odds to simulate betting against the market.

tennis-data.co.uk format (ATP):
  - one .zip archive per year: http://www.tennis-data.co.uk/{year}/{year}.zip
  - inside, an .xlsx with columns: Date, Winner, Loser, WRank, LRank,
    B365W/B365L (Bet365), PSW/PSL (Pinnacle), MaxW/MaxL, AvgW/AvgL, ...
  - names are in the format "Surname I." (e.g. "Federer R.")

Exposes:
  download_odds_years(years)         -> download/cache the yearly archives
  load_odds(start, end) -> DataFrame [date, winner_key, loser_key, odd_winner, odd_loser, source]
  sackmann_name_to_key(full_name)    -> "surname i." to join with Sackmann names
  build_pair_index(odds_df)          -> dict frozenset({key_a,key_b}) -> list of odds rows
"""

from __future__ import annotations

import io
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TOUR, TOUR_DIR

RAW_ODDS_DIR = TOUR_DIR / "raw_odds"
RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)

# tennis-data.co.uk: ATP in /{year}/, WTA in /{year}w/
_TD_SUF = "w" if TOUR == "wta" else ""
TD_ZIP_URL = "http://www.tennis-data.co.uk/{year}" + _TD_SUF + "/{year}.zip"
TD_XLSX_URL = "http://www.tennis-data.co.uk/{year}" + _TD_SUF + "/{year}.xlsx"

# Odds column preference (from most reliable to fallback)
ODDS_PAIRS = [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL"), ("MaxW", "MaxL")]


def _xlsx_path(year: int) -> Path:
    return RAW_ODDS_DIR / f"{year}.xlsx"


def _is_valid_xlsx(path: Path) -> bool:
    """xlsx is a zip archive: it must start with the 'PK' signature."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def download_odds_year(year: int, overwrite: bool = False) -> Path | None:
    """Download one year's odds archive (zip or direct xlsx). Cached on disk."""
    target = _xlsx_path(year)
    if target.exists() and not overwrite:
        if _is_valid_xlsx(target):
            return target
        target.unlink()  # corrupt cache: re-download

    # Attempt 1: zip
    try:
        r = requests.get(TD_ZIP_URL.format(year=year), timeout=30)
        if r.status_code == 200 and r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                xlsx_name = next((n for n in z.namelist() if n.lower().endswith(".xlsx")), None)
                if xlsx_name:
                    target.write_bytes(z.read(xlsx_name))
                    return target
    except (requests.RequestException, zipfile.BadZipFile) as e:
        print(f"  ! zip {year} fallito ({e}), provo xlsx diretto")

    # Attempt 2: direct xlsx
    try:
        r = requests.get(TD_XLSX_URL.format(year=year), timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            target.write_bytes(r.content)
            return target
    except requests.RequestException as e:
        print(f"  ! xlsx {year} fallito ({e})")

    print(f"  ! quote anno {year} non disponibili")
    return None


def download_odds_years(years: list[int]) -> list[Path]:
    paths = []
    for y in years:
        p = download_odds_year(y)
        if p:
            paths.append(p)
    return paths


def _strip_accents(s: str) -> str:
    """'Médvédev' -> 'medvedev' (fold accents for matching)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def sackmann_name_to_key(full_name: str) -> str:
    """
    'Roger Federer'   -> 'federer r'
    'Alex De Minaur'  -> 'de minaur a'   (surname = everything except the first name)
    Initial = first letter of the first token; surname = the remaining tokens.
    """
    parts = _strip_accents(str(full_name)).split()
    if len(parts) < 2:
        return _strip_accents(str(full_name)).strip().lower()
    initial = parts[0][0]
    surname = " ".join(parts[1:])
    return f"{surname} {initial}".lower()


def _td_name_to_key(td_name: str) -> str:
    """
    'Federer R.'    -> 'federer r'
    'De Minaur A.'  -> 'de minaur a'   (last token = initial, rest = surname)
    """
    tokens = _strip_accents(str(td_name)).replace(".", " ").split()
    if len(tokens) < 2:
        return " ".join(tokens).lower()
    initial = tokens[-1][0]
    surname = " ".join(tokens[:-1])
    return f"{surname} {initial}".lower()


def implied_probs(odd_winner: float, odd_loser: float) -> tuple[float, float]:
    """From decimal odds to probabilities (overround removed). Returns (p_winner, p_loser)."""
    if odd_winner <= 1.0 or odd_loser <= 1.0:
        return 0.5, 0.5
    iw, il = 1.0 / odd_winner, 1.0 / odd_loser
    s = iw + il
    return iw / s, il / s


def _parse_one(path: Path, book: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if "Winner" not in df.columns or "Loser" not in df.columns:
        return pd.DataFrame()

    # If a book is specified (e.g. 'PS'), use only that pair; otherwise fall back in order
    pairs = [(f"{book}W", f"{book}L")] if book else ODDS_PAIRS

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    out["winner_key"] = df["Winner"].map(_td_name_to_key)
    out["loser_key"] = df["Loser"].map(_td_name_to_key)

    odd_w = pd.Series(np.nan, index=df.index, dtype="float64")
    odd_l = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series("", index=df.index, dtype="object")
    for cw, cl in pairs:
        if cw in df.columns and cl in df.columns:
            w = pd.to_numeric(df[cw], errors="coerce")
            l = pd.to_numeric(df[cl], errors="coerce")
            fill = odd_w.isna() & w.notna() & l.notna()
            odd_w = odd_w.mask(fill, w)
            odd_l = odd_l.mask(fill, l)
            source = source.mask(fill, cw[:-1])  # 'PS', 'B365', 'Avg', 'Max'
    out["odd_winner"] = odd_w
    out["odd_loser"] = odd_l
    out["source"] = source
    out = out.dropna(subset=["date", "odd_winner", "odd_loser"])
    # Valid decimal odds are > 1.0 (0 / spurious values = missing data)
    out = out[(out["odd_winner"] > 1.0) & (out["odd_loser"] > 1.0)].reset_index(drop=True)
    return out


def load_odds(start: str, end: str, auto_download: bool = True,
              book: str | None = None) -> pd.DataFrame:
    """Load odds for the window [start, end] (downloads missing years).

    book: None = best available pair (PS>B365>Avg>Max); otherwise force a
    source ('PS' Pinnacle, 'B365' Bet365, 'Avg' average, 'Max' maximum).
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    years = list(range(start_ts.year, end_ts.year + 1))
    if auto_download:
        download_odds_years(years)

    frames = []
    for y in years:
        p = _xlsx_path(y)
        if not p.exists():
            continue
        try:
            frames.append(_parse_one(p, book=book))
        except Exception as e:  # noqa: BLE001 - corrupt/incomplete xlsx: skip the year
            print(f"  ! quote {y} illeggibili ({e}), salto. "
                  f"Per riscaricare: rm {p}")
    if not frames:
        raise RuntimeError(
            f"Nessuna quota disponibile per {years}. Controlla la connessione "
            "o gli anni richiesti su tennis-data.co.uk."
        )
    odds = pd.concat(frames, ignore_index=True)
    mask = (odds["date"] >= start_ts) & (odds["date"] <= end_ts)
    return odds[mask].reset_index(drop=True)


def build_pair_index(odds_df: pd.DataFrame) -> dict:
    """frozenset({winner_key, loser_key}) -> list of row dicts (for order-independent join)."""
    index: dict = {}
    for r in odds_df.itertuples(index=False):
        key = frozenset((r.winner_key, r.loser_key))
        index.setdefault(key, []).append({
            "date": r.date,
            "winner_key": r.winner_key,
            "odd_winner": float(r.odd_winner),
            "odd_loser": float(r.odd_loser),
            "source": r.source,
        })
    return index


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    df = load_odds(args.start, args.end)
    print(f"{len(df):,} match con quote tra {args.start} e {args.end}")
    print(df.head(10).to_string())
    print("\nFonti quote usate:")
    print(df["source"].value_counts().to_string())
