"""
Loader quote bookmaker da tennis-data.co.uk (gratuito, CSV/xlsx).

USO ESCLUSIVO: valutazione ROI nel backtest. Le quote NON entrano come feature
nel modello (il modello resta "puro" su ELO/form/rank/...). Qui scarichiamo solo
le quote di chiusura storiche per simulare le scommesse contro il mercato.

Formato tennis-data.co.uk (ATP):
  - un archivio .zip per anno: http://www.tennis-data.co.uk/{year}/{year}.zip
  - dentro un .xlsx con colonne: Date, Winner, Loser, WRank, LRank,
    B365W/B365L (Bet365), PSW/PSL (Pinnacle), MaxW/MaxL, AvgW/AvgL, ...
  - i nomi sono nel formato "Cognome I." (es. "Federer R.")

Espone:
  download_odds_years(years)         -> scarica/cacha gli archivi annuali
  load_odds(start, end) -> DataFrame [date, winner_key, loser_key, odd_winner, odd_loser, source]
  sackmann_name_to_key(full_name)    -> "cognome i." per fare il join con i nomi Sackmann
  build_pair_index(odds_df)          -> dict frozenset({key_a,key_b}) -> list di righe quote
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

# Preferenza colonne quote (dalla piu' affidabile alla fallback)
ODDS_PAIRS = [("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL"), ("MaxW", "MaxL")]


def _xlsx_path(year: int) -> Path:
    return RAW_ODDS_DIR / f"{year}.xlsx"


def _is_valid_xlsx(path: Path) -> bool:
    """xlsx e' un archivio zip: deve iniziare con la firma 'PK'."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def download_odds_year(year: int, overwrite: bool = False) -> Path | None:
    """Scarica l'archivio quote di un anno (zip o xlsx diretto). Cache su disco."""
    target = _xlsx_path(year)
    if target.exists() and not overwrite:
        if _is_valid_xlsx(target):
            return target
        target.unlink()  # cache corrotta: ri-scarica

    # Tentativo 1: zip
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

    # Tentativo 2: xlsx diretto
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
    """'Médvédev' -> 'medvedev' (fold accenti per il matching)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def sackmann_name_to_key(full_name: str) -> str:
    """
    'Roger Federer'   -> 'federer r'
    'Alex De Minaur'  -> 'de minaur a'   (cognome = tutto tranne il primo nome)
    Iniziale = prima lettera del primo token; cognome = i token restanti.
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
    'De Minaur A.'  -> 'de minaur a'   (ultimo token = iniziale, resto = cognome)
    """
    tokens = _strip_accents(str(td_name)).replace(".", " ").split()
    if len(tokens) < 2:
        return " ".join(tokens).lower()
    initial = tokens[-1][0]
    surname = " ".join(tokens[:-1])
    return f"{surname} {initial}".lower()


def implied_probs(odd_winner: float, odd_loser: float) -> tuple[float, float]:
    """Da quote decimali a probabilita' (overround rimosso). Ritorna (p_winner, p_loser)."""
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

    # Se book specificato (es. 'PS'), usa solo quella coppia; altrimenti fallback in ordine
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
    # Quote decimali valide sono > 1.0 (0 / valori spuri = dati mancanti)
    out = out[(out["odd_winner"] > 1.0) & (out["odd_loser"] > 1.0)].reset_index(drop=True)
    return out


def load_odds(start: str, end: str, auto_download: bool = True,
              book: str | None = None) -> pd.DataFrame:
    """Carica le quote per la finestra [start, end] (scarica gli anni mancanti).

    book: None = miglior coppia disponibile (PS>B365>Avg>Max); altrimenti forza
    una fonte ('PS' Pinnacle, 'B365' Bet365, 'Avg' media, 'Max' massima).
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
        except Exception as e:  # noqa: BLE001 - xlsx corrotto/incompleto: salta l'anno
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
    """frozenset({winner_key, loser_key}) -> list di dict riga (per join ordine-indipendente)."""
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
