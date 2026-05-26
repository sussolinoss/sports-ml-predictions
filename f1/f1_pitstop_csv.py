"""
Loader del CSV strategy/pitstop fornito (Kaggle/FastF1-derived).
Aggrega lap-by-lap per (year, race, driver) -> n_stops, avg_deg, used_soft.
Mappa driver 3-letter -> Ergast driverId per merge con load_results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent / "data" / "f1" / "raw" / "pitstop" / "f1_strategy_dataset_v4.csv"

# 3-letter abbr -> Ergast driverId (piloti 2022-2025)
ABBR_TO_ID = {
    "ALB": "albon", "ALO": "alonso", "ANT": "antonelli", "BEA": "bearman",
    "BOR": "bortoleto", "BOT": "bottas", "COL": "colapinto", "DEV": "de_vries",
    "DOO": "doohan", "GAS": "gasly", "HAD": "hadjar", "HAM": "hamilton",
    "HUL": "hulkenberg", "LAT": "latifi", "LAW": "lawson", "LEC": "leclerc",
    "MAG": "kevin_magnussen", "MSC": "mick_schumacher", "NOR": "norris",
    "OCO": "ocon", "PER": "perez", "PIA": "piastri", "RIC": "ricciardo",
    "RUS": "russell", "SAI": "sainz", "SAR": "sargeant", "STR": "stroll",
    "TSU": "tsunoda", "VER": "max_verstappen", "VET": "vettel",
    "ZHO": "zhou", "LIN": "arvid_lindblad",
}


def load_strategy_agg() -> pd.DataFrame:
    """Per (season, race, driver) -> n_stops, avg_deg, used_soft.
    Le colonne sono race_name (non round) e driver_abbr.
    Restituisce DataFrame con: season, race, driver_id, n_stops, avg_deg, used_soft.
    """
    if not CSV.exists():
        return pd.DataFrame(columns=["season", "race", "driver_id", "n_stops",
                                     "avg_deg", "used_soft"])
    d = pd.read_csv(CSV)
    agg = d.groupby(["Year", "Race", "Driver"]).agg(
        n_stops=("Stint", lambda s: int(s.max()) - 1),
        avg_deg=("Cumulative_Degradation", "mean"),
        used_soft=("Compound", lambda c: int((c == "SOFT").any())),
    ).reset_index()
    agg = agg.rename(columns={"Year": "season", "Race": "race", "Driver": "abbr"})
    agg["driver_id"] = agg["abbr"].map(ABBR_TO_ID)
    return agg.dropna(subset=["driver_id"])[
        ["season", "race", "driver_id", "n_stops", "avg_deg", "used_soft"]]


if __name__ == "__main__":
    a = load_strategy_agg()
    print(f"{len(a)} righe agg (race-driver)")
    print(f"anni: {sorted(a.season.unique())}  driver mappati: {a.driver_id.nunique()}")
    print(a.head(10).to_string(index=False))
