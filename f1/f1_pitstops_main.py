"""
PitStops-main loader: stationary time (in the box, ~2-3s) per race/team/driver.
Different from Ergast pitstops (in-lane ~25s) — measures pure pit crew speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "data" / "f1" / "raw" / "pitstop" / \
       "unzipped" / "PitStops-main" / "PitStops-main"

# surname -> Ergast driverId for 2018-2026 drivers
SURNAME_TO_ID = {
    "Verstappen": "max_verstappen", "Hamilton": "hamilton", "Russell": "russell",
    "Leclerc": "leclerc", "Sainz": "sainz", "Norris": "norris", "Piastri": "piastri",
    "Perez": "perez", "Pérez": "perez", "Alonso": "alonso", "Stroll": "stroll",
    "Gasly": "gasly", "Ocon": "ocon", "Tsunoda": "tsunoda", "Hülkenberg": "hulkenberg",
    "Hulkenberg": "hulkenberg", "Bottas": "bottas", "Zhou": "zhou", "Albon": "albon",
    "Sargeant": "sargeant", "Bearman": "bearman", "Magnussen": "kevin_magnussen",
    "Ricciardo": "ricciardo", "Lawson": "lawson", "Schumacher": "mick_schumacher",
    "De Vries": "de_vries", "Latifi": "latifi", "Mazepin": "mazepin",
    "Räikkönen": "raikkonen", "Raikkonen": "raikkonen", "Giovinazzi": "giovinazzi",
    "Vettel": "vettel", "Kvyat": "kvyat", "Grosjean": "grosjean", "Gasly": "gasly",
    "Antonelli": "antonelli", "Bortoleto": "bortoleto", "Hadjar": "hadjar",
    "Doohan": "doohan", "Colapinto": "colapinto", "Lindblad": "arvid_lindblad",
}


def load_pitstops_main() -> pd.DataFrame:
    """[season, race, driver, pit_min_s, pit_avg_s, pit_n] per driver/race."""
    if not BASE.exists():
        return pd.DataFrame(columns=["season", "race", "driver",
                                     "pit_min_s", "pit_avg_s", "pit_n"])
    rows = []
    for year_dir in sorted(BASE.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for f in year_dir.glob("*.json"):
            race = f.stem
            try:
                data = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            for s in data:
                drv = SURNAME_TO_ID.get(s.get("Driver", "").strip())
                t = s.get("Time (sec)")
                if drv is None or t is None:
                    continue
                rows.append({"season": year, "race": race, "driver": drv, "t": float(t)})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return d.groupby(["season", "race", "driver"]).agg(
        pit_min_s=("t", "min"), pit_avg_s=("t", "mean"), pit_n=("t", "size"),
    ).reset_index()


if __name__ == "__main__":
    d = load_pitstops_main()
    print(f"{len(d)} righe (race-driver), piloti unici: {d.driver.nunique()}")
    print(d.head(5).to_string(index=False))
    print(f"pit_min_s: mean={d.pit_min_s.mean():.2f}s  min={d.pit_min_s.min():.2f}s")
