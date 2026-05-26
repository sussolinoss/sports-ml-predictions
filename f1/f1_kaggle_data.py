"""
Loader dataset Kaggle (RaceData-main). Fornisce con COPERTURA COMPLETA:
- standings driver post-round (clean, no rate-limit)
- safety car rate per circuit (chaos proxy, statico)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "data" / "f1" / "raw" / "pitstop" / \
       "unzipped" / "RaceData-main" / "RaceData-main" / "data_extracted"


def _load_csv(name):
    p = BASE / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, na_values=["\\N", "\\\\N"])


def load_driver_standings_full() -> pd.DataFrame:
    """[season, round, driver, k_champ_pos, k_champ_pts] (post-round, FULL coverage)."""
    races = _load_csv("races.csv")
    drv = _load_csv("drivers.csv")
    ds = _load_csv("driver_standings.csv")
    if races.empty or drv.empty or ds.empty:
        return pd.DataFrame()
    out = ds.merge(races[["raceId", "year", "round"]], on="raceId", how="left") \
            .merge(drv[["driverId", "driverRef"]], on="driverId", how="left")
    return out.rename(columns={"year": "season", "driverRef": "driver",
                               "position": "k_champ_pos", "points": "k_champ_pts"}) \
              [["season", "round", "driver", "k_champ_pos", "k_champ_pts"]]


def load_circuit_chaos_rate() -> pd.DataFrame:
    """Per circuitId, frazione gare passate con safety car deployed (chaos proxy)."""
    races = _load_csv("races.csv")
    sc = _load_csv("safety_cars.csv")
    if races.empty or sc.empty:
        return pd.DataFrame()
    # safety_cars usa Race name "1994 Japanese Grand Prix"; estraggo year+name e join
    sc[["yr_str", "race_name"]] = sc["Race"].str.extract(r"^(\d{4})\s+(.+)$")
    sc["yr_str"] = pd.to_numeric(sc["yr_str"], errors="coerce")
    sc_races = sc.dropna(subset=["yr_str"]).rename(columns={"yr_str": "year",
                                                            "race_name": "name"})
    races_sc = races.merge(sc_races[["year", "name"]].assign(_has=1),
                           on=["year", "name"], how="left")
    races_sc["_has"] = races_sc["_has"].fillna(0)
    # per ogni circuitId, frazione di gare passate con SC
    rate = races_sc.groupby("circuitId")["_has"].mean().reset_index() \
                   .rename(columns={"circuitId": "circuit_kaggle_id", "_has": "circuit_chaos_rate"})
    # mapping circuit kaggle (int) -> ergast circuitId (string) via circuits.csv
    cir = _load_csv("circuits.csv")
    if not cir.empty and "circuitRef" in cir.columns:
        rate = rate.merge(cir[["circuitId", "circuitRef"]],
                          left_on="circuit_kaggle_id", right_on="circuitId", how="left") \
                   .rename(columns={"circuitRef": "circuit"})[["circuit", "circuit_chaos_rate"]]
    return rate


if __name__ == "__main__":
    ds = load_driver_standings_full()
    print(f"standings: {len(ds)} righe, {ds.driver.nunique()} piloti, anni {ds.season.min()}-{ds.season.max()}")
    cc = load_circuit_chaos_rate()
    print(f"chaos rate per circuit: {len(cc)} circuiti")
    print(cc.sort_values("circuit_chaos_rate", ascending=False).head(8).to_string(index=False))
