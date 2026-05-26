"""
Passo gara e degrado gomme da FastF1 (giri delle gare, 2018+).

Per ogni gara estrae, per pilota:
  - pace_gap: mediana dei giri "puliti" - mediana del piu' veloce (secondi di
    distacco nel PASSO GARA, non sul giro singolo come la qualifica).
  - deg: degrado gomme = pendenza media (LapTime vs TyreLife) dentro gli stint
    (sec/giro; piu' alto = consuma di piu').

Questi sono dati di GARA -> usabili SOLO come feature ROLLING dalle gare PASSATE
(vedi build_features in f1_podium): cosi' restano anti-leakage. Misurano un tratto
(passo lungo, gestione gomme) che la qualifica non cattura.

Join FastF1 -> Ergast: FastF1 results ha la colonna 'DriverId' (= driverId Ergast).

Richiede:  pip install fastf1
Uso:       python -m fastf1_pace        # LENTO la prima volta (carica i giri di ~190 gare)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "f1" / "fastf1_cache"
PACE_FILE = ROOT / "data" / "f1" / "processed" / "race_pace.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PACE_FILE.parent.mkdir(parents=True, exist_ok=True)
FIRST_SEASON = 2018


def _race_pace(year: int, rnd: int) -> dict | None:
    """{driverId: {'pace_gap': s, 'deg': s/giro}} per la gara. None se non disponibile."""
    import fastf1
    try:
        fastf1.set_log_level("ERROR")
    except Exception:  # noqa: BLE001
        pass
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        res = s.results
        abbr2id = dict(zip(res["Abbreviation"], res["DriverId"]))
        laps = s.laps.pick_quicklaps()
        laps = laps[laps["LapTime"].notna()]
        if len(laps) == 0:
            return None
        laps = laps.assign(_sec=laps["LapTime"].dt.total_seconds())
        med = laps.groupby("Driver")["_sec"].median()
        best = med.min()
        out = {}
        for abbr, g in laps.groupby("Driver"):
            did = abbr2id.get(abbr)
            if not did:
                continue
            # degrado: pendenza media LapTime vs TyreLife per stint
            slopes = []
            if "Stint" in g and "TyreLife" in g:
                for _, gs in g.groupby("Stint"):
                    x = gs["TyreLife"].astype(float).values
                    y = gs["_sec"].values
                    if len(gs) >= 5 and np.std(x) > 0:
                        slopes.append(float(np.polyfit(x, y, 1)[0]))
            out[did] = {"pace_gap": float(med[abbr] - best),
                        "deg": float(np.mean(slopes)) if slopes else None}
        return out
    except Exception as e:  # noqa: BLE001
        if "RateLimit" in type(e).__name__:
            raise  # gestito dal loop di build_pace (stop con grazia)
        return None


def build_pace(years, max_round: int = 24, overwrite: bool = False) -> dict:
    data = json.loads(PACE_FILE.read_text()) if (PACE_FILE.exists() and not overwrite) else {}
    for y in years:
        if y < FIRST_SEASON:
            continue
        for rnd in range(1, max_round + 1):
            key = f"{y}-{rnd}"
            if key in data and not overwrite:
                continue
            try:
                r = _race_pace(y, rnd)
            except Exception as e:  # noqa: BLE001
                if "RateLimit" in type(e).__name__:
                    print(f"\n! Rate limit FastF1 (500/h) a {key}. "
                          f"Progresso salvato ({len(data)} gare). Riprova tra ~1h: "
                          f"riparte da qui.")
                    return data
                continue
            if r is None:
                continue
            data[key] = r
            print(f"  {key}: {len(r)} piloti")
            PACE_FILE.write_text(json.dumps(data))  # salva incrementale
    return data


def load_pace_map() -> dict:
    """{(season, round, driverId): {'pace_gap':..,'deg':..}}."""
    if not PACE_FILE.exists():
        return {}
    raw = json.loads(PACE_FILE.read_text())
    out = {}
    for k, drivers in raw.items():
        y, r = k.split("-")
        for did, vals in drivers.items():
            out[(int(y), int(r), did)] = vals
    return out


if __name__ == "__main__":
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=FIRST_SEASON,
                    help="Anno di partenza (es. 2022 per meno chiamate API)")
    args = ap.parse_args()
    years = list(range(max(FIRST_SEASON, args.since), datetime.date.today().year + 1))
    print(f"Estraggo passo/gomme {years[0]}-{years[-1]} (lento; rate limit 500/h, ripartibile)...")
    d = build_pace(years)
    print(f"\n{len(d)} gare con passo gara estratto.")
