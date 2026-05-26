"""
Meteo gara via FastF1 (dati 2018+). Per ogni gara dice se ha piovuto.

ATTENZIONE LEAKAGE: la pioggia *durante la gara* non e' nota PRIMA della gara.
Quindi:
  - `is_wet` (gara corrente bagnata) = informazione col-senno-di-poi / da forecast:
    usala solo per analisi di scenario o se hai un forecast affidabile.
  - `driver_wet_form` (resa storica del pilota sul bagnato) = anti-leakage, legittima.

Richiede:  pip install fastf1   (gia' in requirements.txt)

Uso:
    python -m fastf1_weather        # scarica/cacha la pioggia per gara 2018+
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "f1" / "fastf1_cache"
WET_FILE = ROOT / "data" / "f1" / "processed" / "race_weather.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WET_FILE.parent.mkdir(parents=True, exist_ok=True)

FIRST_WEATHER_SEASON = 2018  # FastF1 ha meteo affidabile dal 2018


def _race_wet_fraction(year: int, rnd: int) -> float | None:
    """Frazione di campioni meteo con pioggia durante la gara. None se non disponibile."""
    import fastf1
    try:
        fastf1.set_log_level("ERROR")   # silenzia i log INFO verbosi
    except Exception:  # noqa: BLE001
        pass
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=False, telemetry=False, weather=True, messages=False)
        w = s.weather_data
        if w is None or "Rainfall" not in w or len(w) == 0:
            return None
        return float(w["Rainfall"].mean())  # frazione di tempo con pioggia
    except Exception as e:  # noqa: BLE001  (sessione mancante / errore rete)
        if "RateLimit" in type(e).__name__:
            raise
        return None


def build_weather(years, max_round: int = 24, overwrite: bool = False) -> dict:
    """Costruisce/aggiorna race_weather.json: {"season-round": wet_fraction}."""
    data = {}
    if WET_FILE.exists() and not overwrite:
        data = json.loads(WET_FILE.read_text())
    for y in years:
        if y < FIRST_WEATHER_SEASON:
            continue
        for rnd in range(1, max_round + 1):
            key = f"{y}-{rnd}"
            if key in data and not overwrite:
                continue
            try:
                frac = _race_wet_fraction(y, rnd)
            except Exception as e:  # noqa: BLE001
                if "RateLimit" in type(e).__name__:
                    print(f"\n! Rate limit FastF1 (500/h) a {key}. "
                          f"Salvato {len(data)} gare. Riprova tra ~1h.")
                    WET_FILE.write_text(json.dumps(data))
                    return data
                continue
            if frac is None:
                continue
            data[key] = frac
            print(f"  {key}: pioggia {frac:.0%}")
            WET_FILE.write_text(json.dumps(data))  # incrementale
    return data


def load_wet_map(wet_threshold: float = 0.10) -> dict:
    """Ritorna {(season, round): is_wet 0/1} da race_weather.json."""
    if not WET_FILE.exists():
        return {}
    raw = json.loads(WET_FILE.read_text())
    out = {}
    for k, frac in raw.items():
        y, r = k.split("-")
        out[(int(y), int(r))] = int(frac >= wet_threshold)
    return out


if __name__ == "__main__":
    import datetime
    years = list(range(FIRST_WEATHER_SEASON, datetime.date.today().year + 1))
    print(f"Scarico meteo gare {years[0]}-{years[-1]} (lento la prima volta)...")
    d = build_weather(years)
    wet = sum(1 for v in d.values() if v >= 0.10)
    print(f"\n{len(d)} gare con meteo, di cui {wet} bagnate (>=10% pioggia)")
