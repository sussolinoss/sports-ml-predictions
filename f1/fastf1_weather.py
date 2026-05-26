"""
Race weather via FastF1 (2018+ data). For each race, tells whether it rained.

LEAKAGE WARNING: rain *during the race* is not known BEFORE the race.
So:
  - `is_wet` (current race was wet) = hindsight / forecast information:
    use it only for scenario analysis or if you have a reliable forecast.
  - `driver_wet_form` (driver's historical wet-weather performance) = anti-leakage, legitimate.

Requires:  pip install fastf1   (already in requirements.txt)

Usage:
    python -m fastf1_weather        # download/cache the rain per race for 2018+
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "f1" / "fastf1_cache"
WET_FILE = ROOT / "data" / "f1" / "processed" / "race_weather.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WET_FILE.parent.mkdir(parents=True, exist_ok=True)

FIRST_WEATHER_SEASON = 2018  # FastF1 has reliable weather from 2018 on


def _race_wet_fraction(year: int, rnd: int) -> float | None:
    """Fraction of weather samples with rain during the race. None if unavailable."""
    import fastf1
    try:
        fastf1.set_log_level("ERROR")   # silence verbose INFO logs
    except Exception:  # noqa: BLE001
        pass
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=False, telemetry=False, weather=True, messages=False)
        w = s.weather_data
        if w is None or "Rainfall" not in w or len(w) == 0:
            return None
        return float(w["Rainfall"].mean())  # fraction of time with rain
    except Exception as e:  # noqa: BLE001  (missing session / network error)
        if "RateLimit" in type(e).__name__:
            raise
        return None


def build_weather(years, max_round: int = 24, overwrite: bool = False) -> dict:
    """Build/update race_weather.json: {"season-round": wet_fraction}."""
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
            WET_FILE.write_text(json.dumps(data))  # incremental
    return data


def load_wet_map(wet_threshold: float = 0.10) -> dict:
    """Returns {(season, round): is_wet 0/1} from race_weather.json."""
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
