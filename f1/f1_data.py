"""
F1 data from Ergast (jolpica-f1 mirror, free, no install).
Downloads race results per season and loads them into a sorted DataFrame.

Cache: data/f1/raw/results_{year}.json

Run:
    python -m f1_data            # download the configured seasons and print a summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
F1_DIR = ROOT / "data" / "f1"
RAW_DIR = F1_DIR / "raw"
PROCESSED_DIR = F1_DIR / "processed"
for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASE = "https://api.jolpi.ca/ergast/f1"
FIRST_SEASON = 1950
import datetime
SEASONS = list(range(FIRST_SEASON, datetime.date.today().year + 1))
HEADERS = {"User-Agent": "f1-podium-model"}


def _get(url, params, timeout=30, retries=4):
    """GET with retry on 429 (rate limit): exponential sleep 5s,15s,45s,120s."""
    delay = 5
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code != 429:
            return r
        if attempt == retries:
            return r
        time.sleep(delay); delay *= 3


def _fetch_season_raw(year: int) -> list[dict]:
    """All season results (paginated by 100), as flat rows."""
    rows, offset, total = [], 0, 1
    while offset < total:
        r = _get(f"{BASE}/{year}/results/",
                         params={"limit": 100, "offset": offset}, timeout=30)
        r.raise_for_status()
        md = r.json()["MRData"]
        total = int(md["total"])
        for race in md["RaceTable"]["Races"]:
            for res in race.get("Results", []):
                if "position" not in res or "Driver" not in res:
                    continue
                fl = res.get("FastestLap") or {}
                tm = res.get("Time") or {}
                drv = res["Driver"]; ct = res["Constructor"]
                circuit = race["Circuit"]
                rows.append({
                    "season": int(year),
                    "round": int(race["round"]),
                    "date": race["date"],
                    "circuit": circuit["circuitId"],
                    "race": race["raceName"],
                    "country": circuit.get("Location", {}).get("country", ""),
                    "driver": drv["driverId"],
                    "driver_dob": drv.get("dateOfBirth", ""),
                    "driver_nat": drv.get("nationality", ""),
                    "constructor": ct["constructorId"],
                    "grid": int(res.get("grid", 0)),
                    "position": int(res["position"]),
                    "position_text": res.get("positionText", ""),
                    "status": res.get("status", ""),
                    "points": float(res.get("points", 0)),
                    "race_time_ms": int(tm["millis"]) if tm.get("millis") else None,
                    "fl_speed_kph": float(fl.get("AverageSpeed", {}).get("speed", 0)) or None,
                    "fl_rank": int(fl["rank"]) if fl.get("rank") else None,
                })
        offset += 100
        time.sleep(0.3)  # be gentle with the API
    return rows


def download_season(year: int, overwrite: bool = False) -> Path:
    path = RAW_DIR / f"results_{year}.json"
    if path.exists() and not overwrite:
        return path
    rows = _fetch_season_raw(year)
    path.write_text(json.dumps(rows))
    return path


def _time_to_ms(t: str):
    """'1:29.708' -> 89708 ms. '' / None -> None."""
    if not t:
        return None
    try:
        if ":" in t:
            m, s = t.split(":")
            return int((int(m) * 60 + float(s)) * 1000)
        return int(float(t) * 1000)
    except (ValueError, TypeError):
        return None


def _fetch_quali_raw(year: int) -> list[dict]:
    """Best qualifying time per driver (min of Q1/Q2/Q3)."""
    rows, offset, total = [], 0, 1
    while offset < total:
        r = _get(f"{BASE}/{year}/qualifying/",
                         params={"limit": 100, "offset": offset}, timeout=30)
        r.raise_for_status()
        md = r.json()["MRData"]
        total = int(md["total"])
        for race in md["RaceTable"]["Races"]:
            for q in race.get("QualifyingResults", []):
                times = [_time_to_ms(q.get(k, "")) for k in ("Q1", "Q2", "Q3")]
                times = [t for t in times if t]
                rows.append({
                    "season": int(year), "round": int(race["round"]),
                    "driver": q["Driver"]["driverId"],
                    "quali_position": int(q["position"]),
                    "best_ms": min(times) if times else None,
                })
        offset += 100
        time.sleep(0.3)
    return rows


def download_quali(year: int, overwrite: bool = False) -> Path:
    path = RAW_DIR / f"quali_{year}.json"
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(_fetch_quali_raw(year)))
    return path


def fetch_round_quali(year: int, rnd: int) -> dict | None:
    """Qualifying for a round (including a future one, once held): entry + grid + times.
    Returns {circuit, date, rows:[{driver, constructor, grid, best_ms}]} or None."""
    r = requests.get(f"{BASE}/{year}/{rnd}/qualifying/",
                     params={"limit": 100}, timeout=30)
    r.raise_for_status()
    races = r.json()["MRData"]["RaceTable"]["Races"]
    if not races:
        return None
    race = races[0]
    rows = []
    for q in race.get("QualifyingResults", []):
        times = [_time_to_ms(q.get(k, "")) for k in ("Q1", "Q2", "Q3")]
        times = [t for t in times if t]
        rows.append({
            "driver": q["Driver"]["driverId"],
            "constructor": q["Constructor"]["constructorId"],
            "grid": int(q["position"]),
            "best_ms": min(times) if times else None,
        })
    return {"circuit": race["Circuit"]["circuitId"], "date": race["date"], "rows": rows}


def load_quali(years=None) -> pd.DataFrame:
    """[season, round, driver, quali_gap_ms] = gap to the pole of the same race."""
    years = years or SEASONS
    frames = []
    for y in years:
        p = RAW_DIR / f"quali_{y}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if data:
                frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame(columns=["season", "round", "driver", "quali_gap_ms"])
    q = pd.concat(frames, ignore_index=True).dropna(subset=["best_ms"])
    if "quali_position" not in q.columns:
        q["quali_position"] = float("nan")    # old cache: re-download to fill it in
    q["pole_ms"] = q.groupby(["season", "round"])["best_ms"].transform("min")
    q["quali_gap_ms"] = q["best_ms"] - q["pole_ms"]
    return q[["season", "round", "driver", "quali_gap_ms", "best_ms", "quali_position"]]


def _teammate_gap(df: pd.DataFrame) -> pd.Series:
    """Per driver: best_ms - best best_ms of TEAMMATES (same constructor/race).
    Negative = faster than the teammate. NaN if no teammate or no time."""
    out = pd.Series(float("nan"), index=df.index)
    for _, idx in df.groupby(["season", "round", "constructor"]).groups.items():
        bests = df.loc[idx, "best_ms"]
        for i in idx:
            mine = df.at[i, "best_ms"]
            others = bests.drop(i).dropna()
            if pd.notna(mine) and len(others):
                out.at[i] = mine - others.min()
    return out


def _fetch_sprint_raw(year: int) -> list[dict]:
    """Season sprint results (empty if the season has no sprints)."""
    rows, offset, total = [], 0, 1
    while offset < total:
        r = _get(f"{BASE}/{year}/sprint/",
                         params={"limit": 100, "offset": offset}, timeout=30)
        if r.status_code != 200:
            return rows
        md = r.json()["MRData"]
        total = int(md["total"])
        for race in md["RaceTable"]["Races"]:
            for s in race.get("SprintResults", []):
                rows.append({
                    "season": int(year), "round": int(race["round"]),
                    "driver": s["Driver"]["driverId"],
                    "sprint_position": int(s["position"]),
                    "sprint_points": float(s.get("points", 0)),
                })
        offset += 100
        time.sleep(0.3)
    return rows


def download_sprint(year: int, overwrite: bool = False) -> Path:
    path = RAW_DIR / f"sprint_{year}.json"
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(_fetch_sprint_raw(year)))
    return path


def _fetch_standings_raw(year: int, max_round: int = 25) -> list[dict]:
    """Driver standings after EVERY round of the season."""
    rows = []
    for rnd in range(1, max_round + 1):
        r = _get(f"{BASE}/{year}/{rnd}/driverStandings/",
                         params={"limit": 100}, timeout=30)
        if r.status_code != 200:
            continue
        sl = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if not sl:
            time.sleep(0.2); continue
        for s in sl[0].get("DriverStandings", []):
            if "position" not in s or "Driver" not in s:
                continue
            rows.append({
                "season": int(year), "round": int(rnd),
                "driver": s["Driver"]["driverId"],
                "champ_pos": int(s["position"]),
                "champ_pts": float(s.get("points", 0)),
            })
        time.sleep(0.2)
    return rows


def download_standings(year: int, overwrite: bool = False) -> Path:
    path = RAW_DIR / f"standings_{year}.json"
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(_fetch_standings_raw(year)))
    return path


def load_standings(years=None) -> pd.DataFrame:
    """[season, round, driver, champ_pos, champ_pts] = standings AFTER that round."""
    years = years or SEASONS
    frames = []
    for y in years:
        p = RAW_DIR / f"standings_{y}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if data:
                frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame(columns=["season", "round", "driver", "champ_pos", "champ_pts"])
    return pd.concat(frames, ignore_index=True)


def _fetch_pitstops_raw(year: int, max_round: int = 25) -> list[dict]:
    """Pit stops per round of the season (duration in ms)."""
    rows = []
    for rnd in range(1, max_round + 1):
        r = _get(f"{BASE}/{year}/{rnd}/pitstops/", params={"limit": 200}, timeout=30)
        if r.status_code != 200:
            continue
        races = r.json()["MRData"]["RaceTable"]["Races"]
        if not races:
            time.sleep(0.15); continue
        for ps in races[0].get("PitStops", []):
            d = ps.get("duration", "0")
            try:
                if ":" in d:
                    m, s = d.split(":"); dur_ms = int((int(m) * 60 + float(s)) * 1000)
                else:
                    dur_ms = int(float(d) * 1000)
            except (ValueError, TypeError):
                continue
            rows.append({"season": int(year), "round": int(rnd),
                         "driver": ps["driverId"], "stop": int(ps["stop"]),
                         "lap": int(ps["lap"]), "duration_ms": dur_ms})
        time.sleep(0.15)
    return rows


def download_pitstops(year: int, overwrite: bool = False) -> Path:
    path = RAW_DIR / f"pitstops_{year}.json"
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(_fetch_pitstops_raw(year)))
    return path


def load_pitstops(years=None) -> pd.DataFrame:
    """Aggregate by (season, round, driver): n_stops, avg, min duration."""
    years = years or SEASONS
    frames = []
    for y in years:
        p = RAW_DIR / f"pitstops_{y}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if data:
                frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame(columns=["season", "round", "driver", "n_stops",
                                     "avg_pit_ms", "min_pit_ms"])
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["season", "round", "driver"]).agg(
        n_stops=("stop", "max"),
        avg_pit_ms=("duration_ms", "mean"),
        min_pit_ms=("duration_ms", "min"),
    ).reset_index()


def load_sprint(years=None) -> pd.DataFrame:
    years = years or SEASONS
    frames = []
    for y in years:
        p = RAW_DIR / f"sprint_{y}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if data:
                frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame(columns=["season", "round", "driver", "sprint_position", "sprint_points"])
    return pd.concat(frames, ignore_index=True)


def download_all(years=None, refresh_recent: int = 1) -> None:
    years = years or SEASONS
    refresh = set(years[-refresh_recent:]) if refresh_recent else set()
    for y in years:
        try:
            download_season(y, overwrite=(y in refresh))
            download_quali(y, overwrite=(y in refresh))
            download_sprint(y, overwrite=(y in refresh))
            download_standings(y, overwrite=(y in refresh))
            download_pitstops(y, overwrite=(y in refresh))
            print(f"  {y} ok")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {y} fallito: {e}")


def load_results(years=None) -> pd.DataFrame:
    years = years or SEASONS
    frames = []
    for y in years:
        path = RAW_DIR / f"results_{y}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data:
                frames.append(pd.DataFrame(data))
    if not frames:
        raise RuntimeError("Nessun dato F1. Esegui download_all() prima.")
    df = pd.concat(frames, ignore_index=True)
    # Old cache missing new columns: add as NaN
    for c in ["race_time_ms", "fl_speed_kph", "fl_rank"]:
        if c not in df.columns:
            df[c] = float("nan")
    # Time gap to the winner per race (winner_time = min race_time_ms)
    df["winner_time_ms"] = df.groupby(["season", "round"])["race_time_ms"].transform("min")
    df["time_gap_ms"] = df["race_time_ms"] - df["winner_time_ms"]
    df["date"] = pd.to_datetime(df["date"])
    # DNF / not classified: non-numeric positionText (R, W, D, E, F, N)
    df["finished"] = df["status"].str.contains("Finished", na=False) | \
        df["status"].str.contains(r"\+\d+ Lap", regex=True, na=False)
    df["podium"] = (df["position"] <= 3).astype(int)
    # driver age at the race (years)
    if "driver_dob" in df.columns:
        dob = pd.to_datetime(df["driver_dob"], errors="coerce")
        df["age_years"] = ((df["date"] - dob).dt.days / 365.25).round(2)
    else:
        df["age_years"] = float("nan")
    # home race: driver nationality matches the race country (cheap binary)
    NAT_TO_CTRY = {"British": "UK", "Italian": "Italy", "Dutch": "Netherlands",
                   "Spanish": "Spain", "French": "France", "German": "Germany",
                   "Mexican": "Mexico", "Brazilian": "Brazil", "Australian": "Australia",
                   "Japanese": "Japan", "Finnish": "Finland", "Danish": "Denmark",
                   "American": "USA", "Monegasque": "Monaco", "Canadian": "Canada",
                   "Thai": "Thailand", "Argentine": "Argentina", "Belgian": "Belgium",
                   "Russian": "Russia", "Polish": "Poland", "Chinese": "China",
                   "Austrian": "Austria", "Swedish": "Sweden", "New Zealander": "New Zealand"}
    if "driver_nat" in df.columns and "country" in df.columns:
        df["home_race"] = ((df["driver_nat"].map(NAT_TO_CTRY).fillna(df["driver_nat"]))
                           == df["country"]).astype(int)
    else:
        df["home_race"] = 0
    # merge standings: for each row take the standings AFTER the PREVIOUS round
    # (pre-race, anti-leakage). NaN for round 1 (no previous in the same year).
    st = load_standings(years)
    if len(st):
        st_prev = st.rename(columns={"round": "_st_round"}).assign(round=lambda d: d["_st_round"] + 1)
        st_prev = st_prev.drop(columns=["_st_round"])
        df = df.merge(st_prev, on=["season", "round", "driver"], how="left")
    else:
        df["champ_pos"] = float("nan"); df["champ_pts"] = float("nan")
    # merge qualifying gap (pre-race, no leakage); NaN where missing
    q = load_quali(years)
    if len(q):
        df = df.merge(q, on=["season", "round", "driver"], how="left")
        df["teammate_gap_ms"] = _teammate_gap(df)
        # Grid penalty = grid - quali_position (positive = moved back by a penalty)
        df["grid_penalty"] = (df["grid"] - df["quali_position"]).fillna(0)
        df = df.drop(columns=["best_ms"])
    else:
        df["quali_gap_ms"] = float("nan")
        df["teammate_gap_ms"] = float("nan")
        df["grid_penalty"] = 0.0
    # merge sprint (pre-race: sprint runs Saturday, race Sunday); NaN if no sprint that weekend
    sp = load_sprint(years)
    if len(sp):
        df = df.merge(sp, on=["season", "round", "driver"], how="left")
    else:
        df["sprint_position"] = float("nan")
        df["sprint_points"] = float("nan")
    # merge CSV strategy (race-day, rolling) + PitStops-main (crew stationary time)
    try:
        from f1_pitstop_csv import load_strategy_agg
        sa = load_strategy_agg()
    except Exception:
        sa = pd.DataFrame()
    if len(sa):
        df = df.merge(sa.rename(columns={"driver_id": "driver"}),
                      on=["season", "race", "driver"], how="left")
    else:
        for c in ("n_stops", "avg_deg", "used_soft"):
            df[c] = float("nan")
    try:
        from f1_pitstops_main import load_pitstops_main
        pm = load_pitstops_main()
    except Exception:
        pm = pd.DataFrame()
    if len(pm):
        df = df.merge(pm, on=["season", "race", "driver"], how="left")
    else:
        for c in ("pit_min_s", "pit_avg_s", "pit_n"):
            df[c] = float("nan")
    # === KAGGLE: standings full coverage (anti-leakage: round-1 = pre-race) + chaos rate
    try:
        from f1_kaggle_data import load_driver_standings_full, load_circuit_chaos_rate
        kst = load_driver_standings_full()
        kcc = load_circuit_chaos_rate()
    except Exception:
        kst = pd.DataFrame(); kcc = pd.DataFrame()
    if len(kst):
        kprev = kst.rename(columns={"round": "_r"}).assign(round=lambda d: d["_r"] + 1) \
                   .drop(columns=["_r"])
        df = df.merge(kprev, on=["season", "round", "driver"], how="left")
    else:
        df["k_champ_pos"] = float("nan"); df["k_champ_pts"] = float("nan")
    if len(kcc):
        df = df.merge(kcc, on="circuit", how="left")
    else:
        df["circuit_chaos_rate"] = float("nan")
    df = df.sort_values(["date", "round", "position"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"Scarico stagioni F1 {SEASONS[0]}-{SEASONS[-1]}...")
    download_all()
    df = load_results()
    print(f"\n{len(df):,} risultati, {df['season'].nunique()} stagioni, "
          f"{df['driver'].nunique()} piloti")
    print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Tasso podio: {df['podium'].mean():.3f} (atteso ~0.15 = 3/20)")
