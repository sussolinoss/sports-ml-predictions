"""
The Odds API client — UPCOMING tennis matches + pre-match odds.
Free 500 calls/month: https://the-odds-api.com/  -> register, get the API key.

  export THE_ODDS_API_KEY="your-key"   (fish: set -x THE_ODDS_API_KEY "...")

Exposes:
  upcoming_tennis(book="pinnacle", regions="eu") -> list of future matches with odds.
Each match: {event_id, sport_title, commence_time(UTC), p1, p2, odds{p1,p2}, book, link}
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

BASE = "https://api.the-odds-api.com/v4"

# Generic book links (The Odds API does not provide per-match deep links)
BOOK_HOME = {
    "pinnacle": "https://www.pinnacle.com/en/tennis",
    "betfair_ex_eu": "https://www.betfair.com/exchange/plus/tennis",
    "williamhill": "https://sports.williamhill.com/betting/en-gb/tennis",
    "unibet_eu": "https://www.unibet.com/betting/sports/filter/tennis",
}


def _key() -> str:
    k = os.environ.get("THE_ODDS_API_KEY")
    if not k:
        raise RuntimeError("THE_ODDS_API_KEY non impostata (vedi docstring odds_api.py).")
    return k


def _tennis_sport_keys() -> list[str]:
    r = requests.get(f"{BASE}/sports/", params={"apiKey": _key()}, timeout=20)
    r.raise_for_status()
    return [s["key"] for s in r.json()
            if s.get("group") == "Tennis" and s.get("active")
            and s["key"].startswith("tennis_atp")]


def _best_odd(bookmakers: list, book: str | None):
    """Return (book_key, {name: price}). If a book is specified use only that one;
    otherwise take the highest odd for each player across all books."""
    if book:
        for b in bookmakers:
            if b["key"] == book:
                for m in b["markets"]:
                    if m["key"] == "h2h":
                        return b["key"], {o["name"]: o["price"] for o in m["outcomes"]}
        return None, {}
    # best available odd for each outcome
    best: dict[str, float] = {}
    best_book = "best"
    for b in bookmakers:
        for m in b["markets"]:
            if m["key"] != "h2h":
                continue
            for o in m["outcomes"]:
                if o["price"] > best.get(o["name"], 0):
                    best[o["name"]] = o["price"]
    return best_book, best


def upcoming_tennis(book: str | None = "pinnacle", regions: str = "eu") -> list[dict]:
    out = []
    for sport in _tennis_sport_keys():
        r = requests.get(f"{BASE}/sports/{sport}/odds/", params={
            "apiKey": _key(), "regions": regions, "markets": "h2h", "oddsFormat": "decimal",
        }, timeout=20)
        if r.status_code != 200:
            continue
        for ev in r.json():
            bk, prices = _best_odd(ev.get("bookmakers", []), book)
            if len(prices) < 2:
                continue
            p1, p2 = ev["home_team"], ev["away_team"]
            if p1 not in prices or p2 not in prices:
                continue
            out.append({
                "event_id": ev["id"],
                "sport_title": ev.get("sport_title", sport),
                "sport_key": sport,
                "commence_time": datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")),
                "p1": p1, "p2": p2,
                "odds": {p1: prices[p1], p2: prices[p2]},
                "book": bk,
                "link": BOOK_HOME.get(bk, "https://the-odds-api.com"),
            })
    return out


def results(sport_key: str, days_from: int = 3) -> dict[str, str | None]:
    """event_id -> winner name (or None if not completed). days_from max 3 (free tier)."""
    r = requests.get(f"{BASE}/sports/{sport_key}/scores/", params={
        "apiKey": _key(), "daysFrom": days_from,
    }, timeout=20)
    r.raise_for_status()
    out = {}
    for ev in r.json():
        if not ev.get("completed") or not ev.get("scores"):
            out[ev["id"]] = None
            continue
        best_name, best_score = None, -1
        for s in ev["scores"]:
            try:
                sc = int(s["score"])
            except (ValueError, TypeError):
                sc = -1
            if sc > best_score:
                best_name, best_score = s["name"], sc
        out[ev["id"]] = best_name
    return out


def infer_surface_bestof(sport_title: str) -> tuple[str, int, str]:
    """Heuristic for surface/best_of/level from the tournament name."""
    t = sport_title.lower()
    slam = any(s in t for s in ["grand slam", "wimbledon", "us open", "french open",
                                "roland", "australian open"])
    if "wimbledon" in t:
        surface = "Grass"
    elif "french" in t or "roland" in t:
        surface = "Clay"
    else:
        surface = "Hard"
    return surface, (5 if slam else 3), ("G" if slam else "A")
