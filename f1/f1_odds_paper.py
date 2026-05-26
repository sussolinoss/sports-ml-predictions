"""
F1 outright paper trading: driver championship odds from The Odds API.
Compares against the P(top-3) model and logs bets with positive edge.

Setup:
  1. Sign up for free at https://the-odds-api.com (500 calls/month free)
  2. export THE_ODDS_API_KEY="your-key"
  3. python -m f1_odds_paper --fetch           # fetch odds + log bets
  4. python -m f1_odds_paper --roi             # cumulative ROI + report

Supported markets:
  - outrights (Drivers Championship Winner) - season odds
  - Future: race winner for a single race

Paper strategy:
  - Compute edge: model_p * decimal_odd - 1
  - Bet if edge > MIN_EDGE (default 5%)
  - Flat stake of 1 unit (fractional Kelly optional)
  - Settle: at season end, hit = driver wins the championship (for outright winner)
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR

BASE = "https://api.the-odds-api.com/v4"
SPORT = "motorsport_f1"
LOG_FILE = PROCESSED_DIR / "f1_paper_bets.csv"
MIN_EDGE = 0.05  # minimum 5% edge


def _key() -> str:
    k = os.environ.get("THE_ODDS_API_KEY")
    if not k:
        raise RuntimeError("THE_ODDS_API_KEY non impostata. Signup su the-odds-api.com")
    return k


def fetch_outrights(regions: str = "eu", book: str | None = None) -> dict:
    """Fetch F1 outright odds (Drivers Championship Winner).
    Returns {event_title: {driver_name: best_odd}}."""
    r = requests.get(
        f"{BASE}/sports/{SPORT}/odds/",
        params={"apiKey": _key(), "regions": regions, "markets": "outrights",
                "oddsFormat": "decimal"},
        timeout=30,
    )
    r.raise_for_status()
    used = r.headers.get("x-requests-used", "?")
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"API: used {used}, remaining {remaining}")
    events = r.json()
    out = {}
    for ev in events:
        title = ev.get("sport_title", "") + " | " + ev.get("commence_time", "")[:10]
        # for each outcome (driver), take the BEST odds across available books (or a specific one)
        best: dict[str, tuple[float, str]] = {}
        for b in ev.get("bookmakers", []):
            if book and b.get("key") != book:
                continue
            for m in b.get("markets", []):
                if m.get("key") != "outrights":
                    continue
                for o in m.get("outcomes", []):
                    nm = o.get("name", "").lower()
                    pr = float(o.get("price", 0))
                    if pr <= 1.01:
                        continue
                    if nm not in best or pr > best[nm][0]:
                        best[nm] = (pr, b.get("key"))
        if best:
            out[title] = best
    return out


# Mapping bookmaker name -> Ergast driverId
NAME_MAP = {
    "max verstappen": "max_verstappen", "verstappen": "max_verstappen",
    "lando norris": "norris", "norris": "norris",
    "oscar piastri": "piastri", "piastri": "piastri",
    "charles leclerc": "leclerc", "leclerc": "leclerc",
    "carlos sainz": "sainz", "sainz": "sainz",
    "lewis hamilton": "hamilton", "hamilton": "hamilton",
    "george russell": "russell", "russell": "russell",
    "andrea kimi antonelli": "antonelli", "kimi antonelli": "antonelli",
    "antonelli": "antonelli",
    "fernando alonso": "alonso", "alonso": "alonso",
    "lance stroll": "stroll", "stroll": "stroll",
    "pierre gasly": "gasly", "gasly": "gasly",
    "esteban ocon": "ocon", "ocon": "ocon",
    "yuki tsunoda": "tsunoda", "tsunoda": "tsunoda",
    "isack hadjar": "hadjar", "hadjar": "hadjar",
    "liam lawson": "lawson", "lawson": "lawson",
    "franco colapinto": "colapinto", "colapinto": "colapinto",
    "alexander albon": "albon", "albon": "albon",
    "oliver bearman": "bearman", "bearman": "bearman",
    "nico hulkenberg": "hulkenberg", "hulkenberg": "hulkenberg",
    "gabriel bortoleto": "bortoleto", "bortoleto": "bortoleto",
    "arvid lindblad": "arvid_lindblad", "lindblad": "arvid_lindblad",
}


def model_probs() -> dict[str, float]:
    """Load the P(wins championship) model and predict on the current round."""
    import joblib
    from catboost import CatBoostClassifier
    from f1_data import load_results
    from f1_champ import FEAT_COLS, build_champ_dataset

    m = CatBoostClassifier()
    m.load_model(str(PROCESSED_DIR / "f1_champ.cbm"))
    cal = joblib.load(PROCESSED_DIR / "f1_champ_cal.pkl")
    df = load_results()
    ds = build_champ_dataset(df)
    cur_season = int(ds["season"].max())
    cur_round = int(ds[ds.season == cur_season]["round"].max())
    snap = ds[(ds.season == cur_season) & (ds["round"] == cur_round)].copy()
    p = cal.predict(m.predict_proba(snap[FEAT_COLS].values)[:, 1])
    snap["p"] = p
    # normalise cross-driver because P(wins) must sum to 1 (a single champion/year)
    snap["p_norm"] = snap["p"] / snap["p"].sum().clip(lower=1e-9)
    return dict(zip(snap["driver"], snap["p_norm"])), cur_season, cur_round


def cmd_fetch():
    print("Scarico quote outright F1...")
    odds = fetch_outrights()
    if not odds:
        print("Nessuna quota outright. Riprova fra giorni (mercato chiuso?).")
        return
    print(f"Eventi outright: {len(odds)}")

    print("\nCarico modello P(vince campionato)...")
    probs, season, rnd = model_probs()
    print(f"Predict per stagione {season} dopo round {rnd}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "season", "round_at_bet", "event",
                        "driver", "odd_decimal", "book", "model_p", "edge",
                        "stake", "status", "settled_winner"])
        ts = datetime.now(timezone.utc).isoformat()
        n_bets = 0; n_value = 0
        for event, drivers in odds.items():
            for name, (odd, book) in drivers.items():
                drv_id = NAME_MAP.get(name.lower())
                if drv_id is None:
                    print(f"  no-map: {name}")
                    continue
                p = probs.get(drv_id, 0.0)
                edge = p * odd - 1
                bet = edge > MIN_EDGE
                if bet:
                    w.writerow([ts, season, rnd, event, drv_id, odd, book,
                                f"{p:.4f}", f"{edge:.4f}", 1.0, "open", ""])
                    n_bets += 1
                    print(f"  BET  {drv_id:20s} @ {odd:.2f} ({book})  p={p:.3f}  edge={edge:+.1%}")
                if edge > 0:
                    n_value += 1
        print(f"\nBet aperte loggate: {n_bets}  (value-bet totali: {n_value})")


def cmd_roi():
    if not LOG_FILE.exists():
        print("Nessun bet loggato. Esegui --fetch prima.")
        return
    import pandas as pd
    bets = pd.read_csv(LOG_FILE)
    settled = bets[bets["status"] == "settled"]
    open_bets = bets[bets["status"] == "open"]
    print(f"Bet totali: {len(bets)}  (settled: {len(settled)}  open: {len(open_bets)})")
    if settled.empty:
        print("Nessuna bet settled. Risultati attesi fine stagione.")
        # show the top open edges
        if not open_bets.empty:
            print("\nTop edge open:")
            top = open_bets.sort_values("edge", ascending=False).head(10)
            print(top[["timestamp", "driver", "odd_decimal", "model_p", "edge"]]
                  .to_string(index=False))
        return
    wins = (settled["driver"] == settled["settled_winner"]).sum()
    stake_tot = settled["stake"].astype(float).sum()
    payout = (settled.apply(
        lambda r: float(r.stake) * (float(r.odd_decimal) - 1)
        if r.driver == r.settled_winner else -float(r.stake), axis=1)).sum()
    roi = payout / stake_tot * 100 if stake_tot else 0
    print(f"\nSettled: {wins}/{len(settled)} hit  stake {stake_tot:.1f}  pl {payout:+.2f}  ROI {roi:+.1f}%")


def cmd_settle(winner_id: str, season: int):
    """Mark all season bets as settled and record the winner."""
    import pandas as pd
    bets = pd.read_csv(LOG_FILE)
    mask = (bets["season"] == season) & (bets["status"] == "open")
    bets.loc[mask, "status"] = "settled"
    bets.loc[mask, "settled_winner"] = winner_id
    bets.to_csv(LOG_FILE, index=False)
    print(f"Settled {mask.sum()} bet per season {season}, winner={winner_id}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true", help="scarica quote + log bet con edge")
    g.add_argument("--roi", action="store_true", help="report ROI cumulato")
    g.add_argument("--settle", nargs=2, metavar=("WINNER_ID", "SEASON"),
                   help="marca bet stagione come settled (es: max_verstappen 2025)")
    args = ap.parse_args()
    if args.fetch:
        cmd_fetch()
    elif args.roi:
        cmd_roi()
    elif args.settle:
        cmd_settle(args.settle[0], int(args.settle[1]))


if __name__ == "__main__":
    main()
