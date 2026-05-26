"""
Manual odds loader: paste odds from bookmaker screenshots, compute edge + log.

Usage:
  python -m f1_odds_manual --book sisal --date 2026-06-12 --quote-csv quotes.csv
  python -m f1_odds_manual --report
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR

LOG = PROCESSED_DIR / "f1_paper_bets.csv"

# Mapping book name -> Ergast driverId (case-insensitive)
NAME_MAP = {
    "antonelli": "antonelli", "kimi antonelli": "antonelli", "antonelli kimi": "antonelli",
    "russell": "russell", "russell george": "russell", "george russell": "russell",
    "norris": "norris", "norris lando": "norris", "lando norris": "norris",
    "piastri": "piastri", "piastri oscar": "piastri", "oscar piastri": "piastri",
    "leclerc": "leclerc", "leclerc charles": "leclerc", "charles leclerc": "leclerc",
    "verstappen": "max_verstappen", "max verstappen": "max_verstappen",
    "hamilton": "hamilton", "hamilton lewis": "hamilton", "lewis hamilton": "hamilton",
    "sainz": "sainz", "sainz carlos": "sainz", "carlos sainz": "sainz",
    "alonso": "alonso", "fernando alonso": "alonso",
    "gasly": "gasly", "ocon": "ocon", "stroll": "stroll",
    "tsunoda": "tsunoda", "hadjar": "hadjar", "lawson": "lawson",
    "colapinto": "colapinto", "albon": "albon", "bearman": "bearman",
    "hulkenberg": "hulkenberg", "bortoleto": "bortoleto",
    "lindblad": "arvid_lindblad", "arvid lindblad": "arvid_lindblad",
}


def model_predict():
    """Predict P(wins) v2 calibrated: cross-driver softmax with T scaling."""
    import joblib
    from catboost import CatBoostClassifier
    from f1_data import load_results
    from f1_champ import FEAT_COLS, build_champ_dataset
    from f1_champ_v2 import softmax_per_group

    m = CatBoostClassifier()
    m.load_model(str(PROCESSED_DIR / "f1_champ_v2.cbm"))
    T = joblib.load(PROCESSED_DIR / "f1_champ_v2_T.pkl")["T"]
    df = load_results()
    ds = build_champ_dataset(df)
    season = int(ds.season.max())
    rnd = int(ds[ds.season == season]["round"].max())
    snap = ds[(ds.season == season) & (ds["round"] == rnd)].copy()
    logit = m.predict(snap[FEAT_COLS].values, prediction_type="RawFormulaVal")
    import numpy as np
    g = np.zeros(len(snap), dtype=int)
    p_cal = softmax_per_group(logit, g, T=T)
    snap["p_raw"] = p_cal
    snap["p_norm"] = p_cal  # already normalised
    return dict(zip(snap["driver"], zip(snap["p_raw"], snap["p_norm"]))), season, rnd


def cmd_add(book: str, date: str, quotes: dict[str, float], min_edge: float = 0.05):
    probs, season, rnd = model_predict()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    rows = []
    print(f"\n{'driver':<20}{'quota':>7}{'impl%':>7}{'model_raw%':>11}{'model_norm%':>12}"
          f"{'edge_raw':>10}{'edge_norm':>11}")
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "book", "date", "season", "round_at_bet",
                        "driver", "odd_decimal", "implied_p", "model_p_raw",
                        "model_p_norm", "edge_raw", "edge_norm", "stake",
                        "status", "settled_winner"])
        ts = datetime.now().isoformat()
        n_bet = 0
        for name_raw, odd in quotes.items():
            drv = NAME_MAP.get(name_raw.lower().strip())
            if drv is None:
                print(f"  no-map: {name_raw} (skip)")
                continue
            p_raw, p_norm = probs.get(drv, (0.0, 0.0))
            impl = 1 / odd
            edge_raw = p_raw * odd - 1
            edge_norm = p_norm * odd - 1
            mark = "BET" if edge_norm > min_edge else "   "
            print(f"  {drv:<18}{odd:>7.2f}{impl*100:>6.1f}%{p_raw*100:>10.1f}%"
                  f"{p_norm*100:>11.1f}%{edge_raw*100:>+9.1f}%{edge_norm*100:>+10.1f}%  {mark}")
            w.writerow([ts, book, date, season, rnd, drv, odd, f"{impl:.4f}",
                        f"{p_raw:.4f}", f"{p_norm:.4f}",
                        f"{edge_raw:.4f}", f"{edge_norm:.4f}",
                        1.0 if edge_norm > min_edge else 0.0,
                        "open" if edge_norm > min_edge else "no-bet", ""])
            if edge_norm > min_edge:
                n_bet += 1
    # bookmaker margin
    tot_impl = sum(1 / o for o in quotes.values())
    print(f"\nMargine bookie: {(tot_impl - 1) * 100:+.1f}%  (overround)")
    print(f"Value-bet aperte (edge_norm > {min_edge*100:.0f}%): {n_bet}")


def cmd_report():
    if not LOG.exists():
        print("Nessun log.")
        return
    df = pd.read_csv(LOG)
    bets = df[df.status == "open"]
    settled = df[df.status == "settled"]
    print(f"Open: {len(bets)}  Settled: {len(settled)}  No-bet logged: {(df.status=='no-bet').sum()}")
    if not bets.empty:
        print("\nTop edge open bet:")
        print(bets.sort_values("edge_norm", ascending=False)[
            ["date", "book", "driver", "odd_decimal", "model_p_norm", "edge_norm"]
        ].head(10).to_string(index=False))
    if not settled.empty:
        settled = settled.copy()
        settled["pl"] = settled.apply(
            lambda r: float(r.stake) * (float(r.odd_decimal) - 1)
            if r.driver == r.settled_winner else -float(r.stake), axis=1)
        stake_tot = settled.stake.astype(float).sum()
        pl = settled.pl.sum()
        roi = pl / stake_tot * 100 if stake_tot else 0
        wins = (settled.driver == settled.settled_winner).sum()
        print(f"\nSettled summary: {wins}/{len(settled)} hits  stake {stake_tot}  pl {pl:+.2f}  ROI {roi:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", nargs="+", metavar="DRIVER=ODD",
                   help="es: --add antonelli=1.55 russell=3.75 norris=12")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--book", default="manual")
    ap.add_argument("--date", default=datetime.now().date().isoformat())
    ap.add_argument("--min-edge", type=float, default=0.05)
    args = ap.parse_args()

    if args.add:
        quotes = {}
        for kv in args.add:
            if "=" not in kv:
                print(f"bad arg: {kv}"); continue
            k, v = kv.split("=", 1)
            quotes[k] = float(v)
        cmd_add(args.book, args.date, quotes, args.min_edge)
    elif args.report:
        cmd_report()


if __name__ == "__main__":
    main()
