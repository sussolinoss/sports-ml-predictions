"""
Single-race podium paper trading. Uses the f1_podium P(podium) model (0.833 test).

Usage:
  # Compute P(podium) for the next race (current round +1)
  python -m f1_race_paper --predict

  # Log bookmaker odds for the "podium finish" (top-3) market per race
  python -m f1_race_paper --book sisal --race "spanish_gp_2026" \\
      --add antonelli=1.40 norris=1.50 russell=2.00 piastri=2.50 ...

  # Cumulative ROI report
  python -m f1_race_paper --report

Target market: "Podium Finish" / "Top 3" per driver (individual odds for each driver
to finish on the podium). Different from "Race Winner" (a single winner).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR, load_results
from f1_odds_manual import NAME_MAP
from f1_podium import FEATURE_COLS, build_features

LOG = PROCESSED_DIR / "f1_race_paper.csv"
MIN_EDGE = 0.05


def predict_next_race():
    """Load model + predict P(podium) for the latest available race (proxy for 'next')."""
    m = CatBoostClassifier()
    m.load_model(str(PROCESSED_DIR / "f1_podium.cbm"))
    cal = joblib.load(PROCESSED_DIR / "f1_calibrator.pkl")
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    cur_season = int(feat.season.max())
    cur_round = int(feat[feat.season == cur_season]["round"].max())
    snap = feat[(feat.season == cur_season) & (feat["round"] == cur_round)].copy()
    p = cal.predict(m.predict_proba(snap[FEATURE_COLS].values)[:, 1])
    snap["p_podium"] = p
    snap = snap.sort_values("p_podium", ascending=False)
    return snap, cur_season, cur_round


def cmd_predict():
    snap, season, rnd = predict_next_race()
    print(f"\nP(podio) modello — ultima gara analizzata: {season} round {rnd}")
    print(f"(per gara FUTURA: stessa feature finche' non hai quali, poi rilancia)\n")
    print(f"{'driver':<20}{'grid':>5}{'P(podio)':>11}")
    for _, r in snap.iterrows():
        print(f"  {r['driver']:<18}{int(r['grid']):>5}{r['p_podium']*100:>10.1f}%")


def cmd_add(book: str, race: str, quotes: dict[str, float], min_edge: float):
    snap, season, rnd = predict_next_race()
    probs = dict(zip(snap["driver"], snap["p_podium"]))

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    print(f"\nUsing modello P(podio) per {season} R{rnd} (rilancia training se gara nuova!)\n")
    print(f"{'driver':<20}{'quota':>7}{'impl%':>7}{'model_p':>10}{'edge':>9}")
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "book", "race", "season", "round", "driver",
                        "odd_decimal", "implied_p", "model_p", "edge", "stake",
                        "status", "actual_podium"])
        ts = datetime.now().isoformat()
        n_bet = 0
        for name_raw, odd in quotes.items():
            drv = NAME_MAP.get(name_raw.lower().strip())
            if drv is None:
                print(f"  no-map: {name_raw}"); continue
            p = float(probs.get(drv, 0.0))
            impl = 1 / odd
            edge = p * odd - 1
            mark = "BET" if edge > min_edge else "   "
            print(f"  {drv:<18}{odd:>7.2f}{impl*100:>6.1f}%{p*100:>9.1f}%"
                  f"{edge*100:>+8.1f}%  {mark}")
            w.writerow([ts, book, race, season, rnd, drv, odd, f"{impl:.4f}",
                        f"{p:.4f}", f"{edge:.4f}",
                        5.0 if edge > min_edge else 0.0,  # default stake 5 EUR/bet
                        "open" if edge > min_edge else "no-bet", ""])
            if edge > min_edge:
                n_bet += 1
    tot_impl = sum(1/o for o in quotes.values())
    print(f"\nMargine bookie (podio): {(tot_impl - 3) * 100:+.1f}%  (target ~10-15%)")
    print(f"Value-bet aperte (edge > {min_edge*100:.0f}%): {n_bet}")


def cmd_settle(race: str, podium: list[str]):
    """Mark race as settled. podium = list of 3 driver_ids (the actual podium)."""
    if not LOG.exists():
        print("Nessun log"); return
    df = pd.read_csv(LOG)
    mask = (df.race == race) & (df.status == "open")
    pod_set = set(podium)
    df.loc[mask, "status"] = "settled"
    df.loc[mask, "actual_podium"] = df.loc[mask, "driver"].apply(
        lambda d: 1 if d in pod_set else 0)
    df.to_csv(LOG, index=False)
    print(f"Settled {mask.sum()} bet per race {race}, podium = {podium}")


def cmd_report():
    if not LOG.exists():
        print("Nessun log"); return
    df = pd.read_csv(LOG)
    open_ = df[df.status == "open"]
    settled = df[df.status == "settled"]
    print(f"Open: {len(open_)}  Settled: {len(settled)}")
    if not open_.empty:
        print("\nOpen value-bet (top edge):")
        print(open_.sort_values("edge", ascending=False)[
            ["race", "driver", "odd_decimal", "model_p", "edge", "stake"]
        ].head(10).to_string(index=False))
    if not settled.empty:
        settled = settled.copy()
        settled["pl"] = settled.apply(
            lambda r: float(r.stake) * (float(r.odd_decimal) - 1)
            if int(r.actual_podium) == 1 else -float(r.stake), axis=1)
        stake_tot = settled.stake.astype(float).sum()
        pl = settled.pl.sum()
        roi = pl / stake_tot * 100 if stake_tot else 0
        wins = (settled.actual_podium.astype(int) == 1).sum()
        print(f"\nSettled: {wins}/{len(settled)} hits  stake {stake_tot:.1f}€  "
              f"pl {pl:+.2f}€  ROI {roi:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--predict", action="store_true")
    g.add_argument("--add", nargs="+", metavar="DRIVER=ODD")
    g.add_argument("--settle", nargs="+", metavar="DRIVER",
                   help="--settle race_id driver1 driver2 driver3 (i 3 a podio)")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--book", default="sisal")
    ap.add_argument("--race", default="")
    ap.add_argument("--min-edge", type=float, default=0.05)
    args = ap.parse_args()

    if args.predict:
        cmd_predict()
    elif args.add:
        if not args.race:
            print("--race richiesto con --add"); return
        quotes = {}
        for kv in args.add:
            k, v = kv.split("=", 1)
            quotes[k] = float(v)
        cmd_add(args.book, args.race, quotes, args.min_edge)
    elif args.settle:
        if len(args.settle) < 4:
            print("formato: --settle race_id drv1 drv2 drv3"); return
        cmd_settle(args.settle[0], args.settle[1:4])
    elif args.report:
        cmd_report()


if __name__ == "__main__":
    main()
