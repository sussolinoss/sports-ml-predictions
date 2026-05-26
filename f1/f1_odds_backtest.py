"""
[SCIENTIFIC/PRIVATE USE] Does the F1 model beat the betting market?

Ingests a CSV of historical odds and measures, with the rigour used for tennis,
whether betting on the podium according to the model yields ROI > 0 (with bootstrap CI).

It is NOT a betting bot: it is a measurement tool to document the result
(without publishing the model). Liquid F1 markets are efficient: expected ~0/negative.

Expected odds CSV (decimal), columns: season, round, driver, odds
  - driver = Ergast driverId (e.g. max_verstappen)
  - odds   = decimal odds "finishes on the podium" (or wins, see --target)
Possible sources: Kaggle (historical), The Odds API motorsport_f1 (prospective).

Usage:
    python -m f1_odds_backtest --odds quote.csv --min_edge 0.05 --bootstrap 10000
    python -m f1_odds_backtest --sim          # demo with synthetic odds (no file)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import PROCESSED_DIR, load_results


def _bootstrap_roi(profit, stake, n=10000, seed=42):
    profit, stake = np.asarray(profit), np.asarray(stake)
    rng = np.random.default_rng(seed)
    rois = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(profit), len(profit))
        s = stake[idx].sum()
        rois[i] = profit[idx].sum() / s * 100 if s > 0 else 0.0
    return np.percentile(rois, 2.5), np.percentile(rois, 50), np.percentile(rois, 97.5), (rois > 0).mean()


def _model_podium_proba(df):
    """P(podium) for each (season,round,driver) using the saved model + pre-race features."""
    feat = F.build_features(df)
    model = xgb.Booster(); model.load_model(str(PROCESSED_DIR / "f1_podium.json"))
    p = model.predict(xgb.DMatrix(feat[F.FEATURE_COLS], feature_names=F.FEATURE_COLS))
    cal = PROCESSED_DIR / "f1_calibrator.pkl"
    if cal.exists():
        import joblib
        p = joblib.load(cal).predict(p)
    feat = feat[["season", "round", "driver"]].copy()
    feat["p_model"] = p
    return feat


def run(odds: pd.DataFrame, min_edge: float, n_boot: int):
    df = load_results()
    actual = df[["season", "round", "driver", "podium"]]
    pred = _model_podium_proba(df)

    m = odds.merge(pred, on=["season", "round", "driver"], how="inner") \
            .merge(actual, on=["season", "round", "driver"], how="inner")
    if len(m) == 0:
        print("Nessun match tra quote, predizioni e risultati. Controlla season/round/driver.")
        return
    m["edge"] = m["p_model"] - 1.0 / m["odds"]
    bets = m[m["edge"] > min_edge].copy()
    print(f"Righe quote agganciate: {len(m)}  |  value bet (edge>{min_edge}): {len(bets)}")
    if len(bets) == 0:
        print("Nessuna value bet con questa soglia.")
        return
    bets["stake"] = 1.0
    bets["profit"] = np.where(bets["podium"] == 1, bets["odds"] - 1.0, -1.0)
    roi = bets["profit"].sum() / bets["stake"].sum() * 100
    hit = bets["podium"].mean()
    lo, med, hi, ppos = _bootstrap_roi(bets["profit"], bets["stake"], n_boot)
    print(f"\nHit-rate bet: {hit:.3f}  quota media: {bets['odds'].mean():.2f}")
    print(f"ROI: {roi:+.2f}%   IC95% [{lo:+.2f}%, {hi:+.2f}%]   P(ROI>0)={ppos:.3f}")
    if lo > 0:
        print("=> EDGE reale (IC>0). Mercato F1 battibile sul podio (sorprendente).")
    elif hi < 0:
        print("=> PERDITA provata (IC<0): il mercato e' piu' bravo del modello.")
    else:
        print("=> BREAK-EVEN: IC contiene zero, nessun edge dimostrato.")


def _synthetic_odds(df, margin=0.08, seed=1):
    """Synthetic odds ~ historical real podium probability + margin, to demonstrate the pipeline."""
    rng = np.random.default_rng(seed)
    base = df.groupby("driver")["podium"].mean().clip(0.02, 0.95)
    rows = []
    for r in df.itertuples(index=False):
        p = float(base.get(r.driver, 0.15))
        p = min(0.95, max(0.03, p + rng.uniform(-0.05, 0.05)))
        rows.append({"season": r.season, "round": r.round, "driver": r.driver,
                     "odds": round(1.0 / (p * (1 + margin)), 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", help="CSV quote storiche (season,round,driver,odds)")
    ap.add_argument("--sim", action="store_true", help="Demo con quote sintetiche")
    ap.add_argument("--min_edge", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    if args.sim:
        odds = _synthetic_odds(load_results())
        print("DEMO con quote sintetiche (il ROI qui non significa nulla, solo pipeline).")
    elif args.odds:
        odds = pd.read_csv(args.odds)
    else:
        ap.error("Usa --odds <file.csv> oppure --sim")
    run(odds, args.min_edge, args.bootstrap)


if __name__ == "__main__":
    main()
