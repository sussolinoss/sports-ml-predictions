"""
F1 comparison: podium model WITHOUT weather vs WITH weather.

Prerequisite: build the weather data once ->  python -m fastf1_weather

Measures precision@3 (podium) and precision@1 (winner) on a held-out season,
with and without the weather features. Note: the current race's 'is_wet' is not
known pre-race (a forecast is needed); here it measures the POTENTIAL value of
weather information (upper bound). 'driver_wet_form' instead is anti-leakage.

Usage:
    python -m f1_weather_eval --test 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import load_results
from fastf1_weather import load_wet_map

NOLEAK_COLS = F.FEATURE_COLS + ["driver_wet_form"]              # deployable (anti-leakage)
SCENARIO_COLS = F.FEATURE_COLS + ["is_wet", "driver_wet_form"]   # upper bound (requires forecast)
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
          "min_child_weight": 5, "tree_method": "hist"}


def _train_eval(feat, cols, label, test_y):
    tr = feat[feat.season <= test_y - 2]
    va = feat[feat.season == test_y - 1]
    te = feat[feat.season == test_y].copy()
    dtr = xgb.DMatrix(tr[cols], label=tr[label], feature_names=cols)
    dva = xgb.DMatrix(va[cols], label=va[label], feature_names=cols)
    m = xgb.train(PARAMS, dtr, 600, evals=[(dva, "v")],
                  early_stopping_rounds=40, verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(m.predict(dva), va[label].values)
    te["p"] = cal.predict(m.predict(xgb.DMatrix(te[cols], feature_names=cols)))
    return te


def _prec_at_k(te, col, k, target_pos):
    hit = tot = 0
    for _, g in te.groupby(["season", "round"]):
        top = g.nlargest(k, col)
        hit += (top["position"] <= target_pos).sum()
        tot += k
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=2025)
    ap.add_argument("--mode", choices=["no_leak", "scenario", "both"], default="both",
                    help="no_leak = solo driver_wet_form (deployabile); "
                         "scenario = +is_wet (upper bound, serve forecast)")
    args = ap.parse_args()

    wet_map = load_wet_map()
    if not wet_map:
        print("Nessun meteo: esegui prima  python -m fastf1_weather")
        return
    print(f"Meteo: {len(wet_map)} gare, {sum(wet_map.values())} bagnate.")

    df = load_results()
    pos = df[["season", "round", "driver", "position"]]

    fb = F.build_features(df).merge(pos, on=["season", "round", "driver"], how="left")
    teb = _train_eval(fb, F.FEATURE_COLS, "podium", args.test)
    teb["ng"] = -teb.grid
    print(f"\nbaseline griglia      precision@3 {_prec_at_k(teb,'ng',3,3):.3f}")
    print(f"modello SENZA meteo   AUC {roc_auc_score(teb.podium,teb.p):.3f}  "
          f"precision@3 {_prec_at_k(teb,'p',3,3):.3f}")

    fw = F.build_features(df, wet_map).merge(pos, on=["season", "round", "driver"], how="left")
    if args.mode in ("no_leak", "both"):
        te = _train_eval(fw, NOLEAK_COLS, "podium", args.test)
        print(f"modello +meteo NO-LEAK AUC {roc_auc_score(te.podium,te.p):.3f}  "
              f"precision@3 {_prec_at_k(te,'p',3,3):.3f}  (driver_wet_form, deployabile)")
    if args.mode in ("scenario", "both"):
        te = _train_eval(fw, SCENARIO_COLS, "podium", args.test)
        print(f"modello +meteo SCENARIO AUC {roc_auc_score(te.podium,te.p):.3f}  "
              f"precision@3 {_prec_at_k(te,'p',3,3):.3f}  (con is_wet = UPPER BOUND, serve forecast)")


if __name__ == "__main__":
    main()
