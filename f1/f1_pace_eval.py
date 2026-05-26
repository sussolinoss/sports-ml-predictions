"""
Confronto F1: modello podio SENZA vs CON passo-gara/degrado-gomme (FastF1).

Prerequisito:  python -m fastf1_pace   (estrae race_pace.json, lento la 1a volta)

Le feature driver_race_pace / driver_tyre_deg sono ROLLING dalle gare passate
-> anti-leakage. Misura se aggiungono segnale oltre quali_gap + form.

Uso:  python -m f1_pace_eval --test 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import load_results
from fastf1_pace import load_pace_map

PACE_COLS = F.FEATURE_COLS + ["driver_race_pace", "driver_tyre_deg"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
          "min_child_weight": 5, "tree_method": "hist"}


def _train_eval(feat, cols, test_y):
    tr = feat[feat.season <= test_y - 2]
    va = feat[feat.season == test_y - 1]
    te = feat[feat.season == test_y].copy()
    m = xgb.train(PARAMS, xgb.DMatrix(tr[cols], label=tr.podium, feature_names=cols), 600,
                  evals=[(xgb.DMatrix(va[cols], label=va.podium, feature_names=cols), "v")],
                  early_stopping_rounds=40, verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(m.predict(xgb.DMatrix(va[cols], feature_names=cols)), va.podium.values)
    te["p"] = cal.predict(m.predict(xgb.DMatrix(te[cols], feature_names=cols)))
    return te, m


def _prec3(te, col):
    hit = tot = 0
    for _, g in te.groupby(["season", "round"]):
        hit += (g.nlargest(3, col)["position"] <= 3).sum()
        tot += 3
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=2025)
    args = ap.parse_args()

    pace_map = load_pace_map()
    if not pace_map:
        print("Nessun passo gara: esegui prima  python -m fastf1_pace")
        return
    print(f"Passo gara caricato: {len({(s,r) for s,r,_ in pace_map})} gare.")

    df = load_results()
    pos = df[["season", "round", "driver", "position"]]

    fb = F.build_features(df).merge(pos, on=["season", "round", "driver"], how="left")
    teb, _ = _train_eval(fb, F.FEATURE_COLS, args.test)
    teb["ng"] = -teb.grid
    print(f"\nbaseline griglia     precision@3 {_prec3(teb,'ng'):.3f}")
    print(f"modello SENZA passo  AUC {roc_auc_score(teb.podium,teb.p):.3f}  precision@3 {_prec3(teb,'p'):.3f}")

    fp = F.build_features(df, pace_map=pace_map).merge(pos, on=["season", "round", "driver"], how="left")
    tep, mp = _train_eval(fp, PACE_COLS, args.test)
    print(f"modello CON passo    AUC {roc_auc_score(tep.podium,tep.p):.3f}  precision@3 {_prec3(tep,'p'):.3f}")
    print("\n--- gain feature passo/gomme ---")
    g = mp.get_score(importance_type="gain")
    for k in ("driver_race_pace", "driver_tyre_deg"):
        print(f"  {k:18s} {g.get(k, 0):.1f}")


if __name__ == "__main__":
    main()
