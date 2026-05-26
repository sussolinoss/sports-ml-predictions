"""
Random search over hyperparameters for the F1 podium XGBoost. Scoring = val precision@3
(the metric we actually care about, not log-loss).

Usage:  python -m f1_tune --trials 100 --test 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import PROCESSED_DIR, load_results

SEED = 42

SEARCH = {
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.02, 0.03, 0.05, 0.07, 0.10],
    "min_child_weight": [1, 3, 5, 8, 12],
    "subsample":        [0.6, 0.75, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
    "reg_lambda":       [0.0, 0.5, 1.5, 3.0, 8.0],
    "reg_alpha":        [0.0, 0.3, 1.0, 3.0],
    "gamma":            [0.0, 0.1, 0.5, 1.5],
}


def _prec3(df, col):
    h = t = 0
    for _, g in df.groupby(["season", "round"]):
        h += (g.nlargest(3, col)["podium"] == 1).sum()
        t += 3
    return h / t


def _sample(rng):
    return {k: rng.choice(v) for k, v in SEARCH.items()}


def tune(trials=100, test_y=2025):
    df = load_results()
    feat = F.build_features(df)
    tr = feat[feat.season <= test_y - 2]
    va = feat[feat.season == test_y - 1]
    te = feat[feat.season == test_y].copy()
    print(f"train {len(tr)}  val {len(va)} ({va.groupby(['season','round']).ngroups} gare)  "
          f"test {len(te)} ({te.groupby(['season','round']).ngroups} gare)")

    cols = F.FEATURE_COLS
    dtr = xgb.DMatrix(tr[cols], label=tr.podium, feature_names=cols)
    dva = xgb.DMatrix(va[cols], label=va.podium, feature_names=cols)
    dte = xgb.DMatrix(te[cols], label=te.podium, feature_names=cols)

    rng = np.random.default_rng(SEED)
    base = {"objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "verbosity": 0, "seed": SEED}

    trials_out = []
    best = None
    import time as _t
    t0 = _t.time()
    for i in range(trials):
        p = {**base, **{k: (float(v) if k != "max_depth" else int(v))
                        for k, v in _sample(rng).items()}}
        m = xgb.train(p, dtr, num_boost_round=800, evals=[(dva, "v")],
                      early_stopping_rounds=40, verbose_eval=False)
        va_pred = m.predict(dva)
        va_df = va[["season", "round", "podium"]].copy()
        va_df["p"] = va_pred
        score = _prec3(va_df, "p")
        ll = log_loss(va.podium.values, np.clip(va_pred, 1e-6, 1 - 1e-6))
        trials_out.append({"score": score, "ll": ll, "iter": m.best_iteration, "p": p})
        flag = ""
        if best is None or (score, -ll) > (best["score"], -best["ll"]):
            best = trials_out[-1]
            flag = "  <-- BEST"
        print(f"  trial {i+1:3d}/{trials}  prec@3 {score:.4f}  ll {ll:.4f}  "
              f"d={p['max_depth']} lr={p['learning_rate']:.2f} it={m.best_iteration}"
              f"  [{_t.time()-t0:.0f}s]{flag}", flush=True)

    # Final retrain with the best params (on train; val = early stop)
    pb = best["p"]
    final = xgb.train(pb, dtr, num_boost_round=1500, evals=[(dva, "v")],
                      early_stopping_rounds=50, verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(final.predict(dva), va.podium.values)
    te["p"] = cal.predict(final.predict(dte))
    te["ng"] = -te.grid

    print("\n" + "=" * 60)
    print(f"BEST PARAMS (val prec@3 {best['score']:.4f}):")
    for k, v in pb.items():
        if k not in ("objective", "eval_metric", "tree_method", "verbosity", "seed"):
            print(f"  {k:18s} {v}")
    print("\nTEST 2025:")
    print(f"  baseline griglia       precision@3 {_prec3(te,'ng'):.4f}")
    print(f"  XGBoost tunato         precision@3 {_prec3(te,'p'):.4f}  "
          f"AUC {roc_auc_score(te.podium, te.p):.4f}")
    print(f"  (prima del tuning era 0.806)")

    # Save
    final.save_model(str(PROCESSED_DIR / "f1_podium_tuned.json"))
    joblib.dump(cal, PROCESSED_DIR / "f1_calibrator_tuned.pkl")
    (PROCESSED_DIR / "f1_best_params.json").write_text(json.dumps(pb, indent=2, default=str))
    print(f"\nSalvati: f1_podium_tuned.json, f1_calibrator_tuned.pkl, f1_best_params.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--test", type=int, default=2025)
    args = ap.parse_args()
    tune(args.trials, args.test)


if __name__ == "__main__":
    main()
