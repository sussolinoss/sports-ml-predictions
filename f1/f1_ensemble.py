"""
Ensemble + OOF stacking on the F1 podium.

Base models: XGBoost (default), LightGBM (if available), LogisticRegression.
Combinations: simple average and OOF stacking with a meta-LogReg.

Usage:  python -m f1_ensemble --test 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import load_results

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("! LightGBM non disponibile, salto.")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("! CatBoost non disponibile, salto.")

SEED = 42
import os
USE_GPU = os.environ.get("F1_GPU", "1") == "1"   # F1_GPU=0 to force CPU
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss",
              "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85,
              "colsample_bytree": 0.85, "min_child_weight": 5, "tree_method": "hist",
              "device": "cuda" if USE_GPU else "cpu",
              "verbosity": 0, "seed": SEED}
LGB_PARAMS = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
              "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.85,
              "bagging_fraction": 0.85, "min_child_samples": 5, "seed": SEED}


def _prec3(df, col):
    h = t = 0
    for _, g in df.groupby(["season", "round"]):
        h += (g.nlargest(3, col)["podium"] == 1).sum()
        t += 3
    return h / t


def _fit_xgb(Xtr, ytr, Xva, yva):
    dtr = xgb.DMatrix(Xtr, label=ytr); dva = xgb.DMatrix(Xva, label=yva)
    m = xgb.train(XGB_PARAMS, dtr, 600, evals=[(dva, "v")],
                  early_stopping_rounds=40, verbose_eval=False)
    return lambda X: m.predict(xgb.DMatrix(X))


def _fit_lgb(Xtr, ytr, Xva, yva):
    dtr = lgb.Dataset(Xtr, label=ytr); dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=600, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return lambda X: m.predict(X, num_iteration=m.best_iteration)


def _fit_cat(Xtr, ytr, Xva, yva):
    m = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                           subsample=0.85, min_data_in_leaf=5,
                           loss_function="Logloss", eval_metric="Logloss",
                           random_seed=SEED, allow_writing_files=False, verbose=False,
                           early_stopping_rounds=40,
                           task_type="GPU" if USE_GPU else "CPU", devices="0")
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return lambda X: m.predict_proba(X)[:, 1]


def _fit_logreg(Xtr, ytr, Xva=None, yva=None):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=SEED))])
    pipe.fit(Xtr, ytr)
    return lambda X: pipe.predict_proba(X)[:, 1]


BASES = [("xgb", _fit_xgb), ("logreg", _fit_logreg)]
if HAS_LGB:
    BASES.insert(1, ("lgb", _fit_lgb))
if HAS_CAT:
    BASES.insert(-1, ("cat", _fit_cat))


def _calibrate(p_val, y_val, p_test):
    c = IsotonicRegression(out_of_bounds="clip"); c.fit(p_val, y_val)
    return c.predict(p_test), c.predict(p_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=2025)
    args = ap.parse_args()

    df = load_results()
    feat = F.build_features(df)
    cols = F.FEATURE_COLS
    tr = feat[feat.season <= args.test - 2]
    va = feat[feat.season == args.test - 1]
    te = feat[feat.season == args.test].copy()
    Xtr, ytr = tr[cols].values, tr.podium.values
    Xva, yva = va[cols].values, va.podium.values
    Xte = te[cols].values
    print(f"train {len(tr)}  val {len(va)}  test {len(te)} ({te.groupby(['season','round']).ngroups} gare)")

    # Base model + simple average ensemble
    test_preds = {}
    val_preds = {}
    for name, fit in BASES:
        f = fit(Xtr, ytr, Xva, yva)
        pv = f(Xva); pt = f(Xte)
        pt_c, pv_c = _calibrate(pv, yva, pt)
        val_preds[name] = pv_c
        test_preds[name] = pt_c
        te[f"p_{name}"] = pt_c
        te["ng"] = -te.grid
        print(f"  {name:8s}  test prec@3 {_prec3(te, f'p_{name}'):.4f}  AUC {roc_auc_score(te.podium, pt_c):.4f}")

    te["p_mean"] = np.mean([test_preds[n] for n, _ in BASES], axis=0)
    print(f"\n  ENSEMBLE media  test prec@3 {_prec3(te, 'p_mean'):.4f}  AUC {roc_auc_score(te.podium, te.p_mean):.4f}")

    # OOF stacking (5-fold)
    print("\nStacking OOF 5-fold (genera predizioni oneste sul train):")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {n: np.zeros(len(tr)) for n, _ in BASES}
    for fi, (i_tr, i_va) in enumerate(kf.split(Xtr), 1):
        for name, fit in BASES:
            f = fit(Xtr[i_tr], ytr[i_tr], Xtr[i_va], ytr[i_va])
            oof[name][i_va] = f(Xtr[i_va])
        print(f"  fold {fi}/5 done")

    # meta-LogReg on the OOF predictions -> learns the weights
    X_meta_tr = np.column_stack([oof[n] for n, _ in BASES])
    X_meta_te = np.column_stack([test_preds[n] for n, _ in BASES])  # already calibrated
    meta = LogisticRegression(max_iter=500, C=1.0, random_state=SEED).fit(X_meta_tr, ytr)
    te["p_stack"] = meta.predict_proba(X_meta_te)[:, 1]
    print(f"\n  STACKING meta-LR  test prec@3 {_prec3(te, 'p_stack'):.4f}  "
          f"AUC {roc_auc_score(te.podium, te.p_stack):.4f}")
    print(f"  pesi LogReg meta: {dict(zip([n for n,_ in BASES], np.round(meta.coef_[0], 2)))}")

    print(f"\nbaseline griglia      {_prec3(te, 'ng'):.4f}")
    print(f"XGBoost default solo  {_prec3(te, 'p_xgb'):.4f}  (riferimento 0.806)")


if __name__ == "__main__":
    main()
