"""
Time-decay sample weights: recent races weigh more.
Tau in seasons. Tests multiple tau values. + XGBoost classifier head sanity check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    cols = FEATURE_COLS
    tr = feat[feat.season <= train_max].copy()
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte, yte = te[cols].values, te["podium"].values
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid: precision@3 {_precision_at3(te, 'neg_grid'):.3f}")

    for tau in [None, 5.0, 3.0, 2.0, 1.5]:
        if tau is None:
            w = None; tag = "uniform"
        else:
            age = train_max - tr["season"].values
            w = np.exp(-age / tau)
            tag = f"decay tau={tau}"
        m = CatBoostClassifier(
            iterations=1500, depth=5, learning_rate=0.03,
            boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
            l2_leaf_reg=3.0, min_data_in_leaf=5,
            loss_function="Logloss", random_seed=42,
            allow_writing_files=False, verbose=False,
            early_stopping_rounds=80, task_type="CPU",
        )
        m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(m.predict_proba(Xva)[:, 1], yva)
        te["p"] = cal.predict(m.predict_proba(Xte)[:, 1])
        print(f"CatBoost {tag:15s}: precision@3 {_precision_at3(te, 'p'):.3f}")

    # ===== XGBoost classifier head (no rank) sanity =====
    med = np.nanmedian(np.vstack([Xtr, Xva, Xte]), axis=0)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr)
    Xva2 = np.where(np.isnan(Xva), med, Xva)
    Xte2 = np.where(np.isnan(Xte), med, Xte)
    xgc = xgb.XGBClassifier(
        n_estimators=1500, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=3.0, min_child_weight=5,
        eval_metric="logloss", early_stopping_rounds=80,
        tree_method="hist", random_state=42, verbosity=0,
    )
    xgc.fit(Xtr2, ytr, eval_set=[(Xva2, yva)], verbose=False)
    cal_x = IsotonicRegression(out_of_bounds="clip")
    cal_x.fit(xgc.predict_proba(Xva2)[:, 1], yva)
    te["p_xgc"] = cal_x.predict(xgc.predict_proba(Xte2)[:, 1])
    print(f"XGBoost classifier head: precision@3 {_precision_at3(te, 'p_xgc'):.3f}")


if __name__ == "__main__":
    main()
