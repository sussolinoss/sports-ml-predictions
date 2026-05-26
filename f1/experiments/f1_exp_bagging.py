"""
1) CatBoost bagging ×20 seed (variance reduction puro)
2) LogReg sanity baseline (segnale lineare)
3) RandomForest (controllo gradient boosting vs bagging trees)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    seasons = sorted(feat.season.unique())
    complete = [y for y in seasons if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    cols = FEATURE_COLS
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte, yte = te[cols].values, te["podium"].values
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid:        precision@3 {_precision_at3(te, 'neg_grid'):.3f}")

    # ===== 1) CatBoost bagging ×20 =====
    N = 20
    probs = np.zeros(len(te))
    for seed in range(42, 42 + N):
        m = CatBoostClassifier(
            iterations=1500, depth=5, learning_rate=0.03,
            boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
            l2_leaf_reg=3.0, min_data_in_leaf=5,
            loss_function="Logloss", random_seed=seed,
            allow_writing_files=False, verbose=False,
            early_stopping_rounds=80, task_type="CPU",
        )
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(m.predict_proba(Xva)[:, 1], yva)
        probs += cal.predict(m.predict_proba(Xte)[:, 1])
    probs /= N
    te["p_bag"] = probs
    print(f"CatBoost bagging x{N}: precision@3 {_precision_at3(te, 'p_bag'):.3f}")

    # ===== 2) LogReg sanity =====
    sc = StandardScaler()
    med = np.nanmedian(np.vstack([Xtr, Xva, Xte]), axis=0)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr); Xva2 = np.where(np.isnan(Xva), med, Xva)
    Xte2 = np.where(np.isnan(Xte), med, Xte)
    Xtr2 = sc.fit_transform(Xtr2); Xva2 = sc.transform(Xva2); Xte2 = sc.transform(Xte2)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    lr.fit(Xtr2, ytr)
    cal_lr = IsotonicRegression(out_of_bounds="clip")
    cal_lr.fit(lr.predict_proba(Xva2)[:, 1], yva)
    te["p_lr"] = cal_lr.predict(lr.predict_proba(Xte2)[:, 1])
    print(f"LogReg sanity:        precision@3 {_precision_at3(te, 'p_lr'):.3f}")

    # ===== 3) RandomForest =====
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(Xtr2, ytr)
    cal_rf = IsotonicRegression(out_of_bounds="clip")
    cal_rf.fit(rf.predict_proba(Xva2)[:, 1], yva)
    te["p_rf"] = cal_rf.predict(rf.predict_proba(Xte2)[:, 1])
    print(f"RandomForest 500:     precision@3 {_precision_at3(te, 'p_rf'):.3f}")


if __name__ == "__main__":
    main()
