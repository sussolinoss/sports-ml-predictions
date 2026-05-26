"""
HistGradientBoosting + ExtraTrees + CatBoost with polynomial interactions
on the top features (grid * constructor_form, quali_gap * driver_recovery, etc).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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

    # ===== feature interaction =====
    INT = [
        ("grid_x_cf", "grid", "constructor_form"),
        ("grid_x_pr", "grid", "driver_podium_rate"),
        ("qg_x_dr", "quali_gap_ms", "driver_recovery"),
        ("dpr_x_cpr", "driver_podium_rate", "constructor_podium_rate"),
        ("tg_x_dttavg", "constructor_track_type_avg", "driver_track_type_avg"),
    ]
    for new, a, b in INT:
        feat[new] = feat[a].astype(float) * feat[b].astype(float)
    cols = FEATURE_COLS + [n for n, _, _ in INT]
    print(f"feature set: {len(cols)} (base {len(FEATURE_COLS)} + {len(INT)} interactions)")

    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte, yte = te[cols].values, te["podium"].values
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid:              precision@3 {_precision_at3(te, 'neg_grid'):.3f}")

    # ===== CatBoost con interactions =====
    m = CatBoostClassifier(
        iterations=1500, depth=5, learning_rate=0.03,
        boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=5,
        loss_function="Logloss", random_seed=42,
        allow_writing_files=False, verbose=False,
        early_stopping_rounds=80, task_type="CPU",
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(m.predict_proba(Xva)[:, 1], yva)
    te["p_cat"] = cal.predict(m.predict_proba(Xte)[:, 1])
    print(f"CatBoost + interactions:    precision@3 {_precision_at3(te, 'p_cat'):.3f}")

    # impute NaN per sklearn
    med = np.nanmedian(np.vstack([Xtr, Xva, Xte]), axis=0)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr)
    Xva2 = np.where(np.isnan(Xva), med, Xva)
    Xte2 = np.where(np.isnan(Xte), med, Xte)

    # ===== HistGradientBoosting =====
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.15,
    )
    hgb.fit(Xtr2, ytr)
    cal_h = IsotonicRegression(out_of_bounds="clip")
    cal_h.fit(hgb.predict_proba(Xva2)[:, 1], yva)
    te["p_hgb"] = cal_h.predict(hgb.predict_proba(Xte2)[:, 1])
    print(f"HistGradientBoosting:       precision@3 {_precision_at3(te, 'p_hgb'):.3f}")

    # ===== ExtraTrees =====
    et = ExtraTreesClassifier(
        n_estimators=800, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced",
    )
    et.fit(Xtr2, ytr)
    cal_e = IsotonicRegression(out_of_bounds="clip")
    cal_e.fit(et.predict_proba(Xva2)[:, 1], yva)
    te["p_et"] = cal_e.predict(et.predict_proba(Xte2)[:, 1])
    print(f"ExtraTrees 800:             precision@3 {_precision_at3(te, 'p_et'):.3f}")


if __name__ == "__main__":
    main()
