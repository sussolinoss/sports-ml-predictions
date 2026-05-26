"""
Blend CatBoost binary prob + XGBRanker score (rank-normalized per gara).
+ Mixture-of-experts: modello separato street vs permanent.
Reuse build_features, no TCN per velocita'.
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
from f1_podium import FEATURE_COLS, STREET_CIRCUITS, _precision_at3, build_features


def train_cat(Xtr, ytr, Xva, yva):
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
    return m, cal


def train_xgbrank(tr, va, cols):
    def rl(p):
        return np.where(p.notna(), np.clip(21 - p.fillna(99).astype(int), 0, 30), 0).astype(int)
    ltr, lva = rl(tr["position"]), rl(va["position"])
    qtr = tr.groupby(["season", "round"]).size().values
    qva = va.groupby(["season", "round"]).size().values
    dtr = xgb.DMatrix(tr[cols].values, label=ltr); dtr.set_group(qtr)
    dva = xgb.DMatrix(va[cols].values, label=lva); dva.set_group(qva)
    params = {"objective": "rank:ndcg", "eval_metric": ["ndcg@3"],
              "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
              "subsample": 0.85, "colsample_bytree": 0.85,
              "lambda": 1.0, "tree_method": "hist", "verbosity": 0}
    return xgb.train(params, dtr, num_boost_round=2000,
                     evals=[(dva, "val")], early_stopping_rounds=80, verbose_eval=0)


def per_race_rank01(df, score_col):
    """Normalizza score [0,1] per gara (rank-based)."""
    out = df.copy()
    out["_r"] = out.groupby(["season", "round"])[score_col].rank(method="average")
    sizes = out.groupby(["season", "round"])[score_col].transform("size")
    out["_n"] = (out["_r"] - 1) / (sizes - 1).clip(lower=1)
    return out["_n"].values


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.merge(df[["season", "round", "driver", "position"]],
                      on=["season", "round", "driver"], how="left")
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

    # ===== A) CatBoost base =====
    cat, cal = train_cat(tr[cols].values, tr["podium"].values,
                         va[cols].values, va["podium"].values)
    te["p_cat"] = cal.predict(cat.predict_proba(te[cols].values)[:, 1])
    prec_cat = _precision_at3(te, "p_cat")
    print(f"CatBoost only:        precision@3 {prec_cat:.3f}")

    # ===== B) XGBRanker =====
    xgr = train_xgbrank(tr, va, cols)
    te["p_xgr"] = xgr.predict(xgb.DMatrix(te[cols].values))
    prec_xgr = _precision_at3(te, "p_xgr")
    print(f"XGBRanker only:       precision@3 {prec_xgr:.3f}")

    # ===== C) BLEND rank-normalized =====
    te["p_cat_n"] = per_race_rank01(te, "p_cat")
    te["p_xgr_n"] = per_race_rank01(te, "p_xgr")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        te["p_blend"] = w * te["p_cat_n"] + (1 - w) * te["p_xgr_n"]
        prec = _precision_at3(te, "p_blend")
        print(f"Blend  w_cat={w:.1f}:        precision@3 {prec:.3f}")

    # ===== D) Mixture-of-experts: street vs permanent =====
    print("\n--- Mixture-of-experts (street vs permanent) ---")
    is_street_tr = tr["circuit_is_street"] == 1
    is_street_va = va["circuit_is_street"] == 1
    is_street_te = te["circuit_is_street"] == 1
    n_street_te = is_street_te.sum() // 20 + 1
    n_perm_te = (~is_street_te).sum() // 20 + 1
    print(f"  test: {n_street_te} street races, {n_perm_te} permanent races")

    if is_street_tr.sum() > 200 and is_street_va.sum() > 20:
        cat_s, cal_s = train_cat(tr[is_street_tr][cols].values, tr[is_street_tr]["podium"].values,
                                 va[is_street_va][cols].values, va[is_street_va]["podium"].values)
        cat_p, cal_p = train_cat(tr[~is_street_tr][cols].values, tr[~is_street_tr]["podium"].values,
                                 va[~is_street_va][cols].values, va[~is_street_va]["podium"].values)
        te["p_moe"] = np.nan
        if is_street_te.any():
            te.loc[is_street_te, "p_moe"] = cal_s.predict(cat_s.predict_proba(te[is_street_te][cols].values)[:, 1])
        if (~is_street_te).any():
            te.loc[~is_street_te, "p_moe"] = cal_p.predict(cat_p.predict_proba(te[~is_street_te][cols].values)[:, 1])
        prec_moe = _precision_at3(te, "p_moe")
        print(f"MoE street/permanent: precision@3 {prec_moe:.3f}")
    else:
        print("  skip: pochi dati street")

    te["neg_grid"] = -te["grid"]
    print(f"\nBaseline grid:        precision@3 {_precision_at3(te, 'neg_grid'):.3f}")


if __name__ == "__main__":
    main()
