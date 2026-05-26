"""
Esperimento: LightGBM LambdaRank vs CatBoost binary.
Task naturale = ranking intra-gara (NDCG@3), non binary classification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def main():
    df = load_results()
    feat = build_features(df)
    # join position per label graded
    feat = feat.merge(df[["season", "round", "driver", "position"]],
                      on=["season", "round", "driver"], how="left")
    seasons = sorted(feat["season"].unique())
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in seasons if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()

    cols = FEATURE_COLS
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte, yte = te[cols].values, te["podium"].values
    # LambdaRank label = ordinal (3=podio 1, 2=podio 2, 1=podio 3, 0=resto)
    # Approx: usiamo binary podio come label rank (2=podio, 0=no) sufficiente per NDCG@3
    # label graded da position: P1=20, P2=19, P3=18, ..., NaN/DNF=0. Range max 30 OK per LambdaRank
    def rank_label(pos):
        v = np.where(pos.notna(), np.clip(21 - pos.fillna(99).astype(int), 0, 30), 0)
        return v.astype(int)
    ltr, lva, lte = rank_label(tr["position"]), rank_label(va["position"]), rank_label(te["position"])
    gtr = tr.groupby(["season", "round"]).size().values
    gva = va.groupby(["season", "round"]).size().values
    gte = te.groupby(["season", "round"]).size().values
    print(f"groups train {len(gtr)}  val {len(gva)}  test {len(gte)}")

    m = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
        n_estimators=2000, learning_rate=0.03,
        num_leaves=31, min_data_in_leaf=10,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        lambda_l2=1.0, random_state=42, verbose=-1,
    )
    m.fit(Xtr, ltr, group=gtr, eval_set=[(Xva, lva)], eval_group=[gva],
          callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    p_te = m.predict(Xte)  # raw rank score (non probabilita')
    te["p_rank"] = p_te
    # AUC su podium binario richiede mapping; rank score si presta direttamente a precision@3
    auc = roc_auc_score(yte, p_te)  # AUC su score grezzo ok per ranking
    prec = _precision_at3(te, "p_rank")
    te["neg_grid"] = -te["grid"]
    prec_g = _precision_at3(te, "neg_grid")
    print(f"\nLGBMRanker test {test_y}:  AUC {auc:.4f}  precision@3 {prec:.3f}  vs grid {prec_g:.3f}  delta {(prec-prec_g)*100:+.1f}pt")

    # importance
    imp = sorted(zip(cols, m.feature_importances_), key=lambda x: -x[1])[:10]
    print("\nTop feature:")
    for n, gn in imp:
        print(f"  {n:24s} {gn:.0f}")


if __name__ == "__main__":
    main()
