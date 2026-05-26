"""XGBoost rank:ndcg vs LGBMRanker."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


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

    tr = feat[feat.season <= train_max]; va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    cols = FEATURE_COLS

    def rank_label(pos):
        return np.where(pos.notna(),
                        np.clip(21 - pos.fillna(99).astype(int), 0, 30), 0).astype(int)
    ltr = rank_label(tr["position"]); lva = rank_label(va["position"])
    qtr = tr.groupby(["season", "round"]).size().values
    qva = va.groupby(["season", "round"]).size().values

    dtr = xgb.DMatrix(tr[cols].values, label=ltr); dtr.set_group(qtr)
    dva = xgb.DMatrix(va[cols].values, label=lva); dva.set_group(qva)
    dte = xgb.DMatrix(te[cols].values)

    params = {
        "objective": "rank:ndcg", "eval_metric": ["ndcg@3"],
        "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "lambda": 1.0, "tree_method": "hist", "verbosity": 0,
    }
    m = xgb.train(params, dtr, num_boost_round=2000,
                  evals=[(dva, "val")], early_stopping_rounds=80, verbose_eval=0)
    p = m.predict(dte)
    te["p_rank"] = p
    prec = _precision_at3(te, "p_rank")
    te["neg_grid"] = -te["grid"]
    prec_g = _precision_at3(te, "neg_grid")
    print(f"\nXGBRanker test {test_y}: precision@3 {prec:.3f}  vs grid {prec_g:.3f}  delta {(prec-prec_g)*100:+.1f}pt  best_iter {m.best_iteration}")


if __name__ == "__main__":
    main()
