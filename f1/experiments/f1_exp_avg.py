"""CatBoost(decay) multi-seed averaged probabilities (no TCN)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    cols = FEATURE_COLS
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    te["neg_grid"] = -te["grid"]
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte = te[cols].values
    w = np.exp(-(train_max - tr["season"].values) / 1.5)

    # singolo seed best (sanity)
    for n in [1, 3, 5, 10, 20]:
        probs = np.zeros(len(te))
        for seed in range(42, 42 + n):
            m = CatBoostClassifier(
                iterations=1500, depth=5, learning_rate=0.03,
                boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
                l2_leaf_reg=3.0, min_data_in_leaf=5, loss_function="Logloss",
                random_seed=seed, allow_writing_files=False, verbose=False,
                early_stopping_rounds=80, task_type="CPU",
            )
            m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(m.predict_proba(Xva)[:, 1], yva)
            probs += cal.predict(m.predict_proba(Xte)[:, 1])
        probs /= n
        te["p"] = probs
        print(f"  N={n:2d} avg seeds: precision@3 {_precision_at3(te, 'p'):.3f}")


if __name__ == "__main__":
    main()
