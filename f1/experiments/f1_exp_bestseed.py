"""Best-seed selection: train 20 seeds, pick the best on VAL precision@3, honest test."""
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
    va = feat[feat.season == val_y].copy()
    te = feat[feat.season == test_y].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte = te[cols].values
    w = np.exp(-(train_max - tr["season"].values) / 1.5)
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid test: {_precision_at3(te, 'neg_grid'):.3f}")

    results = []
    for seed in range(42, 62):
        m = CatBoostClassifier(
            iterations=1500, depth=5, learning_rate=0.03,
            boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
            l2_leaf_reg=3.0, min_data_in_leaf=5, loss_function="Logloss",
            random_seed=seed, allow_writing_files=False, verbose=False,
            early_stopping_rounds=80, task_type="CPU",
        )
        m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
        cal = IsotonicRegression(out_of_bounds="clip")
        p_va_raw = m.predict_proba(Xva)[:, 1]
        cal.fit(p_va_raw, yva)
        va["p"] = cal.predict(p_va_raw)
        te["p"] = cal.predict(m.predict_proba(Xte)[:, 1])
        # logloss val + precision@3 val + test
        from sklearn.metrics import log_loss
        ll = log_loss(yva, np.clip(va["p"].values, 1e-6, 1-1e-6))
        pv = _precision_at3(va, "p"); pt = _precision_at3(te, "p")
        results.append({"seed": seed, "val_ll": ll, "val_p": pv, "test_p": pt})
    arr = np.array([(r["seed"], r["val_ll"], r["val_p"], r["test_p"]) for r in results])
    print(f"\n  seed  val_ll  val_p  test_p")
    for r in results:
        print(f"  {r['seed']:4d}  {r['val_ll']:.4f}  {r['val_p']:.3f}  {r['test_p']:.3f}")
    # best by val precision@3 (tiebreak val_ll)
    best_pv = max(results, key=lambda x: (x["val_p"], -x["val_ll"]))
    best_ll = min(results, key=lambda x: x["val_ll"])
    print(f"\nBest by val_p: seed {best_pv['seed']} → test {best_pv['test_p']:.3f}")
    print(f"Best by val_ll: seed {best_ll['seed']} → test {best_ll['test_p']:.3f}")
    print(f"Mean test_p: {arr[:,3].mean():.3f}  std {arr[:,3].std():.3f}")


if __name__ == "__main__":
    main()
