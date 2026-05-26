"""Multi-seed: 25-feature full vs 19-feature no-form. Decay tau=1.5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features

FORM = ["driver_form", "driver_form_pts", "driver_podium_rate",
        "driver_dnf_rate", "constructor_form", "constructor_podium_rate"]


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid: {_precision_at3(te, 'neg_grid'):.3f}")
    w = np.exp(-(train_max - tr["season"].values) / 1.5)

    for cols, tag in [(FEATURE_COLS, "FULL 25"),
                      ([c for c in FEATURE_COLS if c not in FORM], "no-form 19")]:
        Xtr, ytr = tr[cols].values, tr["podium"].values
        Xva, yva = va[cols].values, va["podium"].values
        Xte = te[cols].values
        rows = []
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
            p_va = cal.fit_transform(m.predict_proba(Xva)[:, 1], yva)
            p_te = cal.predict(m.predict_proba(Xte)[:, 1])
            from sklearn.metrics import log_loss
            ll = log_loss(yva, np.clip(p_va, 1e-6, 1-1e-6))
            va_c = va.copy(); va_c["p"] = p_va
            te_c = te.copy(); te_c["p"] = p_te
            rows.append({"seed": seed, "val_ll": ll,
                         "val_p": _precision_at3(va_c, "p"),
                         "test_p": _precision_at3(te_c, "p")})
        arr = np.array([(r["val_ll"], r["val_p"], r["test_p"]) for r in rows])
        best_pv = max(rows, key=lambda x: (x["val_p"], -x["val_ll"]))
        best_ll = min(rows, key=lambda x: x["val_ll"])
        print(f"\n{tag}: mean test {arr[:,2].mean():.3f}±{arr[:,2].std():.3f}  "
              f"min {arr[:,2].min():.3f}  max {arr[:,2].max():.3f}")
        print(f"  best by val_p   seed {best_pv['seed']} → test {best_pv['test_p']:.3f}")
        print(f"  best by val_ll  seed {best_ll['seed']} → test {best_ll['test_p']:.3f}")


if __name__ == "__main__":
    main()
