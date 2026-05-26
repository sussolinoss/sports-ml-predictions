"""Ablation for paper: remove feature groups + decay vs uniform. Single seed, CPU."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features

GROUPS = {
    "base":       ["grid", "quali_gap_ms", "teammate_gap_ms"],
    "form":       ["driver_form", "driver_form_pts", "driver_podium_rate",
                   "driver_dnf_rate", "constructor_form", "constructor_podium_rate"],
    "track":      ["driver_track_avg", "constructor_track_avg"],
    "lag":        ["last1_pos", "last2_pos", "last3_pos",
                   "last1_pod", "last2_pod", "last3_pod",
                   "last1_dnf", "last2_dnf", "last3_dnf"],
    "street":     ["circuit_is_street", "driver_recovery"],
    "track_type": ["constructor_track_type_avg", "driver_track_type_avg"],
    "mech_dnf":   ["constructor_mech_dnf_rate"],
}


def eval_cb(Xtr, ytr, Xva, yva, Xte, te_df, w=None, seed=42):
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
    p = cal.predict(m.predict_proba(Xte)[:, 1])
    te_df["p"] = p
    yte = te_df["podium"].values
    return {
        "prec3": _precision_at3(te_df, "p"),
        "auc": roc_auc_score(yte, p),
        "ll": log_loss(yte, np.clip(p, 1e-6, 1-1e-6)),
    }


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

    def run(cols, w_use, tag):
        r = eval_cb(tr[cols].values, tr["podium"].values,
                    va[cols].values, va["podium"].values,
                    te[cols].values, te.copy(), w=w_use)
        print(f"  {tag:35s} prec3 {r['prec3']:.3f}  AUC {r['auc']:.3f}  ll {r['ll']:.4f}")
        return r

    print("\n=== Full vs decay ablation ===")
    run(FEATURE_COLS, None, "FULL  no_decay")
    run(FEATURE_COLS, w, "FULL  decay_tau=1.5")

    print("\n=== Remove one group (with decay) ===")
    for gname, gcols in GROUPS.items():
        cols = [c for c in FEATURE_COLS if c not in gcols]
        run(cols, w, f"FULL - {gname:12s}")

    print("\n=== Tau sweep (single seed) ===")
    for tau in [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 10.0]:
        wt = np.exp(-(train_max - tr["season"].values) / tau)
        run(FEATURE_COLS, wt, f"tau={tau}")


if __name__ == "__main__":
    main()
