"""
Extra statistics for the paper: ablation study, bootstrap CI on precision@3,
NDCG@3. Production config (decay tau=1.5, seed 42, CPU). Run from f1/:
  PYTHONPATH=.. ../.venv/bin/python -m experiments.f1_paper_stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features

GROUPS = {
    "grid+quali": ["grid", "quali_gap_ms", "teammate_gap_ms"],
    "lag": ["last1_pos", "last2_pos", "last3_pos", "last1_pod", "last2_pod",
            "last3_pod", "last1_dnf", "last2_dnf", "last3_dnf"],
    "track-type": ["constructor_track_type_avg", "driver_track_type_avg"],
    "track-history": ["driver_track_avg", "constructor_track_avg"],
    "street+recovery": ["circuit_is_street", "driver_recovery"],
    "form": ["driver_form", "driver_form_pts", "driver_podium_rate",
             "driver_dnf_rate", "constructor_form", "constructor_podium_rate"],
}


def train(cols, tr, va, te, w):
    model = CatBoostClassifier(
        iterations=1500, depth=5, learning_rate=0.03, boosting_type="Ordered",
        bootstrap_type="Bernoulli", subsample=0.85, l2_leaf_reg=3.0,
        min_data_in_leaf=5, loss_function="Logloss", random_seed=42,
        allow_writing_files=False, verbose=False, early_stopping_rounds=80,
        task_type="CPU")
    model.fit(tr[cols].values, tr["podium"].values, sample_weight=w,
              eval_set=(va[cols].values, va["podium"].values))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(model.predict_proba(va[cols].values)[:, 1], va["podium"].values)
    out = te.copy()
    out["p"] = cal.predict(model.predict_proba(te[cols].values)[:, 1])
    return out


def ndcg_at3(df):
    total = 0.0; n = 0
    for _, g in df.groupby(["season", "round"]):
        top = g.nlargest(3, "p")
        gains = top["podium"].values
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(gains)))
        dcg = (gains * discounts).sum()
        ideal = (np.sort(g["podium"].values)[::-1][:3] * discounts).sum()
        if ideal > 0:
            total += dcg / ideal; n += 1
    return total / n if n else 0.0


def bootstrap_ci(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    races = [g for _, g in df.groupby(["season", "round"])]
    scores = []
    for _ in range(n_boot):
        sample = rng.choice(len(races), len(races), replace=True)
        hits = sum(races[i].nlargest(3, "p")["podium"].sum() for i in sample)
        scores.append(hits / (3 * len(sample)))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def main():
    df = load_results()
    feat = build_features(df).sort_values(["season", "round"]).reset_index(drop=True)
    tr = feat[feat.season <= 2023]
    va = feat[feat.season == 2024]
    te = feat[feat.season == 2025].copy()
    w = np.exp(-(2023 - tr["season"].values) / 1.5)

    full = train(FEATURE_COLS, tr, va, te, w)
    p3 = _precision_at3(full, "p")
    ndcg = ndcg_at3(full)
    mean, lo, hi = bootstrap_ci(full)
    print(f"FULL model: precision@3 {p3:.3f}  NDCG@3 {ndcg:.3f}")
    print(f"bootstrap precision@3: mean {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    full["neg_grid"] = -full["grid"]
    grid = _precision_at3(full, "neg_grid")
    print(f"grid baseline precision@3 {grid:.3f}")

    print("\nAblation (remove one group, retrain):")
    for name, gcols in GROUPS.items():
        cols = [c for c in FEATURE_COLS if c not in gcols]
        out = train(cols, tr, va, te, w)
        s = _precision_at3(out, "p")
        print(f"  - {name:18s} precision@3 {s:.3f}  delta {(s-p3)*100:+.1f}pt")


if __name__ == "__main__":
    main()
