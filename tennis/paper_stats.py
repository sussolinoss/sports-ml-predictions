"""
Extra statistics for the tennis paper: bootstrap confidence interval on accuracy
and an ablation study (remove one feature group at a time, retrain, re-evaluate).
Uses the same features, split and XGBoost params as train_model.py.

Run from tennis/:  PYTHONPATH=. ../.venv/bin/python paper_stats.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

from config import (BURNIN_END_YEAR, EARLY_STOPPING_ROUNDS, TEST_END_YEAR,
                    TRAIN_END_YEAR, VAL_END_YEAR, XGB_PARAMS)
from feature_engineering import FEATURE_COLUMNS

FEATURES_PATH = "data/atp/processed/features.parquet"

# Feature groups for the ablation (which columns to drop together).
GROUPS = {
    "ELO": ["elo_diff", "surface_elo_diff", "p1_elo", "p2_elo",
            "p1_surface_elo", "p2_surface_elo"],
    "ranking": ["rank_diff_log", "rank_pts_diff", "p1_rank", "p2_rank",
                "p1_rank_pts", "p2_rank_pts"],
    "recent form": ["form_diff", "p1_form", "p2_form"],
    "head-to-head": ["p1_h2h_winrate", "h2h_n"],
    "serve stats": ["p1_serve_pts_won", "p1_ace_rate", "p1_df_rate", "p1_first_in",
                    "p1_first_won", "p1_bp_saved", "p1_return_pts_won",
                    "p2_serve_pts_won", "p2_ace_rate", "p2_df_rate", "p2_first_in",
                    "p2_first_won", "p2_bp_saved", "p2_return_pts_won",
                    "serve_pts_won_diff", "ace_rate_diff", "df_rate_diff",
                    "first_in_diff", "first_won_diff", "bp_saved_diff",
                    "return_pts_won_diff"],
    "bookmaker odds": ["book_proba_p1"],
}


def split(df):
    y = df["year"]
    tr = df[(y > BURNIN_END_YEAR) & (y <= TRAIN_END_YEAR)]
    va = df[(y > TRAIN_END_YEAR) & (y <= VAL_END_YEAR)]
    te = df[(y > VAL_END_YEAR) & (y <= TEST_END_YEAR)]
    return tr, va, te


def train_eval(cols, tr, va, te):
    params = XGB_PARAMS.copy()
    n_estimators = params.pop("n_estimators")
    dtr = xgb.DMatrix(tr[cols], label=tr["p1_wins"], feature_names=cols)
    dva = xgb.DMatrix(va[cols], label=va["p1_wins"], feature_names=cols)
    dte = xgb.DMatrix(te[cols], label=te["p1_wins"], feature_names=cols)
    model = xgb.train(params, dtr, num_boost_round=n_estimators,
                      evals=[(dva, "val")],
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(model.predict(dva), va["p1_wins"])
    p_te = cal.predict(model.predict(dte))
    acc = accuracy_score(te["p1_wins"], (p_te > 0.5).astype(int))
    return acc, p_te


def bootstrap_ci(y_true, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); pred = (np.asarray(p) > 0.5).astype(int)
    correct = (pred == y).astype(int)
    n = len(correct)
    scores = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    df = pd.read_parquet(FEATURES_PATH)
    tr, va, te = split(df)
    print(f"split: train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    acc, p_te = train_eval(FEATURE_COLUMNS, tr, va, te)
    mean, lo, hi = bootstrap_ci(te["p1_wins"].values, p_te)
    print(f"\nFULL model: accuracy {acc:.4f}")
    print(f"bootstrap accuracy: mean {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\nAblation (remove one group, retrain):")
    for name, gcols in GROUPS.items():
        cols = [c for c in FEATURE_COLUMNS if c not in gcols]
        a, _ = train_eval(cols, tr, va, te)
        print(f"  - {name:16s} accuracy {a:.4f}  delta {(a-acc)*100:+.2f}pt")


if __name__ == "__main__":
    main()
