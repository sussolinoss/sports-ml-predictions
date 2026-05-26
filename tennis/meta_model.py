"""
Meta-model with Out-of-Fold (OOF) stacking — PRE-MATCH ONLY version.

ARCHITECTURE (all pre-match, no live state):

    Predictor A: pre-match XGBoost (on all features) ---\\
                                                          >--> Meta-model --> P(p1 wins)
    Predictor B: ELO probability (logistic on elo_diff) --/

ANTI-LEAKAGE GOLDEN RULE (Out-of-Fold):
  The meta-model MUST be trained on "honest" predictions from Predictor A:
  predictions made by a model that has NEVER seen that data during training.
  Solution: K-fold on the training data. For each fold k, train XGBoost on the
  OTHER folds and predict on fold k. Concatenating gives honest OOF predictions.

Note: this version does NOT use set-by-set snapshots or a live Markov model. It
exists to demonstrate the (possible) value of stacking pre-match predictors.
For the in-play version, see the git history / live_update.py.

USAGE:
    python -m meta_model

Generates meta_model_lr.pkl, meta_model_xgb.pkl, base_model_for_meta.json in data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BURNIN_END_YEAR,
    EARLY_STOPPING_ROUNDS,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_END_YEAR,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    XGB_PARAMS,
)
from feature_engineering import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Predictor B: ELO probability (closed-form)
# ---------------------------------------------------------------------------
def elo_diff_to_proba(elo_diff: float) -> float:
    """ELO -> P(p1 wins) using the standard logistic formula."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


# ---------------------------------------------------------------------------
# OOF predictions on the training set
# ---------------------------------------------------------------------------
def generate_oof_predictions(train_df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """
    K-fold OOF: each row gets a prediction made by a model trained on the
    other rows. "Honest" predictions for the meta-model.
    """
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["p1_wins"].values
    oof = np.full(len(train_df), np.nan)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    params = XGB_PARAMS.copy()
    n_estimators = params.pop("n_estimators")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_splits}: train={len(tr_idx):,}  val={len(va_idx):,}")
        dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx], feature_names=FEATURE_COLUMNS)
        dva = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=FEATURE_COLUMNS)
        model = xgb.train(
            params, dtr,
            num_boost_round=n_estimators,
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        oof[va_idx] = model.predict(dva)

    assert not np.isnan(oof).any(), "OOF incomplete - bug in K-fold"
    return oof


def train_final_base_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.Booster:
    """Final base model on the ENTIRE training set, to predict val and test."""
    params = XGB_PARAMS.copy()
    n_estimators = params.pop("n_estimators")
    dtr = xgb.DMatrix(train_df[FEATURE_COLUMNS], label=train_df["p1_wins"],
                      feature_names=FEATURE_COLUMNS)
    dva = xgb.DMatrix(val_df[FEATURE_COLUMNS], label=val_df["p1_wins"],
                      feature_names=FEATURE_COLUMNS)
    return xgb.train(
        params, dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=200,
    )


META_FEATURES = [
    "pre_pred",    # output of Predictor A (OOF in training, final model in val/test)
    "elo_proba",   # output of Predictor B (logistic on elo_diff)
]


def _add_meta_features(df: pd.DataFrame, pre_pred: np.ndarray) -> pd.DataFrame:
    """Add the meta columns (pre_pred, elo_proba) to the DataFrame."""
    out = df.copy()
    out["pre_pred"] = pre_pred
    out["elo_proba"] = elo_diff_to_proba(out["elo_diff"].values)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        raise RuntimeError("Esegui prima `python -m feature_engineering`")

    df = pd.read_parquet(features_path)
    print(f"Caricato features.parquet: {len(df):,} match")

    # Temporal split (same as train_model.py)
    y = df["year"]
    train_df = df[(y > BURNIN_END_YEAR) & (y <= TRAIN_END_YEAR)].reset_index(drop=True)
    val_df = df[(y > TRAIN_END_YEAR) & (y <= VAL_END_YEAR)].reset_index(drop=True)
    test_df = df[(y > VAL_END_YEAR) & (y <= TEST_END_YEAR)].reset_index(drop=True)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # STEP 1: OOF predictions on the training set (K-fold)
    print("\n[1/3] Genero predizioni OOF sul training (5-fold)...")
    oof = generate_oof_predictions(train_df, n_splits=5)
    oof_acc = ((oof > 0.5) == train_df["p1_wins"].values).mean()
    print(f"  Accuracy OOF (proxy della performance reale): {oof_acc:.4f}")

    # STEP 2: Final base model on the entire training set
    print("\n[2/3] Alleno modello base finale su tutto il training...")
    base_model = train_final_base_model(train_df, val_df)
    print(f"  Best iter: {base_model.best_iteration}")

    val_pre = base_model.predict(
        xgb.DMatrix(val_df[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    )
    test_pre = base_model.predict(
        xgb.DMatrix(test_df[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    )

    train_meta = _add_meta_features(train_df, oof)
    val_meta = _add_meta_features(val_df, val_pre)
    test_meta = _add_meta_features(test_df, test_pre)

    # STEP 3: Train meta-models
    print("\n[3/3] Alleno meta-modelli (stacking pre-match)...")
    X_meta_tr = train_meta[META_FEATURES].values
    y_meta_tr = train_meta["p1_wins"].values

    meta_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    meta_lr.fit(X_meta_tr, y_meta_tr)
    print("  Logistic Regression coef:")
    for name, coef in zip(META_FEATURES, meta_lr.coef_[0]):
        print(f"    {name:14s} = {coef:+.4f}")
    print(f"    intercept      = {meta_lr.intercept_[0]:+.4f}")

    meta_xgb = xgb.XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.9,
        random_state=RANDOM_SEED, eval_metric="logloss",
        early_stopping_rounds=30,
    )
    meta_xgb.fit(
        X_meta_tr, y_meta_tr,
        eval_set=[(val_meta[META_FEATURES].values, val_meta["p1_wins"].values)],
        verbose=False,
    )
    print(f"  XGBoost best_iter: {meta_xgb.best_iteration}")

    # Evaluation
    print("\n" + "=" * 70)
    print("RISULTATI (pre-match)")
    print("=" * 70)
    base_test_acc = None
    for name, snap in [("VAL", val_meta), ("TEST", test_meta)]:
        if len(snap) == 0:
            continue
        y_true = snap["p1_wins"].values
        X_meta = snap[META_FEATURES].values
        pre_only = snap["pre_pred"].values
        elo_only = snap["elo_proba"].values
        meta_lr_pred = meta_lr.predict_proba(X_meta)[:, 1]
        meta_xgb_pred = meta_xgb.predict_proba(X_meta)[:, 1]

        print(f"\n{name} — {len(snap):,} match "
              f"({snap['tourney_date'].dt.year.min()}-{snap['tourney_date'].dt.year.max()})")
        print(f"  {'Modello':22s}  {'acc':>7s}  {'logloss':>8s}  {'brier':>7s}")
        for label, pred in [
            ("XGBoost base (A)", pre_only),
            ("ELO logistica (B)", elo_only),
            ("Meta — LogReg", meta_lr_pred),
            ("Meta — XGBoost", meta_xgb_pred),
        ]:
            pred_c = np.clip(pred, 1e-6, 1 - 1e-6)
            acc = ((pred > 0.5) == y_true).mean()
            ll = log_loss(y_true, pred_c)
            bs = brier_score_loss(y_true, pred)
            print(f"  {label:22s}  {acc:.4f}  {ll:.4f}   {bs:.4f}")
            if name == "TEST" and label == "XGBoost base (A)":
                base_test_acc = acc
            if name == "TEST" and label == "Meta — XGBoost":
                meta_test_acc = acc

    # Honest verdict: is stacking worthwhile?
    if base_test_acc is not None and len(test_meta) > 0:
        delta = (meta_test_acc - base_test_acc) * 100
        print("\n" + "-" * 70)
        if delta >= 1.0:
            print(f"VERDETTO: il meta-modello batte il base di {delta:+.2f} pt sul test. "
                  "Lo stacking aggiunge valore.")
        else:
            print(f"VERDETTO: il meta-modello fa {delta:+.2f} pt vs base sul test (< 1 pt). "
                  "I due predittori sono correlati: lo stacking NON aggiunge valore reale, "
                  "tieni il solo XGBoost base.")
        print("-" * 70)

    # Save
    joblib.dump(meta_lr, PROCESSED_DIR / "meta_model_lr.pkl")
    joblib.dump(meta_xgb, PROCESSED_DIR / "meta_model_xgb.pkl")
    base_model.save_model(str(PROCESSED_DIR / "base_model_for_meta.json"))
    print(f"\nSalvati:")
    print(f"  {PROCESSED_DIR / 'meta_model_lr.pkl'}")
    print(f"  {PROCESSED_DIR / 'meta_model_xgb.pkl'}")
    print(f"  {PROCESSED_DIR / 'base_model_for_meta.json'}")


# ---------------------------------------------------------------------------
# API for pre-match inference
# ---------------------------------------------------------------------------
def predict_with_meta(pre_pred: float, elo_proba: float, use_xgb: bool = True) -> float:
    """
    Combine pre-match XGBoost (pre_pred) and ELO probability (elo_proba) with
    the trained meta-model, returning P(p1 wins).
    """
    suffix = "xgb" if use_xgb else "lr"
    meta = joblib.load(PROCESSED_DIR / f"meta_model_{suffix}.pkl")
    X = np.array([[pre_pred, elo_proba]])
    return float(meta.predict_proba(X)[0, 1])


if __name__ == "__main__":
    main()
