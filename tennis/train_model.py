"""
Training XGBoost con split TEMPORALE (mai casuale).

Split:
  train: anni > BURNIN_END_YEAR e <= TRAIN_END_YEAR
  val:   TRAIN_END_YEAR < anno <= VAL_END_YEAR  (per early stopping)
  test:  VAL_END_YEAR < anno <= TEST_END_YEAR  (forward-test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BURNIN_END_YEAR,
    EARLY_STOPPING_ROUNDS,
    MODEL_PATH,
    PROCESSED_DIR,
    TEST_END_YEAR,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    XGB_PARAMS,
)
from feature_engineering import FEATURE_COLUMNS


def time_split(df: pd.DataFrame):
    """Return (train, val, test) DataFrames with a temporal split."""
    y = df["year"]
    train = df[(y > BURNIN_END_YEAR) & (y <= TRAIN_END_YEAR)]
    val = df[(y > TRAIN_END_YEAR) & (y <= VAL_END_YEAR)]
    test = df[(y > VAL_END_YEAR) & (y <= TEST_END_YEAR)]
    return train, val, test


def baseline_elo_accuracy(df: pd.DataFrame) -> dict:
    """
    Baseline: predict p1 wins if its ELO is greater than p2's.
    Used to measure how much XGBoost adds over pure ELO.
    """
    pred = (df["elo_diff"] > 0).astype(int)
    acc = accuracy_score(df["p1_wins"], pred)
    # Classic ELO probability
    proba = 1.0 / (1.0 + 10 ** (-df["elo_diff"] / 400.0))
    ll = log_loss(df["p1_wins"], proba.clip(1e-6, 1 - 1e-6))
    bs = brier_score_loss(df["p1_wins"], proba)
    return {"accuracy": acc, "log_loss": ll, "brier": bs}


def evaluate(model: xgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    proba = model.predict(dmat)
    pred = (proba > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba.clip(1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, proba),
        "n": len(y),
    }


def main():
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        raise RuntimeError(
            "features.parquet non trovato. Esegui prima `python -m src.feature_engineering`."
        )

    print(f"Carico {features_path}")
    df = pd.read_parquet(features_path)
    print(f"  {len(df):,} match totali")

    train_df, val_df, test_df = time_split(df)
    print(f"\nSplit temporale:")
    print(f"  train: {len(train_df):,}  ({BURNIN_END_YEAR+1}-{TRAIN_END_YEAR})")
    print(f"  val:   {len(val_df):,}  ({TRAIN_END_YEAR+1}-{VAL_END_YEAR})")
    print(f"  test:  {len(test_df):,}  ({VAL_END_YEAR+1}-{TEST_END_YEAR})")

    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "Val o test set vuoto. Controlla TRAIN_END_YEAR / VAL_END_YEAR in config.py."
        )

    # Pure ELO baseline for reference
    print("\n--- Baseline ELO puro ---")
    for name, split in [("val", val_df), ("test", test_df)]:
        m = baseline_elo_accuracy(split)
        print(f"  {name}: acc={m['accuracy']:.4f}  logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}")

    # XGBoost
    print("\n--- Training XGBoost ---")
    X_tr = train_df[FEATURE_COLUMNS]
    y_tr = train_df["p1_wins"]
    X_va = val_df[FEATURE_COLUMNS]
    y_va = val_df["p1_wins"]
    X_te = test_df[FEATURE_COLUMNS]
    y_te = test_df["p1_wins"]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURE_COLUMNS)

    params = XGB_PARAMS.copy()
    n_estimators = params.pop("n_estimators")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # Final evaluation
    print("\n--- Risultati XGBoost ---")
    val_metrics = evaluate(model, X_va, y_va)
    test_metrics = evaluate(model, X_te, y_te)
    print(f"  val:  acc={val_metrics['accuracy']:.4f}  logloss={val_metrics['log_loss']:.4f}  brier={val_metrics['brier']:.4f}")
    print(f"  test: acc={test_metrics['accuracy']:.4f}  logloss={test_metrics['log_loss']:.4f}  brier={test_metrics['brier']:.4f}")

    # Isotonic calibration (on the val set): raw XGBoost probabilities are often
    # over-confident; this makes the betting EDGE computed from them reliable.
    val_proba = model.predict(dval)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_proba, y_va.values)
    joblib.dump(calibrator, PROCESSED_DIR / "calibrator.pkl")
    cal_test = calibrator.predict(model.predict(xgb.DMatrix(X_te, feature_names=FEATURE_COLUMNS)))
    print("\n--- Calibrazione isotonica (val -> applicata) ---")
    print(f"  test brier:  raw={test_metrics['brier']:.4f} -> "
          f"calibrato={brier_score_loss(y_te, cal_test):.4f}")
    print(f"  test logloss raw={test_metrics['log_loss']:.4f} -> "
          f"calibrato={log_loss(y_te, cal_test.clip(1e-6, 1-1e-6)):.4f}")
    print(f"  (accuracy invariata; cambia solo l'affidabilita' delle probabilita')")

    print("\n--- Top 15 feature ---")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]
    for name, gain in sorted_imp:
        print(f"  {name:25s} gain={gain:.1f}")

    # Save model + metadata
    model.save_model(str(MODEL_PATH))
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "best_iteration": int(model.best_iteration),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": {
            "train_end_year": TRAIN_END_YEAR,
            "val_end_year": VAL_END_YEAR,
            "test_end_year": TEST_END_YEAR,
            "xgb_params": XGB_PARAMS,
        },
    }
    meta_path = MODEL_PATH.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nModello salvato in: {MODEL_PATH}")
    print(f"Metadati in:        {meta_path}")


if __name__ == "__main__":
    main()
