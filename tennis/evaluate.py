"""
Weekly/monthly forward-testing.

Loads the trained model, takes a date range, and for each match:
  1) Generates the pre-match prediction
  2) Compares it with the actual result
  3) Saves everything to CSV for later analysis

Output: data/forward_test.csv with one row per match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_PATH, PROCESSED_DIR, TOUR_DIR
from feature_engineering import FEATURE_COLUMNS


def load_model_and_meta():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modello mancante: {MODEL_PATH}. Esegui train_model prima.")
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    meta_path = MODEL_PATH.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def forward_test(start_date: str, end_date: str) -> pd.DataFrame:
    """Predict all matches in the range, return a DataFrame with outcomes."""
    features_path = PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(features_path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (df["tourney_date"] >= start) & (df["tourney_date"] <= end)
    test_df = df[mask].copy().reset_index(drop=True)

    if len(test_df) == 0:
        raise RuntimeError(
            f"Nessun match tra {start_date} e {end_date}. "
            "Probabilmente i CSV Sackmann non coprono ancora quella finestra."
        )

    model, meta = load_model_and_meta()
    dmat = xgb.DMatrix(test_df[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    proba = model.predict(dmat)

    # Apply isotonic calibration if present (reliable probabilities for the edge)
    cal_path = PROCESSED_DIR / "calibrator.pkl"
    if cal_path.exists():
        proba = joblib.load(cal_path).predict(proba)

    out = test_df[[
        "tourney_date", "year", "surface", "best_of", "level_enc",
        "p1_id", "p2_id", "p1_elo", "p2_elo", "p1_wins",
        "p1_rank", "p2_rank", "p1_matches_played", "p2_matches_played",
    ]].copy()
    out["p1_proba"] = proba
    out["prediction"] = (proba > 0.5).astype(int)
    out["correct"] = (out["prediction"] == out["p1_wins"]).astype(int)
    # Probability of the model's choice (for confidence analysis)
    out["confidence"] = np.where(proba > 0.5, proba, 1 - proba)
    out["week"] = out["tourney_date"].dt.to_period("W-MON").astype(str)
    return out


def summarize(df: pd.DataFrame) -> None:
    """Print weekly + overall + per-surface + per-confidence accuracy."""
    print(f"\n{'='*60}")
    print(f"Forward test: {df['tourney_date'].min().date()} -> {df['tourney_date'].max().date()}")
    print(f"Match totali: {len(df)}")
    print(f"{'='*60}")

    # Overall
    acc = df["correct"].mean()
    ll = log_loss(df["p1_wins"], df["p1_proba"].clip(1e-6, 1 - 1e-6))
    bs = brier_score_loss(df["p1_wins"], df["p1_proba"])
    print(f"\nGLOBALE:  acc={acc:.4f}  logloss={ll:.4f}  brier={bs:.4f}")

    # By week
    print(f"\nACCURATEZZA SETTIMANALE:")
    weekly = df.groupby("week").agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        avg_conf=("confidence", "mean"),
    ).reset_index()
    for _, row in weekly.iterrows():
        print(f"  {row['week']:25s}  n={int(row['n']):4d}  acc={row['acc']:.4f}  "
              f"avg_conf={row['avg_conf']:.3f}")

    # By surface
    print(f"\nACCURATEZZA PER SUPERFICIE:")
    by_surf = df.groupby("surface").agg(n=("correct", "size"), acc=("correct", "mean"))
    for surf, row in by_surf.iterrows():
        print(f"  {surf:10s}  n={int(row['n']):4d}  acc={row['acc']:.4f}")

    # By confidence level (informal calibration)
    print(f"\nACCURATEZZA PER LIVELLO DI CONFIDENZA:")
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df["conf_bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)
    by_conf = df.groupby("conf_bin", observed=True).agg(
        n=("correct", "size"), acc=("correct", "mean")
    )
    for bin_label, row in by_conf.iterrows():
        print(f"  conf={str(bin_label):20s}  n={int(row['n']):4d}  acc={row['acc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Data inizio YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Data fine YYYY-MM-DD")
    parser.add_argument("--out", default=str(TOUR_DIR / "forward_test.csv"),
                        help="File CSV di output")
    args = parser.parse_args()

    df = forward_test(args.start, args.end)
    summarize(df)

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\nRisultati dettagliati salvati in: {out_path}")


if __name__ == "__main__":
    main()
