"""
Full end-to-end pipeline.

Runs in sequence:
  1) Download Sackmann data (cached, skipped if already downloaded)
  2) Anti-leakage feature engineering
  3) XGBoost training with a temporal split
  4) Forward test on the last available month

Usage:
    python run_full_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import download_all, load_matches
from feature_engineering import (
    BURNIN_END_YEAR,
    PROCESSED_DIR,
    build_book_index,
    build_features_with_state,
)
from train_model import main as train_main
from evaluate import forward_test, summarize
from meta_model import main as meta_main


def step_1_download():
    print("\n" + "="*70)
    print("STEP 1/5 — Download dati Sackmann")
    print("="*70)
    download_all()


def step_2_features():
    print("\n" + "="*70)
    print("STEP 2/5 — Feature engineering (anti-leakage)")
    print("="*70)
    df = load_matches()
    print(f"  {len(df):,} match caricati")
    odds_index = build_book_index()
    features, state = build_features_with_state(df, odds_index)
    features = features[features["year"] > BURNIN_END_YEAR].reset_index(drop=True)

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)

    import joblib
    joblib.dump(state, PROCESSED_DIR / "final_state.pkl")
    name_map = {}
    for _, row in df.iterrows():
        name_map[row["winner_name"]] = int(row["winner_id"])
        name_map[row["loser_name"]] = int(row["loser_id"])
    joblib.dump(name_map, PROCESSED_DIR / "name_to_id.pkl")

    print(f"  Salvato features.parquet ({len(features):,} righe)")
    print(f"  Bilanciamento target: {features['p1_wins'].mean():.3f}")


def step_3_train():
    print("\n" + "="*70)
    print("STEP 3/5 — Training XGBoost base")
    print("="*70)
    train_main()


def step_4_meta():
    print("\n" + "="*70)
    print("STEP 4/5 — Meta-modello con stacking OOF")
    print("="*70)
    meta_main()


def step_5_forward_test():
    print("\n" + "="*70)
    print("STEP 5/5 — Forward test ultimo mese disponibile")
    print("="*70)
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    last_date = df["tourney_date"].max()
    start = (last_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = last_date.strftime("%Y-%m-%d")
    print(f"  Range: {start} -> {end}")

    results = forward_test(start, end)
    summarize(results)
    out = PROCESSED_DIR.parent / "forward_test.csv"
    results.to_csv(out, index=False)
    print(f"\n  Risultati: {out}")


def main():
    step_1_download()
    step_2_features()
    step_3_train()
    step_4_meta()
    step_5_forward_test()
    print("\n" + "="*70)
    print("PIPELINE COMPLETATA")
    print("="*70)
    print("\nProssimi passi:")
    print("  - Forward test su range custom:")
    print("    python -m evaluate --start 2025-09-01 --end 2025-09-30")
    print("  - Backtest scommesse (ROI) su un mese coperto da tennis-data.co.uk:")
    print("    python -m backtest --start 2025-09-01 --end 2025-09-30")
    print("  - Predire una partita specifica:")
    print("    python -m predict --p1 'Jannik Sinner' --p2 'Carlos Alcaraz' --surface Hard")
    print("  - Meta-modello pre-match (Python):")
    print("    from meta_model import predict_with_meta")
    print("    predict_with_meta(pre_pred=0.65, elo_proba=0.60)")


if __name__ == "__main__":
    main()
