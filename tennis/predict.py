"""
Single pre-match prediction.

Example:
    python -m src.predict --p1 "Jannik Sinner" --p2 "Carlos Alcaraz" \
        --surface Hard --best_of 5

Uses the state saved by feature_engineering (final_state.pkl) to get ELO,
recent form and H2H updated to the last match in the data.
For rank/points/age it uses the last observed value for each player.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_PATH, PROCESSED_DIR
from feature_engineering import (
    FEATURE_COLUMNS,
    SERVE_METRICS,
    _encode_hand,
    _encode_level,
    _encode_round,
)


def _last_known(features_df: pd.DataFrame, player_id: int) -> dict:
    """Last observed value of rank/age/height/hand for the player."""
    p1_rows = features_df[features_df["p1_id"] == player_id].assign(side=1)
    p2_rows = features_df[features_df["p2_id"] == player_id].assign(side=2)
    if len(p1_rows) == 0 and len(p2_rows) == 0:
        return {}
    rows = pd.concat([p1_rows, p2_rows], ignore_index=True)
    rows = rows.sort_values("tourney_date")
    last = rows.iloc[-1]
    side = last["side"]
    pref = "p1_" if side == 1 else "p2_"
    return {
        "rank": last[f"{pref}rank"],
        "rank_pts": last[f"{pref}rank_pts"],
        "age": last[f"{pref}age"],
        "ht": last[f"{pref}ht"],
        "hand": last[f"{pref}hand"],
    }


def _find_player_id(name: str, name_map: dict) -> int:
    if name in name_map:
        return name_map[name]
    # Fuzzy: case-insensitive search
    lower_map = {k.lower(): v for k, v in name_map.items()}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    # Partial match
    matches = [k for k in name_map if name.lower() in k.lower()]
    if len(matches) == 1:
        print(f"  (match parziale: '{name}' -> '{matches[0]}')")
        return name_map[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Più giocatori contengono '{name}': {matches[:5]}")
    raise ValueError(f"Giocatore '{name}' non trovato nei dati.")


def predict_match(
    p1_name: str,
    p2_name: str,
    surface: str,
    best_of: int = 3,
    round_label: str = "R32",
    level: str = "A",
) -> dict:
    # Load resources
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    name_map = joblib.load(PROCESSED_DIR / "name_to_id.pkl")
    state = joblib.load(PROCESSED_DIR / "final_state.pkl")
    features_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    p1_id = _find_player_id(p1_name, name_map)
    p2_id = _find_player_id(p2_name, name_map)

    # Features from state
    p1_elo = state.elo.get_general(p1_id)
    p2_elo = state.elo.get_general(p2_id)
    p1_selo = state.elo.get_surface(p1_id, surface)
    p2_selo = state.elo.get_surface(p2_id, surface)
    p1_form = state.recent_form(p1_id)
    p2_form = state.recent_form(p2_id)
    p1_h2h = state.h2h_winrate(p1_id, p2_id)
    h2h_n = state.h2h_count(p1_id, p2_id)
    p1_played = state.elo.get_matches(p1_id)
    p2_played = state.elo.get_matches(p2_id)
    p1_ss = state.serve_stats(p1_id)
    p2_ss = state.serve_stats(p2_id)

    p1_info = _last_known(features_df, p1_id)
    p2_info = _last_known(features_df, p2_id)

    # Estimate fatigue as 0 (we don't know what they'll play between now and the match)
    p1_fatigue = 0
    p2_fatigue = 0

    row = {
        "p1_elo": p1_elo, "p2_elo": p2_elo,
        "p1_surface_elo": p1_selo, "p2_surface_elo": p2_selo,
        "p1_form": p1_form, "p2_form": p2_form,
        "p1_h2h_winrate": p1_h2h, "h2h_n": h2h_n,
        "p1_fatigue": p1_fatigue, "p2_fatigue": p2_fatigue,
        "p1_matches_played": p1_played, "p2_matches_played": p2_played,
        "p1_rank": p1_info.get("rank"), "p2_rank": p2_info.get("rank"),
        "p1_rank_pts": p1_info.get("rank_pts"), "p2_rank_pts": p2_info.get("rank_pts"),
        "p1_age": p1_info.get("age"), "p2_age": p2_info.get("age"),
        "p1_ht": p1_info.get("ht"), "p2_ht": p2_info.get("ht"),
        "p1_hand": p1_info.get("hand", 0), "p2_hand": p2_info.get("hand", 0),
        "best_of": best_of,
        "round_enc": _encode_round(round_label),
        "level_enc": _encode_level(level),
        "surface_Hard": int(surface == "Hard"),
        "surface_Clay": int(surface == "Clay"),
        "surface_Grass": int(surface == "Grass"),
        "surface_Carpet": int(surface == "Carpet"),
    }
    # Differences
    row["elo_diff"] = row["p1_elo"] - row["p2_elo"]
    row["surface_elo_diff"] = row["p1_surface_elo"] - row["p2_surface_elo"]
    row["form_diff"] = row["p1_form"] - row["p2_form"]
    row["fatigue_diff"] = row["p1_fatigue"] - row["p2_fatigue"]
    row["rank_diff_log"] = (
        np.log1p(row["p2_rank"] or 2000) - np.log1p(row["p1_rank"] or 2000)
    )
    row["rank_pts_diff"] = (row["p1_rank_pts"] or 0) - (row["p2_rank_pts"] or 0)
    row["age_diff"] = (row["p1_age"] or 0) - (row["p2_age"] or 0)
    row["ht_diff"] = (row["p1_ht"] or 0) - (row["p2_ht"] or 0)
    row["hand_diff"] = row["p1_hand"] - row["p2_hand"]

    # Rolling serve/return stats
    for m in SERVE_METRICS:
        row[f"p1_{m}"] = p1_ss[m]
        row[f"p2_{m}"] = p2_ss[m]
        row[f"{m}_diff"] = p1_ss[m] - p2_ss[m]

    # Closing odds unknown pre-match for a future game -> NaN (handled by XGBoost)
    row["book_proba_p1"] = np.nan

    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    dmat = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    proba_p1 = float(model.predict(dmat)[0])

    # Isotonic calibration if available (reliable probabilities for the edge)
    cal_path = PROCESSED_DIR / "calibrator.pkl"
    if cal_path.exists():
        proba_p1 = float(joblib.load(cal_path).predict([proba_p1])[0])

    return {
        "p1_name": p1_name, "p2_name": p2_name,
        "surface": surface, "best_of": best_of,
        "p1_elo": p1_elo, "p2_elo": p2_elo,
        "p1_surface_elo": p1_selo, "p2_surface_elo": p2_selo,
        "p1_form": p1_form, "p2_form": p2_form,
        "p1_h2h_winrate": p1_h2h, "h2h_n": h2h_n,
        "proba_p1_wins": proba_p1,
        "proba_p2_wins": 1 - proba_p1,
        "predicted_winner": p1_name if proba_p1 > 0.5 else p2_name,
        "confidence": max(proba_p1, 1 - proba_p1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True)
    parser.add_argument("--p2", required=True)
    parser.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    parser.add_argument("--best_of", type=int, default=3, choices=[3, 5])
    parser.add_argument("--round", default="R32")
    parser.add_argument("--level", default="A")
    args = parser.parse_args()

    result = predict_match(args.p1, args.p2, args.surface, args.best_of, args.round, args.level)

    print("\n" + "="*60)
    print(f"  {result['p1_name']}  vs  {result['p2_name']}")
    print(f"  Superficie: {result['surface']}   Best-of: {result['best_of']}")
    print("="*60)
    print(f"  ELO generale:        {result['p1_elo']:.1f}   vs   {result['p2_elo']:.1f}")
    print(f"  ELO {result['surface']:6s}:         {result['p1_surface_elo']:.1f}   vs   {result['p2_surface_elo']:.1f}")
    print(f"  Forma ultimi 10:      {result['p1_form']:.2f}   vs   {result['p2_form']:.2f}")
    print(f"  H2H ({result['h2h_n']} match):     winrate p1 = {result['p1_h2h_winrate']:.2f}")
    print("-"*60)
    print(f"  PROBABILITÀ P1 ({result['p1_name']}): {result['proba_p1_wins']:.4f}")
    print(f"  PROBABILITÀ P2 ({result['p2_name']}): {result['proba_p2_wins']:.4f}")
    print(f"  PREDIZIONE: {result['predicted_winner']}  (confidenza {result['confidence']:.3f})")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
