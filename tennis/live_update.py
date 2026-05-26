"""
Template for the LIVE (in-play) prediction update.

Strategy: you have two predictions that you combine.

1) PRE-MATCH prediction from the XGBoost model (P_model)
2) LIVE prediction based on the current match state (P_live)
   - For tennis: Markov chain over points (who serves, who leads in sets)
   - For F1: current position, tyre compounds, gaps, SC/VSC
   - For MotoGP: similar to F1

Combined output:
    P_final = alpha * P_live + (1 - alpha) * P_model
with alpha growing as the match progresses
(because the live state becomes more informative than the pre-match one).

For LIVE bookmaker odds:
    Use them as an ADDITIONAL feature, not as direct ground truth:
      bookmaker_proba_p1 = (1/odds_p1) / sum(1/odds_i)   # removes overround
    Then you train a second model that learns when the bookmaker is
    wrong (e.g. it does not see weather conditions).

Note: this file is a skeleton. You must connect your own live data source
(API or scraping) to the methods marked TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_PATH, PROCESSED_DIR
from feature_engineering import FEATURE_COLUMNS
from predict import predict_match


# 1) Markov chain for tennis (simple, single-server)
def prob_win_game_on_serve(p_point_on_serve: float) -> float:
    """
    Probability of winning the service game given p = P(winning a point on serve).
    Closed form for a 4-point game (standard tennis service game).
    """
    p, q = p_point_on_serve, 1 - p_point_on_serve
    # Probability of winning the game (including a possible deuce)
    win_no_deuce = (
        p**4
        + 4 * p**4 * q
        + 10 * p**4 * q**2
    )
    # Deuce: P(reaching deuce) = C(6,3) * p^3 * q^3
    p_deuce = 20 * p**3 * q**3
    win_from_deuce = p**2 / (p**2 + q**2)
    return win_no_deuce + p_deuce * win_from_deuce


def prob_win_set(p_serve_a: float, p_serve_b: float) -> float:
    """
    P(A wins a set) given p on serve for A and B. Simplified model:
    we simulate using the probability of winning the service game.
    From p_serve_a we derive P(A wins the game on his own serve).
    """
    pga = prob_win_game_on_serve(p_serve_a)
    pgb = 1 - prob_win_game_on_serve(p_serve_b)  # A wins the game on return
    # Approximation: P(A wins set) as a logistic function of (pga - (1-pgb)).
    # A rigorous model would use a Markov chain over 6+ score states.
    edge = (pga + pgb) / 2 - 0.5
    return 1 / (1 + np.exp(-edge * 12))  # 12 is an empirical scaling factor


# 2) Live state of a match
@dataclass
class LiveTennisState:
    """Current state of an in-progress tennis match."""
    p1_name: str
    p2_name: str
    surface: str
    best_of: int = 3
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0   # in the current set
    games_p2: int = 0
    points_p1: int = 0  # in the current game
    points_p2: int = 0
    serving: int = 1    # 1 = p1, 2 = p2
    # Aggregate live statistics
    p1_first_serve_in: float = 0.65   # typical ATP default
    p1_first_serve_won: float = 0.72
    p1_second_serve_won: float = 0.52
    p2_first_serve_in: float = 0.65
    p2_first_serve_won: float = 0.72
    p2_second_serve_won: float = 0.52

    def p1_point_win_on_own_serve(self) -> float:
        return (
            self.p1_first_serve_in * self.p1_first_serve_won
            + (1 - self.p1_first_serve_in) * self.p1_second_serve_won
        )

    def p2_point_win_on_own_serve(self) -> float:
        return (
            self.p2_first_serve_in * self.p2_first_serve_won
            + (1 - self.p2_first_serve_in) * self.p2_second_serve_won
        )


# 3) Combined predictor
class LivePredictor:
    """Combines pre-match prediction (XGBoost) + live state (Markov + bookmaker)."""

    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))

    def predict_pre_match(self, p1: str, p2: str, surface: str, best_of: int = 3) -> float:
        result = predict_match(p1, p2, surface, best_of)
        return result["proba_p1_wins"]

    def predict_live_markov(self, state: LiveTennisState) -> float:
        """Probability that p1 wins given the current state (Markov chain)."""
        # Probability of winning the next set
        p_set_p1 = prob_win_set(
            state.p1_point_win_on_own_serve(),
            state.p2_point_win_on_own_serve(),
        )
        # Sets needed to win the match
        sets_to_win = (state.best_of // 2) + 1
        # How many sets does p1 still need to win?
        remaining_p1 = sets_to_win - state.sets_p1
        remaining_p2 = sets_to_win - state.sets_p2
        if remaining_p1 <= 0:
            return 1.0
        if remaining_p2 <= 0:
            return 0.0
        # Approximate negative binomial distribution
        # P(p1 wins) = sum over the ways p1 collects the remaining sets
        from math import comb
        total = 0.0
        max_sets = remaining_p1 + remaining_p2 - 1
        for k in range(max_sets + 1):
            if k >= remaining_p1:
                total += comb(k - 1, remaining_p1 - 1) * (
                    p_set_p1 ** remaining_p1 * (1 - p_set_p1) ** (k - remaining_p1)
                )
        return total

    @staticmethod
    def odds_to_proba(odds_p1: float, odds_p2: float) -> float:
        """Decimal odds -> normalized implied probability (overround removed)."""
        implicit_p1 = 1 / odds_p1
        implicit_p2 = 1 / odds_p2
        norm = implicit_p1 + implicit_p2
        return implicit_p1 / norm

    def combine(
        self,
        p_model: float,
        p_live: float | None = None,
        p_book: float | None = None,
        alpha_live: float = 0.3,
        alpha_book: float = 0.4,
    ) -> float:
        """
        Combine the three sources. The weights must sum to 1; we rebalance
        based on what is available.
        """
        components = [("model", p_model, 1.0 - alpha_live - alpha_book)]
        if p_live is not None:
            components.append(("live", p_live, alpha_live))
        if p_book is not None:
            components.append(("book", p_book, alpha_book))
        # Renormalize the weights (in case some sources are missing)
        total_w = sum(w for _, _, w in components)
        return sum(p * w for _, p, w in components) / total_w

    def predict_full(
        self,
        p1: str,
        p2: str,
        surface: str,
        best_of: int = 3,
        live_state: LiveTennisState | None = None,
        book_odds: tuple[float, float] | None = None,
    ) -> dict:
        p_model = self.predict_pre_match(p1, p2, surface, best_of)
        p_live = self.predict_live_markov(live_state) if live_state else None
        p_book = self.odds_to_proba(*book_odds) if book_odds else None
        p_final = self.combine(p_model, p_live, p_book)
        return {
            "p_model": p_model,
            "p_live": p_live,
            "p_book": p_book,
            "p_final": p_final,
            "predicted_winner": p1 if p_final > 0.5 else p2,
        }


# 4) TODO: connect your own live data source
def scrape_live_state(match_url: str) -> LiveTennisState:
    """
    TODO: implement the scraping/API call to obtain the current state.

    Options:
      - The Odds API (https://the-odds-api.com)  for live odds
      - Betfair Exchange API for market data
      - Sofascore/Flashscore scraping for live scores (legally gray)
      - Official ATP API (paid, realtime data)
    """
    raise NotImplementedError("Implement the scraping for your preferred source.")


if __name__ == "__main__":
    # Usage example
    predictor = LivePredictor()

    # Pre-match
    pre = predictor.predict_pre_match("Jannik Sinner", "Carlos Alcaraz", "Hard", 5)
    print(f"Pre-match P(Sinner vince) = {pre:.4f}")

    # In-play: Sinner won the first set 6-4 and it is 3-2 in the second
    state = LiveTennisState(
        p1_name="Jannik Sinner", p2_name="Carlos Alcaraz",
        surface="Hard", best_of=5,
        sets_p1=1, sets_p2=0,
        games_p1=3, games_p2=2,
    )
    live = predictor.predict_live_markov(state)
    print(f"Live Markov P(Sinner vince) = {live:.4f}")

    # Combined (with simulated bookmaker odds)
    result = predictor.predict_full(
        "Jannik Sinner", "Carlos Alcaraz", "Hard", 5,
        live_state=state,
        book_odds=(1.65, 2.35),
    )
    print(f"\nFinale: {result}")
