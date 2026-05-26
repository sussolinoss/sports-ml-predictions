"""
Template per l'aggiornamento LIVE (in-play) della predizione.

Strategia: hai due predizioni che combini.

1) Predizione PRE-MATCH del modello XGBoost (P_model)
2) Predizione LIVE basata sullo stato corrente del match (P_live)
   - Per tennis: catena di Markov sui punti (chi serve, chi conduce nei set)
   - Per F1: posizione attuale, mescola gomme, distacchi, SC/VSC
   - Per MotoGP: simile a F1

Output combinato:
    P_final = alpha * P_live + (1 - alpha) * P_model
con alpha che cresce man mano che il match avanza
(perché lo stato live diventa più informativo del pre-match).

Per le QUOTE LIVE dei bookmaker (suggerimento dello screenshot Gemini):
    Usale come ULTERIORE feature, non come fonte di verità diretta:
      bookmaker_proba_p1 = (1/quota_p1) / sum(1/quota_i)   # rimuove overround
    Poi addestri un secondo modello che impara quando il bookmaker
    sbaglia (es. non vede le condizioni meteo).

NB: questo file è uno scheletro. Devi collegare la tua fonte di dati live
(API o scraping) ai metodi marcati TODO.
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


# ---------------------------------------------------------------------------
# 1) Catena di Markov per tennis (semplice, single-server)
# ---------------------------------------------------------------------------
def prob_win_game_on_serve(p_point_on_serve: float) -> float:
    """
    Probabilità di vincere il game al servizio dato p = P(vincere punto al servizio).
    Formula chiusa per game a 4 punti (servizio standard tennis).
    """
    p, q = p_point_on_serve, 1 - p_point_on_serve
    # Probabilità di vincere il game (incluso eventuale deuce)
    win_no_deuce = (
        p**4
        + 4 * p**4 * q
        + 10 * p**4 * q**2
    )
    # Deuce: P(arrivare a deuce) = C(6,3) * p^3 * q^3
    p_deuce = 20 * p**3 * q**3
    win_from_deuce = p**2 / (p**2 + q**2)
    return win_no_deuce + p_deuce * win_from_deuce


def prob_win_set(p_serve_a: float, p_serve_b: float) -> float:
    """
    P(A vince un set) data p al servizio di A e di B. Modello semplificato:
    simuliamo con probabilità di vincere il game al servizio.
    Da p_serve_a deriviamo P(A vince game al suo servizio).
    """
    pga = prob_win_game_on_serve(p_serve_a)
    pgb = 1 - prob_win_game_on_serve(p_serve_b)  # A vince game alla risposta
    # Approssimazione: P(A vince set) come funzione logistica di (pga - (1-pgb))
    # Per un modello rigoroso useresti catena di Markov a 6+ stati di score.
    edge = (pga + pgb) / 2 - 0.5
    return 1 / (1 + np.exp(-edge * 12))  # 12 è un fattore di scala empirico


# ---------------------------------------------------------------------------
# 2) Stato live di un match
# ---------------------------------------------------------------------------
@dataclass
class LiveTennisState:
    """Stato corrente di una partita di tennis in corso."""
    p1_name: str
    p2_name: str
    surface: str
    best_of: int = 3
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0   # nel set corrente
    games_p2: int = 0
    points_p1: int = 0  # nel game corrente
    points_p2: int = 0
    serving: int = 1    # 1 = p1, 2 = p2
    # Statistiche live aggregate
    p1_first_serve_in: float = 0.65   # default tipico ATP
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


# ---------------------------------------------------------------------------
# 3) Predittore combinato
# ---------------------------------------------------------------------------
class LivePredictor:
    """Combina predizione pre-match (XGBoost) + stato live (Markov + bookmaker)."""

    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))

    def predict_pre_match(self, p1: str, p2: str, surface: str, best_of: int = 3) -> float:
        result = predict_match(p1, p2, surface, best_of)
        return result["proba_p1_wins"]

    def predict_live_markov(self, state: LiveTennisState) -> float:
        """Probabilità che p1 vinca dato lo stato attuale (Markov chain)."""
        # Probabilità di vincere il prossimo set
        p_set_p1 = prob_win_set(
            state.p1_point_win_on_own_serve(),
            state.p2_point_win_on_own_serve(),
        )
        # Sets necessari per vincere il match
        sets_to_win = (state.best_of // 2) + 1
        # Quanti set deve ancora vincere p1?
        remaining_p1 = sets_to_win - state.sets_p1
        remaining_p2 = sets_to_win - state.sets_p2
        if remaining_p1 <= 0:
            return 1.0
        if remaining_p2 <= 0:
            return 0.0
        # Distribuzione binomiale negativa approssimata
        # P(p1 vince) = somma sui modi in cui p1 raccoglie i set rimanenti
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
        """Quote decimali -> probabilità implicita normalizzata (overround rimosso)."""
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
        Combina le tre fonti. I pesi devono sommare a 1; ribilanciamo
        in base a ciò che è disponibile.
        """
        components = [("model", p_model, 1.0 - alpha_live - alpha_book)]
        if p_live is not None:
            components.append(("live", p_live, alpha_live))
        if p_book is not None:
            components.append(("book", p_book, alpha_book))
        # Rinormalizza i pesi (caso in cui alcune fonti mancano)
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


# ---------------------------------------------------------------------------
# 4) TODO: collega la tua fonte di dati live
# ---------------------------------------------------------------------------
def scrape_live_state(match_url: str) -> LiveTennisState:
    """
    TODO: implementa lo scraping/API call per ottenere lo stato corrente.

    Opzioni:
      - The Odds API (https://the-odds-api.com)  per quote live
      - Betfair Exchange API per dati di mercato
      - Sofascore/Flashscore scraping per punteggi live (legalmente grigio)
      - API ufficiale ATP (a pagamento, dati realtime)
    """
    raise NotImplementedError("Implementa lo scraping della tua fonte preferita.")


if __name__ == "__main__":
    # Esempio di utilizzo
    predictor = LivePredictor()

    # Pre-match
    pre = predictor.predict_pre_match("Jannik Sinner", "Carlos Alcaraz", "Hard", 5)
    print(f"Pre-match P(Sinner vince) = {pre:.4f}")

    # In-play: Sinner ha vinto il primo set 6-4 e siamo 3-2 nel secondo
    state = LiveTennisState(
        p1_name="Jannik Sinner", p2_name="Carlos Alcaraz",
        surface="Hard", best_of=5,
        sets_p1=1, sets_p2=0,
        games_p1=3, games_p2=2,
    )
    live = predictor.predict_live_markov(state)
    print(f"Live Markov P(Sinner vince) = {live:.4f}")

    # Combinato (con quote bookmaker simulate)
    result = predictor.predict_full(
        "Jannik Sinner", "Carlos Alcaraz", "Hard", 5,
        live_state=state,
        book_odds=(1.65, 2.35),
    )
    print(f"\nFinale: {result}")
