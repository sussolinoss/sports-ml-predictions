"""
Dynamic ELO system for tennis.

Maintains both at once:
  - general ELO (across all matches)
  - per-surface ELO (Hard, Clay, Grass, Carpet)

The K-factor decays with the number of matches played: young players move a
lot in rating, veterans move less.

TYPICAL USE (in feature_engineering.py):

    elo = EloSystem()
    for match in matches_in_chronological_order:
        # 1) FIRST read the current rating (this is the feature)
        rating_p1 = elo.get(match.winner_id, surface=match.surface)
        rating_p2 = elo.get(match.loser_id, surface=match.surface)
        save_features(rating_p1, rating_p2, ...)
        # 2) THEN update with the real outcome
        elo.update(match.winner_id, match.loser_id, surface=match.surface)

The order is critical: reversing it causes data leakage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ELO_INITIAL,
    ELO_K_BASE,
    ELO_K_DECAY_SCALE,
    ELO_K_DECAY_EXP,
    ELO_K_SURFACE,
)


@dataclass
class EloSystem:
    """Dynamic general + per-surface ELO."""

    initial: float = ELO_INITIAL
    k_base: float = ELO_K_BASE
    k_surface: float = ELO_K_SURFACE
    decay_scale: float = ELO_K_DECAY_SCALE
    decay_exp: float = ELO_K_DECAY_EXP

    # rating[player_id] = general elo
    rating: dict[int, float] = field(default_factory=dict)
    # surface_rating[(player_id, surface)] = per-surface elo
    surface_rating: dict[tuple[int, str], float] = field(default_factory=dict)
    # matches played per player (for K decay)
    matches_played: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    matches_played_surface: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Read (pre-match) — does not modify state
    def get_general(self, player_id: int) -> float:
        return self.rating.get(player_id, self.initial)

    def get_surface(self, player_id: int, surface: str) -> float:
        # If the player has never played on that surface, start from the general
        # ELO (less biased than the standard 1500).
        key = (player_id, surface)
        if key in self.surface_rating:
            return self.surface_rating[key]
        return self.get_general(player_id)

    def get_matches(self, player_id: int) -> int:
        return self.matches_played[player_id]

    # Dynamic K-factor
    def _k_general(self, player_id: int) -> float:
        n = self.matches_played[player_id]
        return self.k_base / (1 + n / self.decay_scale) ** self.decay_exp

    def _k_surface(self, player_id: int, surface: str) -> float:
        n = self.matches_played_surface[(player_id, surface)]
        return self.k_surface / (1 + n / self.decay_scale) ** self.decay_exp

    # Update (post-match)
    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Expected win probability of A under Elo."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: int, loser_id: int, surface: str) -> None:
        """Update general and per-surface ELO after the real outcome."""
        # general
        r_w = self.get_general(winner_id)
        r_l = self.get_general(loser_id)
        exp_w = self._expected(r_w, r_l)
        k_w = self._k_general(winner_id)
        k_l = self._k_general(loser_id)
        self.rating[winner_id] = r_w + k_w * (1.0 - exp_w)
        self.rating[loser_id] = r_l + k_l * (0.0 - (1.0 - exp_w))
        self.matches_played[winner_id] += 1
        self.matches_played[loser_id] += 1

        # surface
        if surface and surface != "Unknown":
            sr_w = self.get_surface(winner_id, surface)
            sr_l = self.get_surface(loser_id, surface)
            exp_sw = self._expected(sr_w, sr_l)
            sk_w = self._k_surface(winner_id, surface)
            sk_l = self._k_surface(loser_id, surface)
            self.surface_rating[(winner_id, surface)] = sr_w + sk_w * (1.0 - exp_sw)
            self.surface_rating[(loser_id, surface)] = sr_l + sk_l * (0.0 - (1.0 - exp_sw))
            self.matches_played_surface[(winner_id, surface)] += 1
            self.matches_played_surface[(loser_id, surface)] += 1

    # ELO probability (useful as a baseline)
    def predict_proba(
        self, p1_id: int, p2_id: int, surface: str, blend: float = 0.5
    ) -> float:
        """
        Return P(p1 wins) using a mix of general and per-surface ELO.

        blend=0 -> use only general ELO, blend=1 -> only per-surface.
        """
        gen = self._expected(self.get_general(p1_id), self.get_general(p2_id))
        sur = self._expected(
            self.get_surface(p1_id, surface), self.get_surface(p2_id, surface)
        )
        return (1 - blend) * gen + blend * sur
