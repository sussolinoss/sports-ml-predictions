"""
Anti-leakage feature engineering.

GOLDEN RULE: for each match, first compute ALL features using the state of the
world *before* the match, then (and only then) update the state with the real
outcome. Reversing the order = leakage = fake accuracy.

Output: a DataFrame with `p1_*`, `p2_*` columns and the binary target `p1_wins`.
The choice of who is "p1" and who is "p2" is randomized deterministically
(seed from config) to keep the model from learning "p1=always winner".
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BURNIN_END_YEAR,
    PROCESSED_DIR,
    RANDOM_SEED,
    RECENT_FORM_WINDOW,
    SERVE_WINDOW,
)
import joblib

from data_loader import load_matches
from elo_system import EloSystem


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
ROUND_ORDER = {
    "RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6,
    "BR": 6, "F": 7,
}
LEVEL_ORDER = {"D": 0, "C": 1, "S": 2, "A": 3, "F": 4, "M": 5, "G": 6}
HAND_MAP = {"R": 1, "L": -1, "U": 0}


def _encode_round(r: str) -> int:
    return ROUND_ORDER.get(str(r).strip(), 0)


def _encode_level(lvl: str) -> int:
    return LEVEL_ORDER.get(str(lvl).strip(), 2)


def _encode_hand(h: object) -> int:
    return HAND_MAP.get(str(h).strip(), 0)


# Picklable factories for the MatchState defaultdicts (lambdas are NOT picklable,
# and the state is saved with joblib in final_state.pkl).
def _new_form_deque() -> deque:
    return deque(maxlen=RECENT_FORM_WINDOW)


def _new_h2h() -> list[int]:
    return [0, 0]


def _new_serve_deque() -> deque:
    return deque(maxlen=SERVE_WINDOW)


# Rolling serve/return statistics + realistic ATP priors (used until the player
# has enough history).
SERVE_METRICS = [
    "serve_pts_won", "ace_rate", "df_rate", "first_in", "first_won",
    "bp_saved", "return_pts_won",
]
SERVE_PRIORS = {
    "serve_pts_won": 0.63, "ace_rate": 0.07, "df_rate": 0.04, "first_in": 0.60,
    "first_won": 0.73, "bp_saved": 0.62, "return_pts_won": 0.37,
}


def _serve_record(me: dict, opp: dict) -> dict:
    """Raw per-match record for a player (includes return points from the opponent)."""
    return {
        "svpt": me["svpt"], "ace": me["ace"], "df": me["df"],
        "firstIn": me["firstIn"], "firstWon": me["firstWon"], "secondWon": me["secondWon"],
        "bpSaved": me["bpSaved"], "bpFaced": me["bpFaced"],
        "retpts": opp["svpt"],
        "retwon": opp["svpt"] - (opp["firstWon"] + opp["secondWon"]),
    }


def _extract_serve(row, p: str) -> dict | None:
    """Extract the raw serve stats for 'w' or 'l'. None if missing/invalid."""
    def g(col):
        return getattr(row, f"{p}_{col}", np.nan)

    svpt = g("svpt")
    first_in = g("1stIn")
    first_won = g("1stWon")
    second_won = g("2ndWon")
    if any(pd.isna(x) for x in (svpt, first_in, first_won, second_won)) or svpt <= 0:
        return None

    def num(x):
        return 0.0 if pd.isna(x) else float(x)

    return {
        "svpt": float(svpt), "ace": num(g("ace")), "df": num(g("df")),
        "firstIn": float(first_in), "firstWon": float(first_won),
        "secondWon": float(second_won),
        "bpSaved": num(g("bpSaved")), "bpFaced": num(g("bpFaced")),
    }


# Bookmaker closing odds AS A FEATURE (pre-match, no leakage).
# Odds are pre-match: a player's implied probability is known before the match.
# Assignment to p1/p2 follows the deterministic SWAP, not the outcome -> no leakage.
_BOOK_DATE_TOL_DAYS = 14


def build_book_index(years=None):
    """Download tennis-data.co.uk closing odds and build the pair->odds index.
    Returns None if odds are unavailable (book_proba_p1 will be NaN)."""
    try:
        from odds_loader import build_pair_index, download_odds_years, load_odds
        yrs = list(years) if years is not None else None
        if yrs is None:
            from config import YEARS as _YEARS
            yrs = list(_YEARS)
        download_odds_years(yrs)
        odds = load_odds(f"{min(yrs)}-01-01", f"{max(yrs)}-12-31", auto_download=False)
        print(f"  Closing odds caricate come feature: {len(odds):,} match con quota")
        return build_pair_index(odds)
    except Exception as e:  # noqa: BLE001 - network/openpyxl/missing file: degrade to NaN
        print(f"  ! Closing odds non disponibili come feature ({e}); book_proba_p1 = NaN")
        return None


def _book_winner_prob(index: dict, wkey: str, lkey: str, when) -> float | None:
    """Implied prob (overround removed) of player wkey, from the closest odds."""
    from odds_loader import implied_probs
    rows = index.get(frozenset((wkey, lkey)))
    if not rows:
        return None
    best, best_d = None, None
    for r in rows:
        d = abs((r["date"] - when).days)
        if d <= _BOOK_DATE_TOL_DAYS and (best_d is None or d < best_d):
            best, best_d = r, d
    if best is None:
        return None
    p_win, p_los = implied_probs(best["odd_winner"], best["odd_loser"])
    return p_win if best["winner_key"] == wkey else p_los


def _deterministic_swap(date: pd.Timestamp, match_num: object) -> bool:
    """
    Decide deterministically (and independently of the outcome) whether to place
    the winner as p1 (False) or as p2 (True).
    Uses hash of (date, match_num) modulo 2.
    """
    key = f"{date.strftime('%Y%m%d')}_{match_num}_{RANDOM_SEED}"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) % 2 == 1


# Dynamic state (all pre-match)
class MatchState:
    """All live statistics updated day by day."""

    def __init__(self):
        self.elo = EloSystem()
        # last N wins (1) / losses (0) per player
        self.recent_results: dict[int, deque] = defaultdict(_new_form_deque)
        # H2H: key = sorted tuple (id_min, id_max), value = (wins_min, wins_max)
        self.h2h: dict[tuple[int, int], list[int]] = defaultdict(_new_h2h)
        # minutes played in the last 14 days (list of (date, minutes) tuples)
        self.recent_load: dict[int, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
        # last N raw serve/return records per player
        self.serve_history: dict[int, deque] = defaultdict(_new_serve_deque)

    # read (PRE-match)
    def recent_form(self, pid: int) -> float:
        results = self.recent_results[pid]
        if not results:
            return 0.5  # neutral prior
        return sum(results) / len(results)

    def h2h_winrate(self, pid_a: int, pid_b: int) -> float:
        key = (min(pid_a, pid_b), max(pid_a, pid_b))
        wins = self.h2h[key]
        total = wins[0] + wins[1]
        if total == 0:
            return 0.5
        # Return the win rate of pid_a
        if pid_a == key[0]:
            return wins[0] / total
        return wins[1] / total

    def h2h_count(self, pid_a: int, pid_b: int) -> int:
        key = (min(pid_a, pid_b), max(pid_a, pid_b))
        return sum(self.h2h[key])

    def serve_stats(self, pid: int) -> dict:
        """Rolling serve/return rates (sum of counts over the window)."""
        hist = self.serve_history[pid]
        if not hist:
            return dict(SERVE_PRIORS)
        s = defaultdict(float)
        for rec in hist:
            for k, v in rec.items():
                s[k] += v

        def safe(num, den, prior):
            return num / den if den > 0 else prior

        return {
            "serve_pts_won": safe(s["firstWon"] + s["secondWon"], s["svpt"], SERVE_PRIORS["serve_pts_won"]),
            "ace_rate": safe(s["ace"], s["svpt"], SERVE_PRIORS["ace_rate"]),
            "df_rate": safe(s["df"], s["svpt"], SERVE_PRIORS["df_rate"]),
            "first_in": safe(s["firstIn"], s["svpt"], SERVE_PRIORS["first_in"]),
            "first_won": safe(s["firstWon"], s["firstIn"], SERVE_PRIORS["first_won"]),
            "bp_saved": safe(s["bpSaved"], s["bpFaced"], SERVE_PRIORS["bp_saved"]),
            "return_pts_won": safe(s["retwon"], s["retpts"], SERVE_PRIORS["return_pts_won"]),
        }

    def update_serve(self, w_id: int, l_id: int, w_raw: dict | None, l_raw: dict | None) -> None:
        """Append the raw serve records after the match (requires both sides)."""
        if w_raw is None or l_raw is None:
            return
        self.serve_history[w_id].append(_serve_record(w_raw, l_raw))
        self.serve_history[l_id].append(_serve_record(l_raw, w_raw))

    def fatigue(self, pid: int, current_date: pd.Timestamp, window_days: int = 14) -> float:
        """Minutes played in the last `window_days`."""
        cutoff = current_date - pd.Timedelta(days=window_days)
        # Drop old records to bound memory growth
        self.recent_load[pid] = [(d, m) for d, m in self.recent_load[pid] if d >= cutoff]
        return sum(m for _, m in self.recent_load[pid])

    # write (POST-match)
    def update(
        self,
        winner_id: int,
        loser_id: int,
        surface: str,
        date: pd.Timestamp,
        minutes: float | None,
    ) -> None:
        # ELO (delegates to its own update)
        self.elo.update(winner_id, loser_id, surface)
        # Recent form
        self.recent_results[winner_id].append(1)
        self.recent_results[loser_id].append(0)
        # H2H
        key = (min(winner_id, loser_id), max(winner_id, loser_id))
        if winner_id == key[0]:
            self.h2h[key][0] += 1
        else:
            self.h2h[key][1] += 1
        # Fatigue
        if minutes and not np.isnan(minutes):
            self.recent_load[winner_id].append((date, minutes))
            self.recent_load[loser_id].append((date, minutes))


# Main pipeline
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anti-leakage pre-match features (delegates to build_features_with_state)."""
    return build_features_with_state(df)[0]


FEATURE_COLUMNS = [
    "elo_diff", "surface_elo_diff",
    "p1_elo", "p2_elo", "p1_surface_elo", "p2_surface_elo",
    "form_diff", "p1_form", "p2_form",
    "p1_h2h_winrate", "h2h_n",
    "fatigue_diff", "p1_fatigue", "p2_fatigue",
    "p1_matches_played", "p2_matches_played",
    "rank_diff_log", "rank_pts_diff",
    "p1_rank", "p2_rank", "p1_rank_pts", "p2_rank_pts",
    "age_diff", "ht_diff", "hand_diff",
    "p1_age", "p2_age",
    "best_of", "round_enc", "level_enc",
    "surface_Hard", "surface_Clay", "surface_Grass", "surface_Carpet",
    "book_proba_p1",
] + [
    f"p1_{m}" for m in SERVE_METRICS
] + [
    f"p2_{m}" for m in SERVE_METRICS
] + [
    f"{m}_diff" for m in SERVE_METRICS
]


def main():
    print("Carico match...")
    df = load_matches()
    print(f"  {len(df):,} match")
    print("\nCostruisco feature pre-match (anti-leakage)...")
    odds_index = build_book_index()
    features, final_state = build_features_with_state(df, odds_index)

    # Remove burn-in
    features = features[features["year"] > BURNIN_END_YEAR].reset_index(drop=True)
    print(f"  {len(features):,} match dopo burn-in (>{BURNIN_END_YEAR})")

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)

    # Save final state (for predict.py)
    state_path = PROCESSED_DIR / "final_state.pkl"
    joblib.dump(final_state, state_path)

    # Save name -> id mapping (used by predict.py)
    name_map = {}
    for _, row in df.iterrows():
        name_map[row["winner_name"]] = int(row["winner_id"])
        name_map[row["loser_name"]] = int(row["loser_id"])
    joblib.dump(name_map, PROCESSED_DIR / "name_to_id.pkl")

    print(f"\nSalvato: {out_path}")
    print(f"         {state_path}")
    print(f"  Colonne: {len(features.columns)}, di cui {len(FEATURE_COLUMNS)} usate per ML")
    print(f"  Bilanciamento target: p1_wins={features['p1_wins'].mean():.3f} "
          f"(deve essere ~0.5 per via dello swap deterministico)")


def build_features_with_state(df: pd.DataFrame, odds_index: dict | None = None):
    """Like build_features but also returns the final state.

    If odds_index is provided, adds the `book_proba_p1` feature (p1's implied
    probability from the closing odds, NaN where no odds exist)."""
    assert df["tourney_date"].is_monotonic_increasing
    from odds_loader import sackmann_name_to_key
    state = MatchState()
    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Feature engineering"):
        w_id, l_id = row.winner_id, row.loser_id
        surface = row.surface
        date = row.tourney_date

        w_elo = state.elo.get_general(w_id)
        l_elo = state.elo.get_general(l_id)
        w_selo = state.elo.get_surface(w_id, surface)
        l_selo = state.elo.get_surface(l_id, surface)
        w_form = state.recent_form(w_id)
        l_form = state.recent_form(l_id)
        w_h2h = state.h2h_winrate(w_id, l_id)
        h2h_n = state.h2h_count(w_id, l_id)
        w_fatigue = state.fatigue(w_id, date)
        l_fatigue = state.fatigue(l_id, date)
        w_played = state.elo.get_matches(w_id)
        l_played = state.elo.get_matches(l_id)
        w_ss = state.serve_stats(w_id)
        l_ss = state.serve_stats(l_id)
        w_raw = _extract_serve(row, "w")
        l_raw = _extract_serve(row, "l")

        swap = _deterministic_swap(date, row.match_num)

        if not swap:
            p1_id, p2_id = w_id, l_id
            features = {
                "p1_elo": w_elo, "p2_elo": l_elo,
                "p1_surface_elo": w_selo, "p2_surface_elo": l_selo,
                "p1_form": w_form, "p2_form": l_form,
                "p1_h2h_winrate": w_h2h, "p2_h2h_winrate": 1 - w_h2h,
                "p1_fatigue": w_fatigue, "p2_fatigue": l_fatigue,
                "p1_matches_played": w_played, "p2_matches_played": l_played,
                "p1_rank": row.winner_rank, "p2_rank": row.loser_rank,
                "p1_rank_pts": row.winner_rank_points, "p2_rank_pts": row.loser_rank_points,
                "p1_age": row.winner_age, "p2_age": row.loser_age,
                "p1_ht": row.winner_ht, "p2_ht": row.loser_ht,
                "p1_hand": _encode_hand(row.winner_hand),
                "p2_hand": _encode_hand(row.loser_hand),
                "p1_wins": 1,
            }
        else:
            p1_id, p2_id = l_id, w_id
            features = {
                "p1_elo": l_elo, "p2_elo": w_elo,
                "p1_surface_elo": l_selo, "p2_surface_elo": w_selo,
                "p1_form": l_form, "p2_form": w_form,
                "p1_h2h_winrate": 1 - w_h2h, "p2_h2h_winrate": w_h2h,
                "p1_fatigue": l_fatigue, "p2_fatigue": w_fatigue,
                "p1_matches_played": l_played, "p2_matches_played": w_played,
                "p1_rank": row.loser_rank, "p2_rank": row.winner_rank,
                "p1_rank_pts": row.loser_rank_points, "p2_rank_pts": row.winner_rank_points,
                "p1_age": row.loser_age, "p2_age": row.winner_age,
                "p1_ht": row.loser_ht, "p2_ht": row.winner_ht,
                "p1_hand": _encode_hand(row.loser_hand),
                "p2_hand": _encode_hand(row.winner_hand),
                "p1_wins": 0,
            }

        features.update({
            "p1_id": p1_id, "p2_id": p2_id,
            "tourney_date": date, "year": date.year,
            "surface": surface, "best_of": row.best_of,
            "round_enc": _encode_round(row.round),
            "level_enc": _encode_level(row.tourney_level),
            "h2h_n": h2h_n,
            # Required by the meta-model (set-by-set parsing):
            "match_num": row.match_num,
            "score": getattr(row, "score", ""),
        })

        # Rolling serve/return stats (assigned according to the swap)
        p1_ss, p2_ss = (w_ss, l_ss) if not swap else (l_ss, w_ss)
        for m in SERVE_METRICS:
            features[f"p1_{m}"] = p1_ss[m]
            features[f"p2_{m}"] = p2_ss[m]

        # Closing odds as a feature (p1's implied prob, NaN if no odds)
        bpp1 = np.nan
        if odds_index is not None:
            pw = _book_winner_prob(
                odds_index,
                sackmann_name_to_key(row.winner_name),
                sackmann_name_to_key(row.loser_name),
                date,
            )
            if pw is not None:
                # p1 is the winner when there is NO swap (criterion independent of the outcome)
                bpp1 = pw if not swap else (1.0 - pw)
        features["book_proba_p1"] = bpp1

        rows.append(features)
        state.update(w_id, l_id, surface, date, getattr(row, "minutes", None))
        state.update_serve(w_id, l_id, w_raw, l_raw)

    out = pd.DataFrame(rows)
    out["elo_diff"] = out["p1_elo"] - out["p2_elo"]
    out["surface_elo_diff"] = out["p1_surface_elo"] - out["p2_surface_elo"]
    out["form_diff"] = out["p1_form"] - out["p2_form"]
    out["fatigue_diff"] = out["p1_fatigue"] - out["p2_fatigue"]
    out["rank_diff_log"] = (
        np.log1p(out["p2_rank"].fillna(2000)) - np.log1p(out["p1_rank"].fillna(2000))
    )
    out["rank_pts_diff"] = (
        out["p1_rank_pts"].fillna(0) - out["p2_rank_pts"].fillna(0)
    )
    out["age_diff"] = out["p1_age"] - out["p2_age"]
    out["ht_diff"] = out["p1_ht"] - out["p2_ht"]
    out["hand_diff"] = out["p1_hand"] - out["p2_hand"]
    for m in SERVE_METRICS:
        out[f"{m}_diff"] = out[f"p1_{m}"] - out[f"p2_{m}"]
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        out[f"surface_{surf}"] = (out["surface"] == surf).astype(int)

    return out, state


if __name__ == "__main__":
    main()
