"""
Live Markov engine for tennis: given the CURRENT score and each player's
probability of winning a point on their own serve, compute the EXACT probability
of winning the match. Used for in-play: compare this "true" prob against the live
bookmaker odds -> if they diverge, there is value.

Hierarchy: point -> game -> set (tiebreak at 6-6) -> match (best-of-3/5).
All in closed form / memoized DP: no simulation, exact result.

Convention: 'a' and 'b' are the two players. p_a / p_b = P(winning a point when
HE serves). The functions always return P(a wins ...).
"""

from __future__ import annotations

from functools import lru_cache


# ---------------------------------------------------------------------------
# GAME
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _game_from(p: float, i: int, j: int) -> float:
    """P(server wins the game) given the current score i (server) - j (receiver).
    Points: 0,1,2,3 = 0,15,30,40. Deuce/advantage handled in closed form."""
    q = 1.0 - p
    if i >= 4 and i - j >= 2:
        return 1.0
    if j >= 4 and j - i >= 2:
        return 0.0
    if i >= 3 and j >= 3:
        deuce = p * p / (1.0 - 2.0 * p * q)  # P(server wins from deuce)
        if i == j:
            return deuce
        if i - j == 1:               # server advantage
            return p + q * deuce
        return p * deuce             # receiver advantage
    return p * _game_from(p, i + 1, j) + q * _game_from(p, i, j + 1)


def game_win_prob(p: float, i: int = 0, j: int = 0) -> float:
    return _game_from(round(p, 6), i, j)


# TIEBREAK (first to 7, margin 2; the server alternates 1-2-2-2-...)
@lru_cache(maxsize=None)
def _tb_from(pa: float, pb: float, i: int, j: int, a_serving: bool) -> float:
    """P(a wins the tiebreak) given score i(a)-j(b) and who serves NOW."""
    if i >= 7 and i - j >= 2:
        return 1.0
    if j >= 7 and j - i >= 2:
        return 0.0
    if i + j > 40:  # extreme overtime: prob mass ~0, truncate (negligible error)
        return 0.5
    # in the tiebreak the serve changes after the 1st point, then every 2 points
    n = i + j
    # determine who serves the next point: a_serving is already "who serves now"
    p_a_wins_point = pa if a_serving else (1.0 - pb)
    # next server: changes after points 1,3,5,... (i.e. when n+1 is odd -> after the 1st)
    served = n + 1  # number of the point being played (1-based)
    switch = (served % 2 == 1)  # after odd points (1,3,5..) the server changes
    nxt = (not a_serving) if switch else a_serving
    return (p_a_wins_point * _tb_from(pa, pb, i + 1, j, nxt)
            + (1.0 - p_a_wins_point) * _tb_from(pa, pb, i, j + 1, nxt))


def tiebreak_win_prob(pa: float, pb: float, a_serves_first: bool = True,
                      i: int = 0, j: int = 0) -> float:
    return _tb_from(round(pa, 6), round(pb, 6), i, j, a_serves_first)


# SET (first to 6, margin 2; tiebreak at 6-6)
@lru_cache(maxsize=None)
def _set_from(pa: float, pb: float, ga: int, gb: int, a_serving: bool,
              pi: int, pj: int) -> float:
    """P(a wins the set). ga-gb games; pi-pj points of the current game; a_serving=who serves the current game."""
    # set already decided
    if ga >= 6 and ga - gb >= 2:
        return 1.0
    if gb >= 6 and gb - ga >= 2:
        return 0.0
    if ga == 6 and gb == 6:
        # tiebreak: served by whoever is on turn (a_serving)
        return tiebreak_win_prob(pa, pb, a_serving)

    # play the current game starting from point score pi-pj
    server_p = pa if a_serving else pb
    p_server_wins_game = _game_from(round(server_p, 6), pi, pj)
    p_a_wins_game = p_server_wins_game if a_serving else (1.0 - p_server_wins_game)
    # game won by a -> ga+1, serve passes to the other, points reset
    return (p_a_wins_game * _set_from(pa, pb, ga + 1, gb, not a_serving, 0, 0)
            + (1.0 - p_a_wins_game) * _set_from(pa, pb, ga, gb + 1, not a_serving, 0, 0))


def set_win_prob(pa: float, pb: float, ga: int = 0, gb: int = 0,
                 a_serving: bool = True, pi: int = 0, pj: int = 0) -> float:
    return _set_from(round(pa, 6), round(pb, 6), ga, gb, a_serving, pi, pj)


# MATCH (best-of-3 / best-of-5)
@lru_cache(maxsize=None)
def _match_fut(sa: int, sb: int, need: int, p_set: float) -> float:
    """P(a wins the match) from sa-sb sets, future sets with prob p_set (neutral server)."""
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    return p_set * _match_fut(sa + 1, sb, need, p_set) + (1.0 - p_set) * _match_fut(sa, sb + 1, need, p_set)


def match_win_prob(pa: float, pb: float, best_of: int = 3,
                   sets_a: int = 0, sets_b: int = 0,
                   ga: int = 0, gb: int = 0, a_serving: bool = True,
                   pi: int = 0, pj: int = 0) -> float:
    """
    P(a wins the match) from the full live state.
      pa,pb      : P(point on own serve) for a and b
      best_of    : 3 or 5
      sets_a/b   : sets already won
      ga,gb      : games in the CURRENT set
      a_serving  : is a serving the current game?
      pi,pj      : points in the current game (0..3 = 0/15/30/40)
    """
    pa, pb = round(pa, 6), round(pb, 6)
    need = best_of // 2 + 1
    # prob that a wins the CURRENT set (exact, from the current state)
    p_cur_set = _set_from(pa, pb, ga, gb, a_serving, pi, pj)
    # prob that a wins a future "neutral" set: average of a-serve-first and b-serve-first
    p_set_neutral = 0.5 * (_set_from(pa, pb, 0, 0, True, 0, 0)
                           + _set_from(pa, pb, 0, 0, False, 0, 0))
    # resolve the current set first, then the future ones
    win = (p_cur_set * _match_fut(sets_a + 1, sets_b, need, p_set_neutral)
           + (1.0 - p_cur_set) * _match_fut(sets_a, sets_b + 1, need, p_set_neutral))
    return win


if __name__ == "__main__":
    # Sanity checks
    print("hold p=0.65:", round(game_win_prob(0.65), 4), "(atteso ~0.83)")
    print("tiebreak pari 0.62:", round(tiebreak_win_prob(0.62, 0.62), 4), "(atteso ~0.50)")
    print("set pari 0.62 da 0-0:", round(set_win_prob(0.62, 0.62), 4), "(atteso ~0.50)")
    print("match pari BO3 da 0-0:", round(match_win_prob(0.62, 0.62, 3), 4), "(atteso ~0.50)")
    print("match A avanti 1 set BO3:", round(match_win_prob(0.62, 0.62, 3, sets_a=1), 4), "(>0.5)")
    print("match A sotto 0-2 BO5:", round(match_win_prob(0.65, 0.60, 5, sets_a=0, sets_b=2), 4))
    print("match A serve 5-4 40-15 set3 BO3 (2 set pari... ):",
          round(match_win_prob(0.66, 0.62, 3, sets_a=1, sets_b=1, ga=5, gb=4, a_serving=True, pi=3, pj=1), 4))
