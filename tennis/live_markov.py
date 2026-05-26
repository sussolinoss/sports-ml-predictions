"""
Motore Markov live per il tennis: dato il punteggio CORRENTE e la probabilita' di
ogni giocatore di vincere un punto al proprio servizio, calcola la probabilita'
ESATTA di vincere il match. Serve per l'in-play: confronta questa prob "vera" con
la quota live del bookmaker -> se diverge, c'e' value.

Gerarchia: punto -> game -> set (tiebreak al 6-6) -> match (best-of-3/5).
Tutto in forma chiusa / DP memoizzato: niente simulazione, risultato esatto.

Convenzione: 'a' e 'b' sono i due giocatori. p_a / p_b = P(vince un punto quando
serve LUI). Le funzioni restituiscono sempre P(a vince ...).
"""

from __future__ import annotations

from functools import lru_cache


# ---------------------------------------------------------------------------
# GAME
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _game_from(p: float, i: int, j: int) -> float:
    """P(il server vince il game) dato il punteggio corrente i (server) - j (ricevitore).
    Punti: 0,1,2,3 = 0,15,30,40. Deuce/vantaggi gestiti in forma chiusa."""
    q = 1.0 - p
    if i >= 4 and i - j >= 2:
        return 1.0
    if j >= 4 and j - i >= 2:
        return 0.0
    if i >= 3 and j >= 3:
        deuce = p * p / (1.0 - 2.0 * p * q)  # P(server vince da deuce)
        if i == j:
            return deuce
        if i - j == 1:               # vantaggio server
            return p + q * deuce
        return p * deuce             # vantaggio ricevitore
    return p * _game_from(p, i + 1, j) + q * _game_from(p, i, j + 1)


def game_win_prob(p: float, i: int = 0, j: int = 0) -> float:
    return _game_from(round(p, 6), i, j)


# ---------------------------------------------------------------------------
# TIEBREAK (primo a 7, scarto 2; il server alterna 1-2-2-2-...)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tb_from(pa: float, pb: float, i: int, j: int, a_serving: bool) -> float:
    """P(a vince il tiebreak) dato punteggio i(a)-j(b) e chi serve ORA."""
    if i >= 7 and i - j >= 2:
        return 1.0
    if j >= 7 and j - i >= 2:
        return 0.0
    if i + j > 40:  # overtime estremo: massa di prob ~0, tronca (errore trascurabile)
        return 0.5
    # nel tiebreak il servizio cambia dopo il 1° punto, poi ogni 2 punti
    n = i + j
    # determina chi serve il prossimo punto: a_serving e' gia' "chi serve ora"
    p_a_wins_point = pa if a_serving else (1.0 - pb)
    # prossimo servitore: cambia dopo punti 1,3,5,... (cioe' quando n+1 e' dispari -> dopo il 1°)
    served = n + 1  # numero del punto che si sta giocando (1-based)
    switch = (served % 2 == 1)  # dopo punti dispari (1,3,5..) cambia servitore
    nxt = (not a_serving) if switch else a_serving
    return (p_a_wins_point * _tb_from(pa, pb, i + 1, j, nxt)
            + (1.0 - p_a_wins_point) * _tb_from(pa, pb, i, j + 1, nxt))


def tiebreak_win_prob(pa: float, pb: float, a_serves_first: bool = True,
                      i: int = 0, j: int = 0) -> float:
    return _tb_from(round(pa, 6), round(pb, 6), i, j, a_serves_first)


# ---------------------------------------------------------------------------
# SET (primo a 6, scarto 2; tiebreak al 6-6)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _set_from(pa: float, pb: float, ga: int, gb: int, a_serving: bool,
              pi: int, pj: int) -> float:
    """P(a vince il set). ga-gb game; pi-pj punti del game in corso; a_serving=chi serve il game corrente."""
    # set gia' deciso
    if ga >= 6 and ga - gb >= 2:
        return 1.0
    if gb >= 6 and gb - ga >= 2:
        return 0.0
    if ga == 6 and gb == 6:
        # tiebreak: serve chi e' di turno (a_serving)
        return tiebreak_win_prob(pa, pb, a_serving)

    # gioca il game corrente partendo dal punteggio punti pi-pj
    server_p = pa if a_serving else pb
    p_server_wins_game = _game_from(round(server_p, 6), pi, pj)
    p_a_wins_game = p_server_wins_game if a_serving else (1.0 - p_server_wins_game)
    # game vinto da a -> ga+1, servizio passa all'altro, punti azzerati
    return (p_a_wins_game * _set_from(pa, pb, ga + 1, gb, not a_serving, 0, 0)
            + (1.0 - p_a_wins_game) * _set_from(pa, pb, ga, gb + 1, not a_serving, 0, 0))


def set_win_prob(pa: float, pb: float, ga: int = 0, gb: int = 0,
                 a_serving: bool = True, pi: int = 0, pj: int = 0) -> float:
    return _set_from(round(pa, 6), round(pb, 6), ga, gb, a_serving, pi, pj)


# ---------------------------------------------------------------------------
# MATCH (best-of-3 / best-of-5)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _match_fut(sa: int, sb: int, need: int, p_set: float) -> float:
    """P(a vince il match) da sa-sb set, set futuri con prob p_set (server neutro)."""
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
    P(a vince il match) dallo stato live completo.
      pa,pb      : P(punto al proprio servizio) di a e b
      best_of    : 3 o 5
      sets_a/b   : set gia' vinti
      ga,gb      : game nel set CORRENTE
      a_serving  : a sta servendo il game corrente?
      pi,pj      : punti nel game corrente (0..3 = 0/15/30/40)
    """
    pa, pb = round(pa, 6), round(pb, 6)
    need = best_of // 2 + 1
    # prob che a vinca il SET in corso (esatta, dallo stato attuale)
    p_cur_set = _set_from(pa, pb, ga, gb, a_serving, pi, pj)
    # prob che a vinca un set "neutro" futuro: media tra a-serve-first e b-serve-first
    p_set_neutral = 0.5 * (_set_from(pa, pb, 0, 0, True, 0, 0)
                           + _set_from(pa, pb, 0, 0, False, 0, 0))
    # risolvi prima il set corrente, poi i futuri
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
