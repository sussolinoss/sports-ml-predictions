"""
Model for OVER/UNDER games and HANDICAP games betting (not "who wins").

1X2 odds are highly efficient; over/under and handicap games are less so. Here
we simulate the match point-by-point (Monte Carlo) from the two players' serve
strength (p_serve from the model's rolling stats) and obtain the distribution of
TOTAL games and game margin -> P(over line), P(handicap covered).

LIMITATION: tennis-data.co.uk has NO historical over/under/handicap odds -> this
is NOT backtestable for free (like in-play). It is used in paper-trading against
live odds (The Odds API 'totals'/'spreads' markets when available).

USAGE:
    python -m games_model --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" --best_of 3
    python -m games_model --pa 0.66 --pb 0.63 --best_of 5      # direct p_serve
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sim_tiebreak(pa: float, pb: float, a_serves_first: bool, rng: random.Random) -> bool:
    """True if A wins the tiebreak. Serve: 1 point A, then 2 each."""
    i = j = 0
    a_serving = a_serves_first
    n = 0
    while not ((i >= 7 or j >= 7) and abs(i - j) >= 2):
        p_srv = pa if a_serving else pb
        srv_wins = rng.random() < p_srv
        a_win = srv_wins if a_serving else (not srv_wins)
        if a_win:
            i += 1
        else:
            j += 1
        n += 1
        if n % 2 == 1:  # change serve after odd points (1,3,5,...)
            a_serving = not a_serving
    return i > j


def _sim_set(pa: float, pb: float, a_serves_first: bool, rng: random.Random):
    """Simulate a set, return (games_a, games_b, a_won)."""
    ga = gb = 0
    srv_a = a_serves_first
    while True:
        if ga == 6 and gb == 6:
            a_tb = _sim_tiebreak(pa, pb, srv_a, rng)
            if a_tb:
                ga += 1
            else:
                gb += 1
            return ga, gb, ga > gb
        if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
            return ga, gb, ga > gb
        p_srv = pa if srv_a else pb
        a_p = b_p = 0
        while not ((a_p >= 4 or b_p >= 4) and abs(a_p - b_p) >= 2):
            srv_wins = rng.random() < p_srv
            a_win = srv_wins if srv_a else (not srv_wins)
            if a_win:
                a_p += 1
            else:
                b_p += 1
        if a_p > b_p:
            ga += 1
        else:
            gb += 1
        srv_a = not srv_a


def _sim_match(pa: float, pb: float, best_of: int, rng: random.Random):
    need = best_of // 2 + 1
    sa = sb = 0
    tot_a = tot_b = 0
    while sa < need and sb < need:
        ga, gb, a_won = _sim_set(pa, pb, (sa + sb) % 2 == 0, rng)
        tot_a += ga
        tot_b += gb
        if a_won:
            sa += 1
        else:
            sb += 1
    return tot_a, tot_b


def simulate(pa: float, pb: float, best_of: int = 3, n: int = 20000, seed: int = 42):
    """Return (game_totals, game_margins=A-B) over n simulations."""
    rng = random.Random(seed)
    totals, margins = [], []
    for _ in range(n):
        a, b = _sim_match(pa, pb, best_of, rng)
        totals.append(a + b)
        margins.append(a - b)
    return totals, margins


def p_over(totals, line: float) -> float:
    return sum(1 for t in totals if t > line) / len(totals)


def p_handicap(margins, line: float) -> float:
    """P(A covers the handicap): A - B + line > 0 (negative line = favored A giving games)."""
    return sum(1 for m in margins if m + line > 0) / len(margins)


def report(pa, pb, best_of, n=20000):
    totals, margins = simulate(pa, pb, best_of, n=n)
    avg = sum(totals) / len(totals)
    print(f"\np_serve A={pa:.3f}  B={pb:.3f}  BO{best_of}  ({n:,} sim)")
    print(f"Totale game: media {avg:.1f}, min {min(totals)}, max {max(totals)}")
    print("\nOVER/UNDER (prob over):")
    base = round(avg)
    for line in [base - 2.5, base - 1.5, base - 0.5, base + 0.5, base + 1.5, base + 2.5]:
        po = p_over(totals, line)
        print(f"  linea {line:5.1f}:  over {po:.3f}  |  under {1-po:.3f}  "
              f"(quota equa over {1/po:.2f} / under {1/(1-po):.2f})")
    print("\nHANDICAP game (prob che A copra):")
    for line in [-4.5, -2.5, -1.5, +1.5, +2.5, +4.5]:
        ph = p_handicap(margins, line)
        print(f"  A {line:+.1f}:  {ph:.3f}  (quota equa {1/ph:.2f})" if ph > 0 else
              f"  A {line:+.1f}:  {ph:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1")
    p.add_argument("--p2")
    p.add_argument("--pa", type=float, help="p_serve A diretto (bypassa il modello)")
    p.add_argument("--pb", type=float, help="p_serve B diretto")
    p.add_argument("--best_of", type=int, default=3, choices=[3, 5])
    p.add_argument("--n", type=int, default=20000)
    args = p.parse_args()

    if args.pa is not None and args.pb is not None:
        pa, pb = args.pa, args.pb
    elif args.p1 and args.p2:
        from inplay_paper import p_serve_for
        pa, pb = p_serve_for(args.p1, args.p2)
    else:
        p.error("Servono --p1/--p2 (dal modello) oppure --pa/--pb (diretti)")
    report(pa, pb, args.best_of, n=args.n)


if __name__ == "__main__":
    main()
