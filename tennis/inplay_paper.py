"""
In-play value detector in PAPER-TRADING (zero soldi).

Idea: durante un match, il motore Markov (live_markov) calcola la probabilita' VERA
di vittoria dato il punteggio corrente e la forza al servizio dei due giocatori.
La si confronta con la quota LIVE di Betfair: se prob_vera > 1/quota + soglia ->
value bet -> logga (stake finto) su data/inplay_paper.csv.

Forza al servizio (p_serve) = 'serve_pts_won' dalle stat rolling del modello
(final_state.pkl). Probabilita' vera live = live_markov.match_win_prob.

MODI:
  --sim         simula un match punto-punto con quote sintetiche (momentum mispricing)
                e mostra l'intero ciclo: detect -> bet -> settle -> P/L. Nessuna API.
  (default)     live: legge le quote da Betfair, chiede il punteggio corrente a video,
                logga le value bet. (Settlement manuale dopo.)

USO:
    python -m inplay_paper --sim --p1 "Jannik Sinner" --p2 "Daniil Medvedev" --surface Hard --best_of 3
    python -m inplay_paper --market 1.234567890 --p1 "..." --p2 "..."   # live Betfair
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, TOUR_DIR
from live_markov import match_win_prob

PAPER_LOG = TOUR_DIR / "inplay_paper.csv"


# ---------------------------------------------------------------------------
# Forza al servizio dei due giocatori dal modello
# ---------------------------------------------------------------------------
def p_serve_for(p1_name: str, p2_name: str) -> tuple[float, float]:
    import joblib
    from predict import _find_player_id
    state = joblib.load(PROCESSED_DIR / "final_state.pkl")
    name_map = joblib.load(PROCESSED_DIR / "name_to_id.pkl")
    id1 = _find_player_id(p1_name, name_map)
    id2 = _find_player_id(p2_name, name_map)
    pa = state.serve_stats(id1)["serve_pts_won"]
    pb = state.serve_stats(id2)["serve_pts_won"]
    return float(pa), float(pb)


# ---------------------------------------------------------------------------
# Logica value + log
# ---------------------------------------------------------------------------
def _log_paper(row: dict):
    PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not PAPER_LOG.exists()
    with open(PAPER_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def value_bets(pa, pb, best_of, state, odd_a, odd_b, min_edge):
    """Ritorna lista di (lato, true_prob, odd, edge) dove c'e' value."""
    sa, sb, ga, gb, srv_a, pi, pj = state
    true_a = match_win_prob(pa, pb, best_of, sets_a=sa, sets_b=sb, ga=ga, gb=gb,
                            a_serving=srv_a, pi=pi, pj=pj)
    out = []
    for side, true_p, odd in [("A", true_a, odd_a), ("B", 1.0 - true_a, odd_b)]:
        if odd and odd > 1.0:
            edge = true_p - 1.0 / odd
            if edge > min_edge:
                out.append((side, true_p, odd, edge))
    return out


# ---------------------------------------------------------------------------
# Simulatore (paper, self-contained)
# ---------------------------------------------------------------------------
def _play_point(p):
    return random.random() < p


def simulate(pa, pb, best_of, p1, p2, min_edge=0.03, margin=0.05, seed=None):
    if seed is not None:
        random.seed(seed)
    need = best_of // 2 + 1
    sa = sb = 0
    open_bets = []
    print(f"\n--- SIM {p1} (p_serve {pa:.3f}) vs {p2} (p_serve {pb:.3f}), BO{best_of} ---")

    while sa < need and sb < need:
        ga = gb = 0
        srv_a = (sa + sb) % 2 == 0  # alterna chi inizia a servire il set
        while not ((ga >= 6 or gb >= 6) and abs(ga - gb) >= 2) and not (ga == 7 or gb == 7):
            # stato pre-game: calcola true prob e quota sintetica di mercato
            state = (sa, sb, ga, gb, srv_a, 0, 0)
            true_a = match_win_prob(pa, pb, best_of, sets_a=sa, sets_b=sb, ga=ga, gb=gb,
                                    a_serving=srv_a)
            # mercato = true con margine + rumore di momentum (a volte sbaglia)
            noise = random.uniform(-0.06, 0.06)
            mkt_a = min(0.98, max(0.02, true_a + noise))
            odd_a = round(1.0 / (mkt_a * (1 + margin)), 2)
            odd_b = round(1.0 / ((1 - mkt_a) * (1 + margin)), 2)
            for side, tp, odd, edge in value_bets(pa, pb, best_of, state, odd_a, odd_b, min_edge):
                open_bets.append({"side": side, "odd": odd, "edge": edge,
                                  "score": f"{sa}-{sb} {ga}-{gb}"})
            # gioca il game (server vince ogni punto con la sua p)
            p_srv = pa if srv_a else pb
            a_pts = b_pts = 0
            while not ((a_pts >= 4 or b_pts >= 4) and abs(a_pts - b_pts) >= 2):
                srv_wins = _play_point(p_srv)
                a_win = srv_wins if srv_a else (not srv_wins)
                if a_win:
                    a_pts += 1
                else:
                    b_pts += 1
            if a_pts > b_pts:
                ga += 1
            else:
                gb += 1
            srv_a = not srv_a
        if ga > gb:
            sa += 1
        else:
            sb += 1
        print(f"  set finito: {sa}-{sb}")

    a_won = sa > sb
    print(f"  RISULTATO: {'A' if a_won else 'B'} vince ({sa}-{sb})")

    # settle paper bets
    staked = profit = 0.0
    for bet in open_bets:
        win = (bet["side"] == "A") == a_won
        pl = (bet["odd"] - 1.0) if win else -1.0
        staked += 1.0
        profit += pl
        _log_paper({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": "sim", "match": f"{p1} vs {p2}", "side": bet["side"],
            "score": bet["score"], "odd": bet["odd"], "edge": round(bet["edge"], 4),
            "won": int(win), "pl": round(pl, 2),
        })
    print(f"\n  Value bet piazzate (paper): {len(open_bets)}")
    if staked:
        print(f"  Staked {staked:.0f}u  Profit {profit:+.2f}u  ROI {profit/staked*100:+.1f}%")
    print(f"  (loggate in {PAPER_LOG})")
    print("  NB: il +ROI qui e' artificiale, il mercato sintetico ha mispricing finti. "
          "Serve solo a mostrare il ciclo.")


# ---------------------------------------------------------------------------
# Live Betfair (odds reali, punteggio inserito a mano, log paper)
# ---------------------------------------------------------------------------
def live(market_id, pa, pb, best_of, p1, p2, min_edge=0.03, poll=20):
    from betfair_client import BetfairClient
    bf = BetfairClient(); bf.login()
    cat = {m["market_id"]: m for m in bf.list_inplay_tennis()}.get(market_id)
    if not cat:
        print(f"Mercato {market_id} non trovato tra gli in-play. Mercati disponibili:")
        for m in bf.list_inplay_tennis():
            print(f"  {m['market_id']}  {m['event']}  {[r['name'] for r in m['runners']]}")
        return
    runners = cat["runners"]  # [0]=A, [1]=B (ordine Betfair)
    print(f"Live: {cat['event']} | {p1} (p_serve {pa:.3f}) vs {p2} ({pb:.3f})")
    print("A ogni poll inserisci il punteggio. Ctrl-C per uscire.\n")
    try:
        while True:
            prices = bf.best_back_prices(market_id)
            odd_a = prices.get(runners[0]["selection_id"], {}).get("price")
            odd_b = prices.get(runners[1]["selection_id"], {}).get("price")
            print(f"Quote live: A={odd_a}  B={odd_b}")
            raw = input("punteggio 'setsA setsB gamesA gamesB serverA(1/0) puntiA puntiB' (vuoto=skip): ").strip()
            if raw:
                try:
                    sa, sb, ga, gb, srv, pi, pj = [int(x) for x in raw.split()]
                    state = (sa, sb, ga, gb, srv == 1, pi, pj)
                    vb = value_bets(pa, pb, best_of, state, odd_a, odd_b, min_edge)
                    if vb:
                        for side, tp, odd, edge in vb:
                            who = p1 if side == "A" else p2
                            print(f"  >>> VALUE: punta {who} @ {odd} (prob vera {tp:.3f}, edge {edge:+.3f})")
                            _log_paper({"ts": datetime.now().isoformat(timespec="seconds"),
                                        "mode": "live", "match": f"{p1} vs {p2}", "side": side,
                                        "score": f"{sa}-{sb} {ga}-{gb} {pi}-{pj}", "odd": odd,
                                        "edge": round(edge, 4), "won": "", "pl": ""})
                    else:
                        print("  nessun value.")
                except ValueError:
                    print("  formato punteggio non valido, riprovo.")
            time.sleep(poll)
    except KeyboardInterrupt:
        print(f"\nStop. Bet paper loggate in {PAPER_LOG}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1", required=True)
    p.add_argument("--p2", required=True)
    p.add_argument("--surface", default="Hard")
    p.add_argument("--best_of", type=int, default=3, choices=[3, 5])
    p.add_argument("--min_edge", type=float, default=0.03)
    p.add_argument("--sim", action="store_true", help="Simulazione (no API, demo del ciclo)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--market", default=None, help="market_id Betfair per la modalita' live")
    args = p.parse_args()

    pa, pb = p_serve_for(args.p1, args.p2)
    if args.sim:
        simulate(pa, pb, args.best_of, args.p1, args.p2, min_edge=args.min_edge, seed=args.seed)
    elif args.market:
        live(args.market, pa, pb, args.best_of, args.p1, args.p2, min_edge=args.min_edge)
    else:
        print("Specifica --sim (demo) oppure --market <id> (live Betfair).")


if __name__ == "__main__":
    main()
