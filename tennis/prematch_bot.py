"""
Bot pre-match in PAPER-TRADING (zero soldi).

Per ogni match di tennis in arrivo (The Odds API):
  1. il modello calcola la prob calibrata del vincitore
  2. edge = prob - 1/quota; se edge > soglia -> value bet
  3. stake "sicuro" = Kelly frazionario (default 1/4) con cap sul bankroll
  4. manda su Telegram: chi vs chi, pronostico, quota+edge, stake, countdown, link
  5. logga tutto in data/prematch_paper.csv (stake finti)

AVVERTENZA: l'edge pre-match e' ~0 dimostrato (walk-forward 8k bet). Questo bot
serve a RACCOGLIERE DATI prospettici in paper-trading, non a guadagnare.

Setup:
  set -x THE_ODDS_API_KEY "..."          # the-odds-api.com (gratis 500/mese)
  set -x TELEGRAM_BOT_TOKEN "..."        # @BotFather
  set -x TELEGRAM_CHAT_ID "..."          # il tuo chat id

Uso:
  python -m prematch_bot --dry-run                 # stampa a video, niente Telegram
  python -m prematch_bot --bankroll 200 --min_edge 0.04 --kelly 0.25
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TOUR_DIR
from odds_api import infer_surface_bestof, upcoming_tennis
from predict import predict_match

PAPER_LOG = TOUR_DIR / "prematch_paper.csv"


def _countdown(commence) -> str:
    delta = commence - datetime.now(timezone.utc)
    secs = delta.total_seconds()
    if secs < 0:
        return "gia' iniziato"
    h, m = int(secs // 3600), int((secs % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _kelly_stake(prob: float, odd: float, bankroll: float,
                 kelly_frac: float, max_frac: float) -> tuple[float, float]:
    """Kelly frazionario con cap. Ritorna (stake_eur, frazione_bankroll)."""
    b = odd - 1.0
    f_full = (prob * odd - 1.0) / b if b > 0 else 0.0   # Kelly pieno
    frac = max(0.0, min(max_frac, kelly_frac * f_full))
    return round(frac * bankroll, 2), frac


def _telegram(msg: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  ! TELEGRAM_BOT_TOKEN/CHAT_ID mancanti: salto invio")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=15)
    return r.status_code == 200


def _log(row: dict):
    PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not PAPER_LOG.exists()
    with open(PAPER_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def evaluate_match(ev: dict, bankroll, min_edge, kelly_frac, max_frac):
    """Ritorna (messaggio, row_log) se c'e' value, altrimenti None."""
    surface, best_of, level = infer_surface_bestof(ev["sport_title"])
    try:
        res = predict_match(ev["p1"], ev["p2"], surface, best_of=best_of, level=level)
    except Exception:
        return None  # giocatore non nei dati / nome non matchato

    prob_p1 = res["proba_p1_wins"]
    for name, prob in [(ev["p1"], prob_p1), (ev["p2"], 1 - prob_p1)]:
        odd = ev["odds"][name]
        if odd <= 1.0:
            continue
        edge = prob - 1.0 / odd
        if edge <= min_edge:
            continue
        stake_eur, frac = _kelly_stake(prob, odd, bankroll, kelly_frac, max_frac)
        if stake_eur <= 0:
            continue
        opp = ev["p2"] if name == ev["p1"] else ev["p1"]
        msg = (
            f"🎾 <b>{name}</b> vs {opp}\n"
            f"🏆 {ev['sport_title']}\n"
            f"🔮 Pronostico: <b>{name}</b> ({prob:.0%})\n"
            f"📊 Quota {ev['book']}: <b>{odd:.2f}</b>  |  Edge: {edge:+.1%}\n"
            f"💰 Punta: <b>{stake_eur:.2f}€</b> ({frac:.1%} bankroll, ¼-Kelly)\n"
            f"⏰ Inizia tra: {_countdown(ev['commence_time'])}\n"
            f"🔗 {ev['link']}\n"
            f"<i>paper-trading — edge non provato, traccia e basta</i>"
        )
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_id": ev["event_id"], "sport_key": ev["sport_key"],
            "match": f"{ev['p1']} vs {ev['p2']}", "pick": name, "prob": round(prob, 4),
            "odd": odd, "edge": round(edge, 4), "stake_eur": stake_eur,
            "book": ev["book"], "commence": ev["commence_time"].isoformat(),
            "result": "", "pl": "",   # riempiti da settle_paper.py
        }
        return msg, row
    return None


def run(bankroll, min_edge, kelly_frac, max_frac, book, dry_run):
    events = upcoming_tennis(book=book)
    print(f"{len(events)} match in arrivo da The Odds API")
    sent = 0
    for ev in events:
        r = evaluate_match(ev, bankroll, min_edge, kelly_frac, max_frac)
        if not r:
            continue
        msg, row = r
        print("\n" + msg)
        if not dry_run:
            _telegram(msg)
        _log(row)
        sent += 1
    print(f"\n{sent} value bet trovate e loggate in {PAPER_LOG}"
          + (" (dry-run, niente Telegram)" if dry_run else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--min_edge", type=float, default=0.04)
    p.add_argument("--kelly", type=float, default=0.25, help="Frazione di Kelly (¼ = sicuro)")
    p.add_argument("--max_frac", type=float, default=0.02, help="Cap stake sul bankroll (2%%)")
    p.add_argument("--book", default="pinnacle", help="Book per le quote (default pinnacle)")
    p.add_argument("--dry-run", action="store_true", help="Stampa a video, niente Telegram")
    args = p.parse_args()
    run(args.bankroll, args.min_edge, args.kelly, args.max_frac, args.book, args.dry_run)


if __name__ == "__main__":
    main()
