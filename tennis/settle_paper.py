"""
Settle delle bet paper di prematch_bot: recupera i risultati veri (The Odds API
scores), riempie le colonne result/pl in data/prematch_paper.csv, e stampa il ROI
paper aggiornato con intervallo di confidenza bootstrap.

NB: The Odds API free dà i risultati solo degli ultimi ~3 giorni. Lancia il settle
ogni 1-2 giorni perche' i match piu' vecchi di 3 giorni non saranno piu' recuperabili.

Uso:
    python -m settle_paper
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TOUR_DIR
from odds_api import results

PAPER_LOG = TOUR_DIR / "prematch_paper.csv"


def settle() -> pd.DataFrame:
    if not PAPER_LOG.exists():
        raise RuntimeError(f"Nessun log: {PAPER_LOG}. Lancia prima prematch_bot.")
    df = pd.read_csv(PAPER_LOG)
    df["result"] = df.get("result", "").astype("object")
    df["pl"] = df.get("pl", "")

    pending = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")]
    print(f"{len(df)} bet totali, {len(pending)} da settlare")

    # raggruppa per sport_key per minimizzare le chiamate API
    winners: dict[str, str | None] = {}
    for sport_key in pending["sport_key"].dropna().unique():
        try:
            winners.update(results(str(sport_key)))
        except Exception as e:  # noqa: BLE001
            print(f"  ! risultati {sport_key} non recuperati ({e})")

    settled = 0
    for idx in pending.index:
        ev_id = df.at[idx, "event_id"]
        win = winners.get(ev_id)
        if not win:
            continue  # non ancora completato o fuori finestra 3gg
        pick = df.at[idx, "pick"]
        stake = float(df.at[idx, "stake_eur"])
        odd = float(df.at[idx, "odd"])
        won = (str(pick).strip().lower() == str(win).strip().lower())
        df.at[idx, "result"] = win
        df.at[idx, "pl"] = round(stake * (odd - 1.0) if won else -stake, 2)
        settled += 1

    df.to_csv(PAPER_LOG, index=False)
    print(f"  settlate ora: {settled}")
    return df


def summarize(df: pd.DataFrame):
    done = df[df["pl"].astype(str).str.strip() != ""].copy()
    if len(done) == 0:
        print("\nNessuna bet ancora settlata. Riprova dopo i match.")
        return
    done["pl"] = done["pl"].astype(float)
    done["stake_eur"] = done["stake_eur"].astype(float)
    staked = done["stake_eur"].sum()
    profit = done["pl"].sum()
    roi = profit / staked * 100 if staked else 0.0
    hit = (done["pl"] > 0).mean()
    print("\n" + "=" * 56)
    print("PAPER ROI (bet settlate)")
    print("=" * 56)
    print(f"  Bet settlate:  {len(done)}")
    print(f"  Hit-rate:      {hit:.3f}")
    print(f"  Staked:        {staked:.2f}€")
    print(f"  Profit:        {profit:+.2f}€")
    print(f"  ROI:           {roi:+.2f}%")

    if len(done) >= 30:
        from backtest import bootstrap_roi
        b = done.rename(columns={"stake_eur": "stake"})[["pl", "stake"]].copy()
        b["profit"] = b["pl"]
        ci = bootstrap_roi(b, n_boot=10000)
        print(f"  IC 95%:        [{ci['lo']:+.2f}% , {ci['hi']:+.2f}%]  P(ROI>0)={ci['p_positive']:.3f}")
        if ci["lo"] > 0:
            print("  => paper ROI sopra zero: segnale incoraggiante (serve comunque volume).")
        else:
            print("  => IC contiene zero: ancora break-even, continua a raccogliere.")
    else:
        print(f"  (servono >=30 bet settlate per l'IC bootstrap; ne hai {len(done)})")


def main():
    df = settle()
    summarize(df)


if __name__ == "__main__":
    main()
