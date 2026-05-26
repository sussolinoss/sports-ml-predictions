"""
Pre-match betting backtest: compares the model's probability with the
bookmaker's closing odds (tennis-data.co.uk) and simulates the ROI of a
value-betting strategy over a time window (e.g. 1 month).

Odds are used ONLY as a market to bet against, NOT as model features.

Logic:
  - for each match in the window, the model indicates the favorite (p>0.5) and
    its probability.
  - the odds for that player are retrieved from the tennis-data file (join on
    the two names, order-independent, with a date tolerance).
  - VALUE BET if  model_proba * odd > 1 + margin.
  - stake: flat 1 unit (default) or fractional Kelly (--kelly).
  - payout: +(odd-1)*stake if the backed player wins, -stake otherwise.

USAGE:
    python -m backtest --start 2024-09-01 --end 2024-09-30
    python -m backtest --start 2024-09-01 --end 2024-09-30 --margin 0.05 --kelly 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, TOUR_DIR
from evaluate import forward_test
from odds_loader import build_pair_index, load_odds, sackmann_name_to_key

DATE_TOLERANCE_DAYS = 14

# Inverse of LEVEL_ORDER in feature_engineering (level_enc -> readable label)
LEVEL_LABELS = {0: "Davis", 1: "Challenger", 2: "Satellite", 3: "ATP250/500",
                4: "Finals", 5: "Masters1000", 6: "GrandSlam"}


def _id_to_name() -> dict[int, str]:
    """id -> Sackmann name (inverts name_to_id.pkl)."""
    name_map = joblib.load(PROCESSED_DIR / "name_to_id.pkl")
    id2name: dict[int, str] = {}
    for name, pid in name_map.items():
        id2name.setdefault(int(pid), name)
    return id2name


def _match_odds(pair_index: dict, key_a: str, key_b: str, when: pd.Timestamp):
    """Find the odds row for the pair, the one with the closest date within tolerance."""
    rows = pair_index.get(frozenset((key_a, key_b)))
    if not rows:
        return None
    best, best_dt = None, None
    for r in rows:
        dt = abs((r["date"] - when).days)
        if dt <= DATE_TOLERANCE_DAYS and (best_dt is None or dt < best_dt):
            best, best_dt = r, dt
    return best


def bets_from_predictions(fwd: pd.DataFrame, pair_index: dict, id2name: dict,
                          margin: float = 0.0, kelly: float = 0.0,
                          min_odd: float = 0.0, max_odd: float = float("inf"),
                          min_edge: float = 0.0, max_fav_rank: float = float("inf"),
                          min_fav_rank: float = 0.0, min_prob: float = 0.0,
                          levels: set | None = None, surfaces: set | None = None):
    """Betting core: from predictions (forward_test schema) + odds index to a bet list.
    Reused by run_backtest and the walk-forward. Returns (bets, n_with_odds, unmatched)."""
    n_with_odds = 0
    unmatched = []
    bets = []
    for r in fwd.itertuples(index=False):
        p1_name = id2name.get(int(r.p1_id))
        p2_name = id2name.get(int(r.p2_id))
        if not p1_name or not p2_name:
            continue
        lvl = LEVEL_LABELS.get(int(r.level_enc), "?")
        if levels is not None and lvl not in levels:
            continue
        if surfaces is not None and r.surface not in surfaces:
            continue
        k1 = sackmann_name_to_key(p1_name)
        k2 = sackmann_name_to_key(p2_name)

        favored_is_p1 = r.p1_proba > 0.5
        model_proba = r.p1_proba if favored_is_p1 else 1 - r.p1_proba
        # "Safe bets" filter: only where the model is very confident
        if model_proba < min_prob:
            continue
        fav_key = k1 if favored_is_p1 else k2
        fav_rank = r.p1_rank if favored_is_p1 else r.p2_rank
        fav_played = r.p1_matches_played if favored_is_p1 else r.p2_matches_played

        # Filter by the favorite's ranking band (structural, not per-name)
        if min_fav_rank > 0 or max_fav_rank != float("inf"):
            if pd.isna(fav_rank) or fav_rank < min_fav_rank or fav_rank > max_fav_rank:
                continue

        odds_row = _match_odds(pair_index, k1, k2, r.tourney_date)
        if odds_row is None:
            unmatched.append((r.tourney_date.date(), p1_name, p2_name, lvl))
            continue
        n_with_odds += 1

        fav_odd = odds_row["odd_winner"] if fav_key == odds_row["winner_key"] else odds_row["odd_loser"]
        if not (min_odd <= fav_odd <= max_odd):
            continue
        edge = model_proba - 1.0 / fav_odd
        if edge <= min_edge:
            continue
        if model_proba * fav_odd <= 1.0 + margin:
            continue

        fav_won = (int(r.p1_wins) == 1) if favored_is_p1 else (int(r.p1_wins) == 0)
        if kelly > 0:
            b = fav_odd - 1.0
            f = (b * model_proba - (1 - model_proba)) / b if b > 0 else 0.0
            stake = max(0.0, kelly * f)
        else:
            stake = 1.0
        if stake <= 0:
            continue
        profit = stake * (fav_odd - 1.0) if fav_won else -stake
        bets.append({
            "tourney_date": r.tourney_date,
            "week": getattr(r, "week", ""),
            "surface": r.surface,
            "level": lvl,
            "fav_player": p1_name if favored_is_p1 else p2_name,
            "fav_rank": fav_rank,
            "fav_played": fav_played,
            "model_proba": model_proba,
            "implied_proba": 1.0 / fav_odd,
            "edge": edge,
            "odd": fav_odd,
            "odd_source": odds_row["source"],
            "stake": stake,
            "fav_won": int(fav_won),
            "profit": profit,
        })
    return bets, n_with_odds, unmatched


def run_backtest(start: str, end: str, margin: float = 0.0, kelly: float = 0.0,
                 min_odd: float = 0.0, max_odd: float = float("inf"),
                 min_edge: float = 0.0, max_fav_rank: float = float("inf"),
                 min_fav_rank: float = 0.0, min_prob: float = 0.0,
                 levels: set | None = None, surfaces: set | None = None,
                 book: str | None = None, debug: bool = False) -> pd.DataFrame:
    """Return a DataFrame with one row per placed value bet."""
    fwd = forward_test(start, end)
    id2name = _id_to_name()
    pair_index = build_pair_index(load_odds(start, end, book=book))

    bets, n_with_odds, unmatched = bets_from_predictions(
        fwd, pair_index, id2name, margin=margin, kelly=kelly,
        min_odd=min_odd, max_odd=max_odd, min_edge=min_edge, max_fav_rank=max_fav_rank,
        min_fav_rank=min_fav_rank, min_prob=min_prob, levels=levels, surfaces=surfaces)

    bets_df = pd.DataFrame(bets)
    bets_df.attrs["n_total"] = len(fwd)
    bets_df.attrs["n_with_odds"] = n_with_odds
    bets_df.attrs["model_acc"] = float(fwd["correct"].mean())
    bets_df.attrs["unmatched"] = unmatched
    bets_df.attrs["debug"] = debug
    return bets_df


def bootstrap_roi(bets_df: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    """95% confidence interval on ROI via bootstrap (resampling with replacement)."""
    profit = bets_df["profit"].to_numpy()
    stake = bets_df["stake"].to_numpy()
    n = len(profit)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stake[idx].sum()
        rois[i] = profit[idx].sum() / s * 100 if s > 0 else 0.0
    return {
        "lo": float(np.percentile(rois, 2.5)),
        "med": float(np.percentile(rois, 50)),
        "hi": float(np.percentile(rois, 97.5)),
        "p_positive": float((rois > 0).mean()),
    }


def summarize(bets_df: pd.DataFrame, n_boot: int = 0) -> None:
    n_total = bets_df.attrs.get("n_total", 0)
    n_with_odds = bets_df.attrs.get("n_with_odds", 0)
    model_acc = bets_df.attrs.get("model_acc", float("nan"))

    print("\n" + "=" * 64)
    print("BACKTEST SCOMMESSE — riepilogo")
    print("=" * 64)
    print(f"Match nella finestra:        {n_total}")
    cov = (n_with_odds / n_total * 100) if n_total else 0.0
    print(f"Match con quota agganciata:  {n_with_odds}  (copertura {cov:.1f}%)")
    print(f"Accuracy modello (tutti i match): {model_acc:.4f}")

    # Debug: unmatched pairs (structural challenger/quali vs unmatched names)
    unmatched = bets_df.attrs.get("unmatched", [])
    if bets_df.attrs.get("debug") and unmatched:
        from collections import Counter
        by_level = Counter(u[3] for u in unmatched)
        print(f"\n[debug] {len(unmatched)} coppie SENZA quota, per livello torneo:")
        for lvl, c in by_level.most_common():
            print(f"    {lvl:14s} {c}")
        print("  prime 15 coppie non agganciate:")
        for d, a, b, lvl in unmatched[:15]:
            print(f"    {d}  {a} vs {b}  [{lvl}]")

    if len(bets_df) == 0:
        print("\nNessuna value bet piazzata (nessun match supera la soglia di margine).")
        return

    staked = bets_df["stake"].sum()
    profit = bets_df["profit"].sum()
    roi = profit / staked * 100 if staked else 0.0
    hit = bets_df["fav_won"].mean()
    print(f"\nValue bet piazzate:          {len(bets_df)}")
    print(f"Hit-rate delle bet:          {hit:.4f}")
    print(f"Totale puntato:              {staked:.2f} unita'")
    print(f"Profitto:                    {profit:+.2f} unita'")
    print(f"ROI:                         {roi:+.2f}%")
    print(f"Quota media:                 {bets_df['odd'].mean():.2f}")

    print("\nPer settimana:")
    wk = bets_df.groupby("week").agg(
        n=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum")
    )
    for w, row in wk.iterrows():
        r = row["profit"] / row["stake"] * 100 if row["stake"] else 0.0
        print(f"  {w:25s}  n={int(row['n']):3d}  profit={row['profit']:+7.2f}  roi={r:+7.2f}%")

    print("\nPer superficie:")
    sf = bets_df.groupby("surface").agg(
        n=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum")
    )
    for s, row in sf.iterrows():
        r = row["profit"] / row["stake"] * 100 if row["stake"] else 0.0
        print(f"  {s:10s}  n={int(row['n']):3d}  profit={row['profit']:+7.2f}  roi={r:+7.2f}%")

    print("\nPer livello torneo (qui si vede dove il modello ha eventuale edge):")
    lv = bets_df.groupby("level").agg(
        n=("profit", "size"), stake=("stake", "sum"),
        profit=("profit", "sum"), hit=("fav_won", "mean")
    )
    for s, row in lv.iterrows():
        r = row["profit"] / row["stake"] * 100 if row["stake"] else 0.0
        print(f"  {s:14s}  n={int(row['n']):3d}  hit={row['hit']:.3f}  "
              f"profit={row['profit']:+7.2f}  roi={r:+7.2f}%")

    print("\nPer fascia di ranking del favorito (filtro strutturale, non per-nome):")
    tiers = [(1, 10, "Top 1-10"), (11, 30, "11-30"), (31, 50, "31-50"),
             (51, 100, "51-100"), (101, 9999, "100+")]
    fr = bets_df["fav_rank"]
    for lo, hi, lab in tiers:
        m = fr.between(lo, hi)
        sub = bets_df[m]
        if len(sub) == 0:
            continue
        roi = sub["profit"].sum() / sub["stake"].sum() * 100
        print(f"  {lab:10s}  n={len(sub):4d}  hit={sub['fav_won'].mean():.3f}  roi={roi:+7.2f}%")

    if n_boot > 0:
        ci = bootstrap_roi(bets_df, n_boot=n_boot)
        print(f"\nSignificativita' (bootstrap {n_boot:,} resampling):")
        print(f"  ROI mediano:    {ci['med']:+.2f}%")
        print(f"  IC 95%:         [{ci['lo']:+.2f}% , {ci['hi']:+.2f}%]")
        print(f"  P(ROI > 0):     {ci['p_positive']:.3f}")
        if ci["lo"] > 0:
            print("  => EDGE REALE: l'intero IC 95% e' sopra zero.")
        elif ci["hi"] < 0:
            print("  => PERDITA REALE: l'intero IC 95% e' sotto zero (margine bookmaker).")
        else:
            print("  => BREAK-EVEN: l'IC 95% contiene zero, ROI non distinguibile da 0 "
                  "(ne' edge ne' perdita provati).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Soglia di value: bet se proba*quota > 1+margine")
    parser.add_argument("--kelly", type=float, default=0.0,
                        help="Frazione di Kelly (0 = stake flat 1 unita')")
    parser.add_argument("--min_edge", type=float, default=0.0,
                        help="Edge minimo (prob_calibrata - 1/quota) per scommettere. "
                             "La metrica giusta: 0.04-0.06 e' il sweet spot tipico")
    parser.add_argument("--sweep", action="store_true",
                        help="Tabella ROI/volume al variare di min_edge (0..0.10)")
    parser.add_argument("--max_fav_rank", type=float, default=float("inf"),
                        help="Scommetti solo se il favorito e' entro questo ranking "
                             "(es. 30 = solo top-30). Filtro strutturale, non per-nome")
    parser.add_argument("--min_fav_rank", type=float, default=0.0,
                        help="Ranking minimo del favorito (banda: usa con --max_fav_rank, "
                             "es. 31..50)")
    parser.add_argument("--min_prob", type=float, default=0.0,
                        help="Scommetti solo se la prob del modello supera questa soglia "
                             "(es. 0.80 = solo 'sicure'). ATTENZIONE: quote corte, edge raro")
    parser.add_argument("--min_odd", type=float, default=0.0,
                        help="Quota minima del favorito per piazzare la bet")
    parser.add_argument("--max_odd", type=float, default=float("inf"),
                        help="Quota massima del favorito (es. 4.0 per restare sugli underdog medi)")
    parser.add_argument("--level", default=None,
                        help="Filtra per livello, separati da virgola "
                             "(es. GrandSlam,Masters1000). Vedi LEVEL_LABELS.")
    parser.add_argument("--surface", default=None,
                        help="Filtra per superficie, separate da virgola "
                             "(es. Hard,Grass per escludere la terra)")
    parser.add_argument("--book", default=None, choices=["PS", "B365", "Avg", "Max"],
                        help="Fonte quote: PS=Pinnacle (sharp, margine basso), "
                             "B365=Bet365, Avg=media, Max=massima. Default: migliore disponibile")
    parser.add_argument("--debug", action="store_true",
                        help="Stampa le coppie senza quota (diagnosi copertura)")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="Bootstrap su N resampling per IC 95%% del ROI (es. 10000)")
    parser.add_argument("--out", default=str(TOUR_DIR / "roi_backtest.csv"))
    args = parser.parse_args()

    levels = set(s.strip() for s in args.level.split(",")) if args.level else None
    surfaces = set(s.strip() for s in args.surface.split(",")) if args.surface else None

    if args.sweep:
        # Same base selection (level/surface/odd/book), edge=0, then filter in memory
        base = run_backtest(args.start, args.end, margin=0.0, kelly=args.kelly,
                            min_odd=args.min_odd, max_odd=args.max_odd, min_edge=-1.0,
                            max_fav_rank=args.max_fav_rank, min_fav_rank=args.min_fav_rank,
                            min_prob=args.min_prob,
                            levels=levels, surfaces=surfaces, book=args.book)
        print(f"\nSWEEP edge — copertura {base.attrs['n_with_odds']}/{base.attrs['n_total']} match")
        print(f"  {'min_edge':>8s}  {'n_bet':>5s}  {'hit':>6s}  {'ROI%':>7s}  {'IC95%':>18s}")
        for thr in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            sub = base[base["edge"] > thr]
            if len(sub) == 0:
                print(f"  {thr:8.2f}  {'0':>5s}")
                continue
            roi = sub["profit"].sum() / sub["stake"].sum() * 100
            ci = bootstrap_roi(sub, n_boot=2000)
            flag = "  <-- IC>0" if ci["lo"] > 0 else ""
            print(f"  {thr:8.2f}  {len(sub):5d}  {sub['fav_won'].mean():6.3f}  "
                  f"{roi:+7.2f}  [{ci['lo']:+6.2f},{ci['hi']:+6.2f}]{flag}")
        return

    bets_df = run_backtest(args.start, args.end, margin=args.margin, kelly=args.kelly,
                           min_odd=args.min_odd, max_odd=args.max_odd, min_edge=args.min_edge,
                           max_fav_rank=args.max_fav_rank, min_fav_rank=args.min_fav_rank,
                           min_prob=args.min_prob,
                           levels=levels, surfaces=surfaces, book=args.book, debug=args.debug)
    summarize(bets_df, n_boot=args.bootstrap)

    if len(bets_df):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        bets_df.to_csv(args.out, index=False)
        print(f"\nDettaglio bet salvato in: {args.out}")


if __name__ == "__main__":
    main()
