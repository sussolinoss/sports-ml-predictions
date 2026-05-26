"""
Walk-forward validation — il test ONESTO per capire se l'edge e' reale.

Niente cherry-picking: si definisce UN filtro (min_edge/level/surface/book) e lo si
applica identico su piu' finestre temporali consecutive, ognuna testata su dati MAI
visti in training. Per ogni finestra: ritraina XGBoost + calibrazione isotonica sul
passato, predice la finestra, scommette col filtro fisso. Alla fine: ROI per finestra
+ intervallo di confidenza 95% bootstrap sul pool di TUTTE le bet.

Se l'IC del pool e' sopra ~+1.5% su tutte/quasi le finestre -> edge plausibilmente reale.
Se contiene zero (probabile) -> non scommettere, e' rumore.

USO:
    python -m walkforward --start 2024-01-01 --end 2025-09-30 --step 3 \\
        --book PS --min_edge 0.04 --level GrandSlam,Masters1000 --surface Hard,Grass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BURNIN_END_YEAR,
    EARLY_STOPPING_ROUNDS,
    PROCESSED_DIR,
    RANDOM_SEED,
    XGB_PARAMS,
)
from feature_engineering import FEATURE_COLUMNS
from odds_loader import build_pair_index, load_odds
from backtest import bets_from_predictions, bootstrap_roi, _id_to_name

VAL_DAYS = 90  # ultimi 90 giorni del training usati per early-stopping + calibrazione


def _train_window(train_df: pd.DataFrame, val_df: pd.DataFrame):
    params = XGB_PARAMS.copy()
    n_estimators = params.pop("n_estimators")
    dtr = xgb.DMatrix(train_df[FEATURE_COLUMNS], label=train_df["p1_wins"],
                      feature_names=FEATURE_COLUMNS)
    dva = xgb.DMatrix(val_df[FEATURE_COLUMNS], label=val_df["p1_wins"],
                      feature_names=FEATURE_COLUMNS)
    model = xgb.train(params, dtr, num_boost_round=n_estimators,
                      evals=[(dva, "val")], early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      verbose_eval=False)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(model.predict(dva), val_df["p1_wins"].values)
    return model, cal


def _predict_window(model, cal, test_df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict(xgb.DMatrix(test_df[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS))
    proba = cal.predict(proba)
    out = test_df[["tourney_date", "surface", "level_enc", "p1_id", "p2_id", "p1_wins",
                   "p1_rank", "p2_rank", "p1_matches_played", "p2_matches_played"]].copy()
    out["p1_proba"] = proba
    out["week"] = out["tourney_date"].dt.to_period("W-MON").astype(str)
    out["correct"] = ((proba > 0.5).astype(int) == out["p1_wins"]).astype(int)
    return out


def walk_forward(start, end, step_months=3, book=None, min_edge=0.0, kelly=0.0,
                 min_odd=0.0, max_odd=float("inf"), max_fav_rank=float("inf"),
                 min_fav_rank=0.0, min_prob=0.0, levels=None, surfaces=None):
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    id2name = _id_to_name()

    edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                          freq=pd.DateOffset(months=step_months)).tolist()
    if edges[-1] < pd.Timestamp(end):
        edges.append(pd.Timestamp(end))

    all_bets = []
    rows = []
    print(f"\nWalk-forward: {len(edges)-1} finestre da {step_months} mesi "
          f"(filtro: edge>{min_edge}, book={book}, level={levels}, surface={surfaces})")
    print(f"  {'finestra':23s}  {'n_bet':>5s}  {'hit':>6s}  {'ROI%':>8s}")

    for i in range(len(edges) - 1):
        w_start, w_end = edges[i], edges[i + 1]
        train_all = df[df["tourney_date"] < w_start]
        train_all = train_all[train_all["year"] > BURNIN_END_YEAR]
        test_df = df[(df["tourney_date"] >= w_start) & (df["tourney_date"] < w_end)]
        if len(test_df) == 0 or len(train_all) < 2000:
            continue

        val_cut = w_start - pd.Timedelta(days=VAL_DAYS)
        val_df = train_all[train_all["tourney_date"] >= val_cut]
        tr_df = train_all[train_all["tourney_date"] < val_cut]
        if len(val_df) < 200 or len(tr_df) < 1000:
            continue

        model, cal = _train_window(tr_df, val_df)
        fwd = _predict_window(model, cal, test_df)

        pair_index = build_pair_index(
            load_odds(w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"), book=book)
        )
        bets, _, _ = bets_from_predictions(
            fwd, pair_index, id2name, kelly=kelly, min_odd=min_odd, max_odd=max_odd,
            min_edge=min_edge, max_fav_rank=max_fav_rank, min_fav_rank=min_fav_rank,
            min_prob=min_prob, levels=levels, surfaces=surfaces)

        label = f"{w_start.date()}->{w_end.date()}"
        if bets:
            b = pd.DataFrame(bets)
            roi = b["profit"].sum() / b["stake"].sum() * 100
            print(f"  {label:23s}  {len(b):5d}  {b['fav_won'].mean():6.3f}  {roi:+8.2f}")
            all_bets.append(b)
            rows.append({"window": label, "n": len(b), "roi": roi})
        else:
            print(f"  {label:23s}  {'0':>5s}  {'-':>6s}  {'-':>8s}")

    if not all_bets:
        print("\nNessuna bet in nessuna finestra (filtro troppo stretto o niente quote).")
        return

    pool = pd.concat(all_bets, ignore_index=True)
    roi = pool["profit"].sum() / pool["stake"].sum() * 100
    ci = bootstrap_roi(pool, n_boot=10000)
    n_win = len(rows)
    n_pos = sum(1 for r in rows if r["roi"] > 0)

    print("\n" + "=" * 64)
    print("VERDETTO WALK-FORWARD (out-of-sample, niente cherry-picking)")
    print("=" * 64)
    print(f"  Finestre con bet:     {n_win}  (positive: {n_pos}/{n_win})")
    print(f"  Bet totali (pool):    {len(pool)}")
    print(f"  ROI pool:             {roi:+.2f}%")
    print(f"  IC 95% bootstrap:     [{ci['lo']:+.2f}% , {ci['hi']:+.2f}%]")
    print(f"  P(ROI > 0):           {ci['p_positive']:.3f}")
    if ci["lo"] > 1.5:
        print("  => EDGE REALE e robusto (IC>+1.5%). Ha senso un bot, con cautela.")
    elif ci["lo"] > 0:
        print("  => Edge debole ma positivo (IC sopra 0). Promettente, serve piu' volume.")
    elif ci["hi"] < 0:
        print("  => PERDITA PROVATA: l'IC e' tutto sotto zero. Strategia perdente, NON scommettere.")
    else:
        print("  => NESSUN edge: l'IC contiene zero (break-even). NON scommettere soldi veri.")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--step", type=int, default=3, help="Mesi per finestra (default 3)")
    p.add_argument("--book", default=None, choices=["PS", "B365", "Avg", "Max"])
    p.add_argument("--min_edge", type=float, default=0.0)
    p.add_argument("--kelly", type=float, default=0.0)
    p.add_argument("--min_odd", type=float, default=0.0)
    p.add_argument("--max_odd", type=float, default=float("inf"))
    p.add_argument("--max_fav_rank", type=float, default=float("inf"),
                   help="Solo favoriti entro questo ranking (es. 30)")
    p.add_argument("--min_fav_rank", type=float, default=0.0,
                   help="Ranking minimo favorito (banda con --max_fav_rank, es. 31..50)")
    p.add_argument("--min_prob", type=float, default=0.0,
                   help="Solo bet dove la prob del modello supera la soglia (es. 0.80)")
    p.add_argument("--level", default=None)
    p.add_argument("--surface", default=None)
    args = p.parse_args()

    levels = set(s.strip() for s in args.level.split(",")) if args.level else None
    surfaces = set(s.strip() for s in args.surface.split(",")) if args.surface else None
    walk_forward(args.start, args.end, step_months=args.step, book=args.book,
                 min_edge=args.min_edge, kelly=args.kelly, min_odd=args.min_odd,
                 max_odd=args.max_odd, max_fav_rank=args.max_fav_rank,
                 min_fav_rank=args.min_fav_rank, min_prob=args.min_prob,
                 levels=levels, surfaces=surfaces)


if __name__ == "__main__":
    main()
