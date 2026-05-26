"""
Predizione P(podio) per una gara, usando SOLO informazione pre-gara:
  - feature rolling dalle gare passate (forma, podium rate, affidabilita', track)
  - griglia + gap qualifica del round (note dopo le qualifiche del sabato)

Funziona sia per un round gia' corso (demo) sia per uno FUTURO ma con qualifiche
gia' disputate (inietta l'entry list dalle qualifiche). Usa il modello salvato da
`python -m f1_podium` (f1_podium.json + f1_calibrator.pkl).

Uso:
    python -m f1_predict --year 2026 --round 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
import f1_tcn_core as TC
from f1_data import PROCESSED_DIR, fetch_round_quali, load_results


def _inject_future_round(df: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Se il round non e' nei risultati, aggiunge le righe dalla qualifica (entry+griglia)."""
    q = fetch_round_quali(year, rnd)
    if not q:
        raise RuntimeError(f"Qualifiche {year} round {rnd} non disponibili.")
    best = [r["best_ms"] for r in q["rows"] if r["best_ms"]]
    pole = min(best) if best else None
    # gap dal compagno
    by_team: dict[str, list] = {}
    for r in q["rows"]:
        if r["best_ms"]:
            by_team.setdefault(r["constructor"], []).append(r["best_ms"])
    new = []
    for r in q["rows"]:
        bm = r["best_ms"]
        mates = [t for t in by_team.get(r["constructor"], []) if t != bm]
        new.append({
            "season": year, "round": rnd, "date": pd.Timestamp(q["date"]),
            "circuit": q["circuit"], "race": f"Round {rnd}",
            "driver": r["driver"], "constructor": r["constructor"],
            "grid": r["grid"], "position": 99, "position_text": "",
            "status": "", "points": 0.0, "finished": False, "podium": 0,
            "quali_gap_ms": (bm - pole) if (bm and pole) else float("nan"),
            "teammate_gap_ms": (bm - min(mates)) if (bm and mates) else float("nan"),
        })
    return pd.concat([df, pd.DataFrame(new)], ignore_index=True)


def predict_race(year: int, rnd: int):
    df = load_results()
    if not ((df["season"] == year) & (df["round"] == rnd)).any():
        print(f"Round {year}-{rnd} non ancora nei risultati: uso le qualifiche.")
        df = _inject_future_round(df, year, rnd)
    df = df.sort_values(["date", "round"]).reset_index(drop=True)

    feat = F.build_features(df)
    # p_tcn = media di N TCN seed (riduce varianza); fallback a f1_tcn.pt singolo
    cols = F.FEATURE_COLS[:]
    tcn_seeds = sorted(PROCESSED_DIR.glob("f1_tcn_seed*.pt"))
    fallback = PROCESSED_DIR / "f1_tcn.pt"
    paths = tcn_seeds or ([fallback] if fallback.exists() else [])
    if paths:
        seqs, _ = TC.build_sequences(df)
        acc = np.zeros(len(seqs), dtype=np.float32)
        for p in paths:
            acc += TC.predict_tcn(TC.load(p), seqs)
        feat = feat.assign(p_tcn=acc / len(paths))
        cols = F.FEATURE_COLS + ["p_tcn"]
    sel = feat[(feat["season"] == year) & (feat["round"] == rnd)].copy()
    if len(sel) == 0:
        raise RuntimeError("Nessuna riga per quel round.")

    # produzione = CatBoost (f1_podium.cbm); fallback XGB se solo .json esiste
    cbm_path = PROCESSED_DIR / "f1_podium.cbm"
    xgb_path = PROCESSED_DIR / "f1_podium.json"
    if cbm_path.exists():
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(); model.load_model(str(cbm_path))
        p = model.predict_proba(sel[cols].values)[:, 1]
    else:
        model = xgb.Booster(); model.load_model(str(xgb_path))
        p = model.predict(xgb.DMatrix(sel[cols], feature_names=cols))
    cal_path = PROCESSED_DIR / "f1_calibrator.pkl"
    if cal_path.exists():
        p = joblib.load(cal_path).predict(p)
    sel["p_podium"] = p
    sel = sel.sort_values("p_podium", ascending=False)

    print(f"\nP(podio) — {year} round {rnd}  (solo info pre-gara)")
    print(f"{'pilota':22s} {'griglia':>7s} {'P(podio)':>9s}")
    for r in sel.itertuples(index=False):
        print(f"  {r.driver:20s} {int(r.grid):>7d} {r.p_podium:>8.1%}")
    print(f"\nPodio previsto (top-3): {', '.join(sel['driver'].head(3))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    predict_race(args.year, args.round)


if __name__ == "__main__":
    main()
