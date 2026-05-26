"""
Modello P(vince gara) — label position==1.
Pipeline identica a f1_podium (CatBoost+decay+CPU+seed42) ma label diversa.
A Monaco grid dominante (pole = 80% chance vittoria).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR, load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def precision_at_1(df_eval, prob_col):
    """Top-1 per gara: dei piloti col p piu' alto, quante volte e' il vero vincitore."""
    hits = tot = 0
    for (_, _), g in df_eval.groupby(["season", "round"]):
        top = g.nlargest(1, prob_col)
        hits += int((top["position"] == 1).any()); tot += 1
    return hits / tot if tot else 0.0


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    feat = feat.merge(df[["season", "round", "driver", "position"]],
                      on=["season", "round", "driver"], how="left")
    feat["win"] = (feat["position"] == 1).astype(int)

    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    cols = FEATURE_COLS
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    Xtr, ytr = tr[cols].values, tr["win"].values
    Xva, yva = va[cols].values, va["win"].values
    Xte, yte = te[cols].values, te["win"].values
    w = np.exp(-(train_max - tr["season"].values) / 1.5)

    m = CatBoostClassifier(
        iterations=1500, depth=5, learning_rate=0.03,
        boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=5, loss_function="Logloss",
        random_seed=42, allow_writing_files=False, verbose=False,
        early_stopping_rounds=80, task_type="CPU",
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(m.predict_proba(Xva)[:, 1], yva)
    p_te = cal.predict(m.predict_proba(Xte)[:, 1])
    te["p_win"] = p_te

    print(f"\nTest {test_y}: Brier {brier_score_loss(yte, p_te):.4f}  "
          f"AUC {roc_auc_score(yte, p_te):.3f}")
    p1 = precision_at_1(te, "p_win")
    # baseline grid
    te["neg_grid"] = -te["grid"]
    p1_grid = precision_at_1(te, "neg_grid")
    print(f"Top-1 per gara (winner): modello {p1:.3f}  vs pole {p1_grid:.3f}  "
          f"delta {(p1-p1_grid)*100:+.1f}pt")

    m.save_model(str(PROCESSED_DIR / "f1_winner.cbm"))
    import joblib
    joblib.dump(cal, PROCESSED_DIR / "f1_winner_cal.pkl")

    # softmax cross-driver per gara (sum=1, 1 vincitore)
    from f1_champ_v2 import softmax_per_group
    logit = m.predict(Xte, prediction_type="RawFormulaVal")
    g = (te["season"].astype(str) + "_" + te["round"].astype(str)).values
    p_sm = softmax_per_group(logit, g, T=1.0)
    te["p_win_sm"] = p_sm
    print(f"Softmax T=1: Brier {brier_score_loss(yte, p_sm):.4f}")
    p1_sm = precision_at_1(te, "p_win_sm")
    print(f"Top-1 softmax: {p1_sm:.3f}")

    # predict ultima gara disponibile
    cur_season = int(feat.season.max())
    cur_round = int(feat[feat.season == cur_season]["round"].max())
    snap = feat[(feat.season == cur_season) & (feat["round"] == cur_round)].copy()
    p_now = cal.predict(m.predict_proba(snap[cols].values)[:, 1])
    logit_now = m.predict(snap[cols].values, prediction_type="RawFormulaVal")
    p_now_sm = softmax_per_group(logit_now, np.zeros(len(snap), dtype=int), T=1.0)
    snap["p_win"] = p_now; snap["p_win_sm"] = p_now_sm
    print(f"\n=== P(vince gara) — {cur_season} R{cur_round} ===")
    print(f"{'driver':<20}{'grid':>5}{'p_iso':>9}{'p_softmax':>11}")
    for _, r in snap.sort_values("p_win_sm", ascending=False).head(10).iterrows():
        print(f"  {r['driver']:<18}{int(r['grid']):>5}{r['p_win']*100:>8.1f}%{r['p_win_sm']*100:>10.1f}%")
    print(f"  sum softmax = {snap['p_win_sm'].sum():.3f}")


if __name__ == "__main__":
    main()
