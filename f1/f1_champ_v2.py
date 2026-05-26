"""
v2 model for P(wins championship): cross-driver softmax + temperature scaling.
- CatBoost raw score (logit) for each (season, round, driver)
- Per (season, round): softmax over active drivers, P sums to 1
- Temperature tau optimised on val (minimise Brier or NLL)
- Cross-season calibration check: reliability diagram + ECE
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR, load_results
from f1_champ import FEAT_COLS, build_champ_dataset


def softmax_per_group(logits, group_ids, T=1.0):
    """Softmax with temperature T applied per group (season, round)."""
    out = np.zeros_like(logits, dtype=np.float64)
    for g in np.unique(group_ids):
        mask = group_ids == g
        z = logits[mask] / T
        z = z - z.max()  # numerical stability
        ez = np.exp(z)
        out[mask] = ez / ez.sum()
    return out


def fit_temperature(logits_val, y_val, group_val, metric="brier"):
    """Optimise temperature scaling. Bound T>=1.0 (no anti-calibration)."""
    def loss(T):
        p = softmax_per_group(logits_val, group_val, T=T)
        if metric == "brier":
            return ((p - y_val) ** 2).mean()
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
    res = minimize_scalar(loss, bounds=(1.0, 15.0), method="bounded")
    return float(res.x), float(res.fun)


def ece(p, y, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0; n = len(p)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean(); acc = y[mask].mean()
        err += mask.sum() / n * abs(conf - acc)
    return err


def main():
    df = load_results()
    ds = build_champ_dataset(df)
    print(f"dataset: {len(ds):,} righe, {ds.season.nunique()} stagioni")

    test_y = 2025; train_max = 2019  # val = last 5 seasons (2020-2024)
    tr = ds[ds.season <= train_max].copy()
    va = ds[(ds.season > train_max) & (ds.season < test_y)].copy()
    te = ds[ds.season == test_y].copy()
    Xtr, ytr = tr[FEAT_COLS].values, tr["is_champ"].values
    Xva, yva = va[FEAT_COLS].values, va["is_champ"].values
    Xte, yte = te[FEAT_COLS].values, te["is_champ"].values
    w = np.exp(-(train_max - tr["season"].values) / 10.0)

    # CatBoost RAW logits (no internal calibration - uses RawFormulaVal)
    m = CatBoostClassifier(
        iterations=2000, depth=5, learning_rate=0.03,
        boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=20, loss_function="Logloss",
        random_seed=42, allow_writing_files=False, verbose=False,
        early_stopping_rounds=80, task_type="CPU",
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
    logit_va = m.predict(Xva, prediction_type="RawFormulaVal")
    logit_te = m.predict(Xte, prediction_type="RawFormulaVal")

    # group IDs (season, round)
    g_va = (va["season"].astype(str) + "_" + va["round"].astype(str)).values
    g_te = (te["season"].astype(str) + "_" + te["round"].astype(str)).values

    # baseline: no softmax (sigmoid raw)
    p_sig_te = 1 / (1 + np.exp(-logit_te))
    print(f"\n[baseline sigmoid raw, no calibration]")
    print(f"  Brier {brier_score_loss(yte, p_sig_te):.4f}  ECE {ece(p_sig_te, yte):.4f}  "
          f"AUC {roc_auc_score(yte, p_sig_te):.3f}")

    # softmax T=1 (no scaling)
    p_sm_te = softmax_per_group(logit_te, g_te, T=1.0)
    print(f"\n[softmax cross-driver T=1.0]")
    print(f"  Brier {brier_score_loss(yte, p_sm_te):.4f}  ECE {ece(p_sm_te, yte):.4f}")

    # fit T on val
    T_opt, nll_val = fit_temperature(logit_va, yva, g_va)
    print(f"\n[temperature scaling]: T_opt = {T_opt:.3f}  (NLL_val = {nll_val:.4f})")
    p_cal_te = softmax_per_group(logit_te, g_te, T=T_opt)
    print(f"  Brier {brier_score_loss(yte, p_cal_te):.4f}  ECE {ece(p_cal_te, yte):.4f}")

    # top-1 per round with calibrated softmax
    te2 = te.copy(); te2["p"] = p_cal_te
    hits = tot = 0; sum_p_top = 0
    for rnd, g in te2.groupby("round"):
        top = g.sort_values("p", ascending=False).head(1)
        hits += int(top["is_champ"].iloc[0]); tot += 1
        sum_p_top += float(top["p"].iloc[0])
    print(f"  top-1 per round: {hits}/{tot} = {hits/tot:.3f}  "
          f"mean P(top-pick) = {sum_p_top/tot:.3f}")

    # save model + temperature
    m.save_model(str(PROCESSED_DIR / "f1_champ_v2.cbm"))
    import joblib
    joblib.dump({"T": T_opt}, PROCESSED_DIR / "f1_champ_v2_T.pkl")

    # PREDICT current 2026 round with calibration
    cur_season = int(ds.season.max())
    cur_round = int(ds[ds.season == cur_season]["round"].max())
    snap = ds[(ds.season == cur_season) & (ds["round"] == cur_round)].copy()
    logit_now = m.predict(snap[FEAT_COLS].values, prediction_type="RawFormulaVal")
    g_now = np.zeros(len(snap), dtype=int)  # 1 group (1 race)
    p_now = softmax_per_group(logit_now, g_now, T=T_opt)
    snap["p_champ"] = p_now
    print(f"\n=== PREDICT campione {cur_season} dopo round {cur_round} (calibrato T={T_opt:.2f}) ===")
    show = snap.sort_values("p_champ", ascending=False)[
        ["driver", "champ_rank", "pts_now", "gap_to_leader", "p_champ"]].head(10)
    print(show.to_string(index=False, formatters={"p_champ": "{:.1%}".format}))
    print(f"  sum P_champ = {snap['p_champ'].sum():.3f} (=1.0 ok softmax)")


if __name__ == "__main__":
    main()
