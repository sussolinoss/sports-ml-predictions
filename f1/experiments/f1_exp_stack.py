"""
2-level stacking: 4 base learners + LR meta on val.
Base: CatBoost(decay), XGBC(decay), XGBRanker, TCN(prod stack).
Meta: LR on [p_cat, p_xgc, p_xgr_norm, p_tcn] fit on val, evaluated on test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_tcn_core as TC
from f1_data import PROCESSED_DIR, load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.merge(df[["season", "round", "driver", "position"]],
                      on=["season", "round", "driver"], how="left")
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    # Re-use TCN from saved production file: skip retrain, predict from f1_tcn_seed{1..5}.pt
    df2 = df.sort_values(["season", "round"]).reset_index(drop=True)
    seqs, y_seq = TC.build_sequences(df2)
    feat2 = feat.merge(df2[["season", "round", "driver"]].reset_index().rename(columns={"index": "_di"}),
                       on=["season", "round", "driver"], how="left")
    order = feat2["_di"].values
    seqs = seqs[order]; y_seq = y_seq[order]
    p_tcn = np.zeros(len(feat), dtype=np.float32); cnt = 0
    for sp in sorted(PROCESSED_DIR.glob("f1_tcn_seed*.pt")):
        m = TC.load(sp); p_tcn += TC.predict_tcn(m, seqs); cnt += 1
    p_tcn = (p_tcn / cnt) if cnt else p_tcn
    print(f"TCN avg over {cnt} saved seeds")
    feat = feat.copy(); feat["p_tcn"] = p_tcn

    cols = FEATURE_COLS
    tr_m = (feat.season <= train_max).to_numpy()
    va_m = (feat.season == val_y).to_numpy()
    te_m = (feat.season == test_y).to_numpy()
    tr = feat[tr_m]; va = feat[va_m]; te = feat[te_m].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte, yte = te[cols].values, te["podium"].values

    age = train_max - tr["season"].values
    w = np.exp(-age / 1.5)

    # ===== 1) CatBoost decay =====
    cb = CatBoostClassifier(
        iterations=1500, depth=5, learning_rate=0.03,
        boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=5, loss_function="Logloss",
        random_seed=42, allow_writing_files=False, verbose=False,
        early_stopping_rounds=80, task_type="CPU",
    )
    cb.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
    cal_cb = IsotonicRegression(out_of_bounds="clip")
    cal_cb.fit(cb.predict_proba(Xva)[:, 1], yva)
    p_cat_va = cal_cb.predict(cb.predict_proba(Xva)[:, 1])
    p_cat_te = cal_cb.predict(cb.predict_proba(Xte)[:, 1])

    # ===== 2) XGBC decay =====
    med = np.nanmedian(np.vstack([Xtr, Xva, Xte]), axis=0)
    Xtr2 = np.where(np.isnan(Xtr), med, Xtr)
    Xva2 = np.where(np.isnan(Xva), med, Xva)
    Xte2 = np.where(np.isnan(Xte), med, Xte)
    xgc = xgb.XGBClassifier(
        n_estimators=1500, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=3.0, min_child_weight=5,
        eval_metric="logloss", early_stopping_rounds=80,
        tree_method="hist", random_state=42, verbosity=0,
    )
    xgc.fit(Xtr2, ytr, sample_weight=w, eval_set=[(Xva2, yva)], verbose=False)
    cal_xc = IsotonicRegression(out_of_bounds="clip")
    cal_xc.fit(xgc.predict_proba(Xva2)[:, 1], yva)
    p_xgc_va = cal_xc.predict(xgc.predict_proba(Xva2)[:, 1])
    p_xgc_te = cal_xc.predict(xgc.predict_proba(Xte2)[:, 1])

    # ===== 3) XGBRanker (no decay, label graded position) =====
    def rl(pos):
        return np.where(pos.notna(), np.clip(21 - pos.fillna(99).astype(int), 0, 30), 0).astype(int)
    ltr = rl(tr["position"]); lva = rl(va["position"])
    qtr = tr.groupby(["season", "round"]).size().values
    qva = va.groupby(["season", "round"]).size().values
    dtr = xgb.DMatrix(Xtr2, label=ltr); dtr.set_group(qtr)  # ranker weight is per-group, skip decay
    dva = xgb.DMatrix(Xva2, label=lva); dva.set_group(qva)
    dte = xgb.DMatrix(Xte2)
    xr = xgb.train({"objective": "rank:ndcg", "eval_metric": ["ndcg@3"],
                    "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
                    "subsample": 0.85, "colsample_bytree": 0.85,
                    "lambda": 1.0, "tree_method": "hist", "verbosity": 0},
                   dtr, num_boost_round=2000, evals=[(dva, "val")],
                   early_stopping_rounds=80, verbose_eval=0)
    s_xr_va = xr.predict(xgb.DMatrix(Xva2))
    s_xr_te = xr.predict(dte)

    # ===== 4) TCN (from saved) =====
    p_tcn_va = feat.loc[va_m, "p_tcn"].values
    p_tcn_te = feat.loc[te_m, "p_tcn"].values

    # ===== base learners precision =====
    def pat(score, msk):
        sub = feat[msk].copy(); sub["_s"] = score
        return _precision_at3(sub, "_s")
    print(f"\nbase precision@3 test:")
    print(f"  CatBoost(decay)     {pat(p_cat_te, te_m):.3f}")
    print(f"  XGBC(decay)         {pat(p_xgc_te, te_m):.3f}")
    print(f"  XGBRanker(decay)    {pat(s_xr_te, te_m):.3f}")
    print(f"  TCN(saved avg)      {pat(p_tcn_te, te_m):.3f}")

    # ===== Meta LR on val =====
    # Rank-normalize XGBRanker score per race (for compatibility with prob scale)
    def race_norm(score, msk):
        out = feat[msk].copy(); out["_s"] = score
        out["_r"] = out.groupby(["season", "round"])["_s"].rank()
        n = out.groupby(["season", "round"])["_s"].transform("size")
        return ((out["_r"] - 1) / (n - 1).clip(lower=1)).values
    s_xr_va_n = race_norm(s_xr_va, va_m)
    s_xr_te_n = race_norm(s_xr_te, te_m)

    Mva = np.column_stack([p_cat_va, p_xgc_va, s_xr_va_n, p_tcn_va])
    Mte = np.column_stack([p_cat_te, p_xgc_te, s_xr_te_n, p_tcn_te])

    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(Mva, yva)
    print(f"\nMeta-LR weights: {lr.coef_[0].round(3)}  intercept {lr.intercept_[0]:.3f}")
    p_stack = lr.predict_proba(Mte)[:, 1]
    print(f"Stack meta-LR        {pat(p_stack, te_m):.3f}")

    # also simple mean and weighted-by-val-auc
    p_mean = Mte.mean(axis=1)
    print(f"Stack simple mean    {pat(p_mean, te_m):.3f}")


if __name__ == "__main__":
    main()
