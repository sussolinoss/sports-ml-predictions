"""Fine scan of decay tau with TCN p_tcn included (production stack)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_tcn_core as TC
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"train<={train_max}  val {val_y}  test {test_y}")

    # ===== TCN 3 seed (veloce) =====
    seqs, y_seq = TC.build_sequences(df)
    # sort feat correlated; sequences allineate sul df originale ordinato
    # Devo re-allineare: f1_tcn build_sequences itera df in ordine arrivo
    # → uso indice df originale (load_results sorted by date implicit)
    df2 = df.sort_values(["season", "round"]).reset_index(drop=True)
    # rebuild seqs allineate al feat sorted
    seqs, y_seq = TC.build_sequences(df2)

    feat2 = feat.merge(df2[["season", "round", "driver"]].reset_index().rename(columns={"index": "_di"}),
                       on=["season", "round", "driver"], how="left")
    order = feat2["_di"].values
    seqs = seqs[order]; y_seq = y_seq[order]

    tr_m = (feat.season <= train_max).to_numpy()
    va_m = (feat.season == val_y).to_numpy()
    te_m = (feat.season == test_y).to_numpy()
    tr_idx = np.where(tr_m)[0]
    p_tcn_acc = np.zeros(len(feat), dtype=np.float32)
    N_TCN = 3
    for seed in range(42, 42 + N_TCN):
        p_tcn = np.zeros(len(feat), dtype=np.float32)
        for ia, ib in KFold(3, shuffle=True, random_state=seed).split(tr_idx):
            a, b = tr_idx[ia], tr_idx[ib]
            m = TC.train_tcn(seqs[a], y_seq[a], seqs[b], y_seq[b], seed=seed)
            p_tcn[b] = TC.predict_tcn(m, seqs[b])
        m_full = TC.train_tcn(seqs[tr_m], y_seq[tr_m], seqs[va_m], y_seq[va_m], seed=seed)
        p_tcn[va_m] = TC.predict_tcn(m_full, seqs[va_m])
        p_tcn[te_m] = TC.predict_tcn(m_full, seqs[te_m])
        p_tcn_acc += p_tcn
        print(f"  TCN seed {seed} done")
    p_tcn = p_tcn_acc / N_TCN
    feat = feat.copy(); feat["p_tcn"] = p_tcn

    cols = FEATURE_COLS + ["p_tcn"]
    tr = feat[tr_m]; va = feat[va_m]; te = feat[te_m].copy()
    Xtr, ytr = tr[cols].values, tr["podium"].values
    Xva, yva = va[cols].values, va["podium"].values
    Xte = te[cols].values
    te["neg_grid"] = -te["grid"]
    print(f"\nbaseline grid: {_precision_at3(te, 'neg_grid'):.3f}\n")

    for tau in [None, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0]:
        ps = []
        for seed in range(42, 47):
            if tau is None:
                w = None
            else:
                age = train_max - tr["season"].values
                w = np.exp(-age / tau)
            m = CatBoostClassifier(
                iterations=1500, depth=5, learning_rate=0.03,
                boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
                l2_leaf_reg=3.0, min_data_in_leaf=5,
                loss_function="Logloss", random_seed=seed,
                allow_writing_files=False, verbose=False,
                early_stopping_rounds=80, task_type="CPU",
            )
            m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(m.predict_proba(Xva)[:, 1], yva)
            te["p"] = cal.predict(m.predict_proba(Xte)[:, 1])
            ps.append(_precision_at3(te, "p"))
        ps = np.array(ps)
        tag = "uniform" if tau is None else f"tau={tau}"
        print(f"  {tag:10s}: mean {ps.mean():.3f}  std {ps.std():.3f}  min {ps.min():.3f}  max {ps.max():.3f}")


if __name__ == "__main__":
    main()
