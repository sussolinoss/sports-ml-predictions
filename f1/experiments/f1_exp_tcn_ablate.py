"""TCN ablation with decay active. 10 CatBoost seeds, reuses the saved TCN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_tcn_core as TC
from f1_data import PROCESSED_DIR, load_results
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

    # Allineamento TCN sequences con feat sorted
    df2 = df.sort_values(["season", "round"]).reset_index(drop=True)
    seqs, y_seq = TC.build_sequences(df2)
    # Verifica: build_sequences ritorna 1 seq per riga di df2 nell'ordine? Sì.
    feat2 = feat.merge(df2[["season", "round", "driver"]].reset_index().rename(columns={"index": "_di"}),
                       on=["season", "round", "driver"], how="left")
    order = feat2["_di"].values
    seqs_a = seqs[order]
    p_tcn = np.zeros(len(feat), dtype=np.float32); cnt = 0
    for sp in sorted(PROCESSED_DIR.glob("f1_tcn_seed*.pt")):
        m = TC.load(sp); p_tcn += TC.predict_tcn(m, seqs_a); cnt += 1
    p_tcn = (p_tcn / cnt) if cnt else p_tcn

    # Diagnostica: TCN AUC su test
    from sklearn.metrics import roc_auc_score
    tr_m = (feat.season <= train_max).to_numpy()
    va_m = (feat.season == val_y).to_numpy()
    te_m = (feat.season == test_y).to_numpy()
    y = feat["podium"].values
    print(f"TCN saved AUC: train {roc_auc_score(y[tr_m], p_tcn[tr_m]):.3f}  "
          f"val {roc_auc_score(y[va_m], p_tcn[va_m]):.3f}  "
          f"test {roc_auc_score(y[te_m], p_tcn[te_m]):.3f}")

    feat["p_tcn"] = p_tcn
    age_tr = train_max - feat[tr_m]["season"].values
    w = np.exp(-age_tr / 1.5)
    ytr = feat[tr_m]["podium"].values
    yva = feat[va_m]["podium"].values
    te = feat[te_m].copy()
    te["neg_grid"] = -te["grid"]
    print(f"baseline grid: {_precision_at3(te, 'neg_grid'):.3f}\n")

    for cols, tag in [(FEATURE_COLS, "NO TCN"), (FEATURE_COLS + ["p_tcn"], "WITH TCN")]:
        Xtr = feat[tr_m][cols].values
        Xva = feat[va_m][cols].values
        Xte = feat[te_m][cols].values
        ps = []
        for seed in range(42, 52):
            m = CatBoostClassifier(
                iterations=1500, depth=5, learning_rate=0.03,
                boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
                l2_leaf_reg=3.0, min_data_in_leaf=5, loss_function="Logloss",
                random_seed=seed, allow_writing_files=False, verbose=False,
                early_stopping_rounds=80, task_type="CPU",
            )
            m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(m.predict_proba(Xva)[:, 1], yva)
            te["p"] = cal.predict(m.predict_proba(Xte)[:, 1])
            ps.append(_precision_at3(te, "p"))
        ps = np.array(ps)
        print(f"  CatBoost(decay) {tag:10s}: mean {ps.mean():.3f}  std {ps.std():.3f}  min {ps.min():.3f}  max {ps.max():.3f}")


if __name__ == "__main__":
    main()
