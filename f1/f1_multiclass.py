"""
F1 multi-class winner/podium/points/no-points + macro-F1.
Confronto diretto con Pollub JCSI 2024 (macro-F1 0.778, acc 0.802).
Stessa pipeline f1_podium: CatBoost decay + best-seed-on-val + temporal split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, build_features


def label_class(pos, n=4):
    """4-class: 0=winner(P1), 1=podium(P2-3), 2=points(P4-10), 3=no-points(11+/DNF).
    3-class (n=3): 0=winner, 1=points(P2-10), 2=no-points."""
    if pd.isna(pos) or pos > 90:
        return n - 1
    pos = int(pos)
    if n == 4:
        if pos == 1: return 0
        if pos <= 3: return 1
        if pos <= 10: return 2
        return 3
    else:  # 3-class
        if pos == 1: return 0
        if pos <= 10: return 1
        return 2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=4, choices=[3, 4])
    ap.add_argument("--balanced", action="store_true",
                    help="class_weights bilanciati (boost winner rara)")
    args = ap.parse_args()
    NC = args.classes

    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    feat = feat.merge(df[["season", "round", "driver", "position"]],
                      on=["season", "round", "driver"], how="left")
    feat["cls"] = feat["position"].apply(lambda p: label_class(p, NC))

    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    print(f"classi: {NC}  train<={train_max} val {val_y} test {test_y}")

    cols = FEATURE_COLS
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y]
    Xtr, ytr = tr[cols].values, tr["cls"].values
    Xva, yva = va[cols].values, va["cls"].values
    Xte, yte = te[cols].values, te["cls"].values
    w = np.exp(-(train_max - tr["season"].values) / 1.5)

    # class weights bilanciati (inverso frequenza) per boost winner rara
    cw = None
    if args.balanced:
        counts = np.bincount(ytr, minlength=NC)
        cw = (len(ytr) / (NC * counts)).tolist()
        print(f"class_weights: {[round(x,2) for x in cw]}")

    # best-seed-on-val (val macro-F1)
    best = None
    for seed in range(42, 52):
        m = CatBoostClassifier(
            iterations=1500, depth=5, learning_rate=0.03,
            boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
            l2_leaf_reg=3.0, min_data_in_leaf=10,
            loss_function="MultiClass", classes_count=NC,
            class_weights=cw,
            random_seed=seed, allow_writing_files=False, verbose=False,
            early_stopping_rounds=80, task_type="CPU",
        )
        m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
        pred_va = m.predict(Xva).flatten().astype(int)
        f1_va = f1_score(yva, pred_va, average="macro")
        if best is None or f1_va > best["f1_va"]:
            best = {"seed": seed, "f1_va": f1_va, "model": m}

    m = best["model"]
    pred_te = m.predict(Xte).flatten().astype(int)
    macro_f1 = f1_score(yte, pred_te, average="macro")
    acc = accuracy_score(yte, pred_te)
    print(f"\nBest seed {best['seed']} (val macro-F1 {best['f1_va']:.4f})")
    print(f"\n=== TEST {test_y} ({NC}-class) ===")
    print(f"  macro-F1: {macro_f1:.4f}")
    print(f"  accuracy: {acc:.4f}")
    names = (["winner", "podium", "points", "no-points"] if NC == 4
             else ["winner", "points", "no-points"])
    print("\n" + classification_report(yte, pred_te, target_names=names,
                                        digits=3, zero_division=0))
    # specificity per classe (metrica Pollub): TN/(TN+FP)
    cm = confusion_matrix(yte, pred_te, labels=list(range(NC)))
    tot = cm.sum()
    print("specificity per classe:")
    specs = []
    for k in range(NC):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp; tn = tot - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) else 0
        specs.append(spec)
        print(f"  {names[k]:12s} {spec:.3f}")
    print(f"  macro specificity: {np.mean(specs):.3f}")
    print(f"--- CONFRONTO Pollub JCSI 2024 ---")
    print(f"  Pollub macro-F1: 0.7783  accuracy: 0.8022")
    print(f"  Tuo    macro-F1: {macro_f1:.4f}  accuracy: {acc:.4f}")
    d_f1 = (macro_f1 - 0.7783) * 100
    d_acc = (acc - 0.8022) * 100
    print(f"  delta  macro-F1: {d_f1:+.1f}pt   accuracy: {d_acc:+.1f}pt")


if __name__ == "__main__":
    main()
