"""
Mid-season rolling model for P(wins constructors championship), softmax+T calibrated.
Ergast dataset 1958+ (start of the constructors cup).
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
from f1_champ_v2 import ece, fit_temperature, softmax_per_group
from f1_data import PROCESSED_DIR, load_results


def build_con_dataset(df: pd.DataFrame, min_round: int = 3) -> pd.DataFrame:
    """For each (season, round_N >= min_round, active constructor): features + champion label."""
    df = df.sort_values(["season", "round", "driver"]).reset_index(drop=True)
    # points per (season, round, constructor) = sum of the 2 drivers' points
    sea_con = df.groupby(["season", "round", "constructor"], as_index=False).agg(
        race_pts=("points", "sum"),
        race_pod=("podium", "sum"),
        race_dnf=("finished", lambda x: int((~x.astype(bool)).sum())),
        race_winners=("position", lambda x: int((x == 1).any())),
    )
    sea_con = sea_con.sort_values(["season", "constructor", "round"])
    sea_con["pts_after"] = sea_con.groupby(["season", "constructor"])["race_pts"].cumsum()
    sea_con["pod_after"] = sea_con.groupby(["season", "constructor"])["race_pod"].cumsum()
    sea_con["dnf_after"] = sea_con.groupby(["season", "constructor"])["race_dnf"].cumsum()
    sea_con["win_after"] = sea_con.groupby(["season", "constructor"])["race_winners"].cumsum()
    sea_con["races_after"] = sea_con.groupby(["season", "constructor"]).cumcount() + 1

    total_rounds = df.groupby("season")["round"].max().rename("total_rounds")
    cur_season = int(df["season"].max())
    if total_rounds.loc[cur_season] < 18:
        total_rounds.loc[cur_season] = 24  # estimate for the ongoing calendar
    sea_con = sea_con.merge(total_rounds, on="season")

    # constructors champion per season = max pts_after at season end
    last = sea_con[sea_con["round"] == sea_con["total_rounds"]] \
        .groupby(["season", "constructor"])["pts_after"].max().reset_index()
    champs = last.sort_values(["season", "pts_after"], ascending=[True, False]) \
                 .groupby("season").head(1)[["season", "constructor"]] \
                 .assign(is_champ=1)
    sea_con = sea_con.merge(champs, on=["season", "constructor"], how="left")
    sea_con["is_champ"] = sea_con["is_champ"].fillna(0).astype(int)

    # defending champ
    cp_prev = champs.assign(season=champs.season + 1, is_def_champ=1) \
                    .drop(columns="is_champ")
    sea_con = sea_con.merge(cp_prev, on=["season", "constructor"], how="left")
    sea_con["is_def_champ"] = sea_con["is_def_champ"].fillna(0).astype(int)

    rows = []
    for (season, rnd), g in sea_con.groupby(["season", "round"], sort=True):
        if rnd < min_round:
            continue
        tot = int(g["total_rounds"].iloc[0])
        # leader pts: max pts_after among constructors in the season up to rnd
        sea = sea_con[(sea_con.season == season) & (sea_con["round"] <= rnd)]
        cur = sea.groupby("constructor", as_index=False)["pts_after"].max()
        leader_pts = cur["pts_after"].max()

        snap = g.copy()
        snap["pts_now"] = snap["pts_after"]
        snap["pod_now"] = snap["pod_after"]
        snap["dnf_now"] = snap["dnf_after"]
        snap["win_now"] = snap["win_after"]
        snap["races_now"] = snap["races_after"]
        snap["gap_to_leader"] = leader_pts - snap["pts_now"]
        snap["gap_frac"] = snap["gap_to_leader"] / max(leader_pts, 1)
        snap["champ_rank"] = snap["pts_now"].rank(method="min", ascending=False)
        snap["pod_rate"] = snap["pod_now"] / (snap["races_now"] * 2).clip(lower=1)
        snap["dnf_rate"] = snap["dnf_now"] / (snap["races_now"] * 2).clip(lower=1)
        snap["pts_per_race"] = snap["pts_now"] / snap["races_now"].clip(lower=1)
        snap["frac_round"] = rnd / tot
        snap["rounds_remaining"] = tot - rnd
        snap["max_pts_remaining"] = snap["rounds_remaining"] * 43  # max 25+18 per race
        snap["math_possible"] = (snap["pts_now"] + snap["max_pts_remaining"] >= leader_pts).astype(int)
        # momentum_3: pts over the last 3 races per constructor
        last3 = sea_con[(sea_con.season == season) & (sea_con["round"] >= rnd - 2)
                        & (sea_con["round"] <= rnd)].groupby("constructor")["race_pts"].sum()
        snap["momentum_3"] = snap["constructor"].map(last3).fillna(0) / 129.0  # max 3*43
        snap["is_leader"] = (snap["pts_now"] == leader_pts).astype(int)
        rows.append(snap)
    return pd.concat(rows, ignore_index=True)


FEAT = ["pts_now", "champ_rank", "gap_to_leader", "gap_frac",
        "pod_now", "pod_rate", "dnf_now", "dnf_rate", "win_now", "pts_per_race",
        "races_now", "frac_round", "rounds_remaining",
        "math_possible", "momentum_3", "is_leader", "is_def_champ"]


def main():
    df = load_results()
    ds = build_con_dataset(df)
    print(f"dataset costruttori: {len(ds):,} righe, "
          f"{ds.season.nunique()} stagioni {ds.season.min()}-{ds.season.max()}, "
          f"{ds.constructor.nunique()} team")

    test_y = 2025; train_max = 2019  # val pool 2020-2024
    tr = ds[ds.season <= train_max]
    va = ds[(ds.season > train_max) & (ds.season < test_y)]
    te = ds[ds.season == test_y]
    print(f"split: train ({len(tr):,}) val ({len(va):,}) test ({len(te):,})")
    Xtr, ytr = tr[FEAT].values, tr["is_champ"].values
    Xva, yva = va[FEAT].values, va["is_champ"].values
    Xte, yte = te[FEAT].values, te["is_champ"].values
    w = np.exp(-(train_max - tr["season"].values) / 10.0)

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
    g_va = (va["season"].astype(str) + "_" + va["round"].astype(str)).values
    g_te = (te["season"].astype(str) + "_" + te["round"].astype(str)).values

    T_opt, _ = fit_temperature(logit_va, yva, g_va)
    p_te = softmax_per_group(logit_te, g_te, T=T_opt)
    print(f"\nT_opt = {T_opt:.3f}")
    print(f"Test 2025: Brier {brier_score_loss(yte, p_te):.4f}  "
          f"ECE {ece(p_te, yte):.4f}  AUC {roc_auc_score(yte, p_te):.3f}")

    # top-1 per round
    te2 = te.copy(); te2["p"] = p_te
    hits = tot = 0
    for rnd, g in te2.groupby("round"):
        top = g.sort_values("p", ascending=False).head(1)
        hits += int(top["is_champ"].iloc[0]); tot += 1
    print(f"Top-1 per round: {hits}/{tot} = {hits/tot:.3f}")

    m.save_model(str(PROCESSED_DIR / "f1_constructor.cbm"))
    import joblib
    joblib.dump({"T": T_opt}, PROCESSED_DIR / "f1_constructor_T.pkl")

    # PREDICT 2026
    cur_season = int(ds.season.max())
    cur_round = int(ds[ds.season == cur_season]["round"].max())
    snap = ds[(ds.season == cur_season) & (ds["round"] == cur_round)].copy()
    logit = m.predict(snap[FEAT].values, prediction_type="RawFormulaVal")
    p = softmax_per_group(logit, np.zeros(len(snap), dtype=int), T=T_opt)
    snap["p_champ"] = p
    snap = snap.sort_values("p_champ", ascending=False)
    print(f"\n=== PREDICT costruttori {cur_season} dopo round {cur_round} (T={T_opt:.2f}) ===")
    print(snap[["constructor", "champ_rank", "pts_now", "gap_to_leader", "p_champ"]]
          .head(10).to_string(index=False, formatters={"p_champ": "{:.1%}".format}))
    print(f"  sum P = {snap['p_champ'].sum():.3f}")


if __name__ == "__main__":
    main()
