"""
Mid-season rolling model for P(wins championship) and P(top-3 standings).
Advanced features: H2H vs leader, momentum, defending_champ, historical mid-season conversion.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR, load_results


def build_champ_dataset(df: pd.DataFrame, min_round: int = 3,
                        target: str = "champ") -> pd.DataFrame:
    """target: 'champ' (top-1) or 'top3' (top-3 final standings)."""
    df = df.sort_values(["season", "round", "driver"]).reset_index(drop=True)
    df["pts_after"] = df.groupby(["season", "driver"])["points"].cumsum()
    df["con_pts_after"] = df.groupby(["season", "constructor"])["points"].cumsum()
    df["pod_after"] = df.groupby(["season", "driver"])["podium"].cumsum()
    df["races_after"] = df.groupby(["season", "driver"]).cumcount() + 1
    df["dnf_event"] = (~df["finished"].astype(bool)).astype(int)
    df["dnf_after"] = df.groupby(["season", "driver"])["dnf_event"].cumsum()

    # total_rounds: max observed round; for the CURRENT (last) season, <18
    # means it is ongoing, so use an estimate of 24 (F1 2024+ calendar).
    total_rounds = df.groupby("season")["round"].max().rename("total_rounds")
    current_season = int(df["season"].max())
    if total_rounds.loc[current_season] < 18:
        total_rounds.loc[current_season] = 24  # estimate for the ongoing calendar
    df = df.merge(total_rounds, on="season")
    last_state = df[df["round"] == df["total_rounds"]] \
        .groupby(["season", "driver"])["pts_after"].max().reset_index()
    # champion (top1) and final top-3 per season
    last_state = last_state.sort_values(["season", "pts_after"], ascending=[True, False])
    champs = last_state.groupby("season").head(1)[["season", "driver"]].assign(is_champ=1)
    top3 = last_state.groupby("season").head(3)[["season", "driver"]].assign(is_top3=1)
    df = df.merge(champs, on=["season", "driver"], how="left") \
           .merge(top3, on=["season", "driver"], how="left")
    df["is_champ"] = df["is_champ"].fillna(0).astype(int)
    df["is_top3"] = df["is_top3"].fillna(0).astype(int)
    # defending champ: driver was champion in the previous season
    champs_prev = champs.assign(season=champs.season + 1, is_def_champ=1) \
                        .drop(columns="is_champ").rename(columns={"season": "season"})
    df = df.merge(champs_prev, on=["season", "driver"], how="left")
    df["is_def_champ"] = df["is_def_champ"].fillna(0).astype(int)

    # H2H pos vs leader: avg(pos_driver - pos_leader_when_both_finish) over the last 5 races
    # momentum_3: points over the last 3 races / 75 (theoretical max 3 races * 25)
    # historical mid-season conversion: given leader after round N, P(wins) — computed on train

    rows = []
    for (season, rnd), g in df.groupby(["season", "round"], sort=True):
        if rnd < min_round:
            continue
        tot = int(g["total_rounds"].iloc[0])
        # current leader pts across all drivers that have entered the season
        sea = df[(df.season == season) & (df["round"] <= rnd)]
        cur_pts = sea.groupby("driver", as_index=False)["pts_after"].max() \
                     .rename(columns={"pts_after": "pts_now"})
        leader = cur_pts.sort_values("pts_now", ascending=False).iloc[0]
        leader_drv = leader["driver"]; leader_pts = leader["pts_now"]

        snap = g[["driver", "constructor", "pts_after", "con_pts_after",
                  "pod_after", "races_after", "dnf_after",
                  "is_champ", "is_top3", "is_def_champ"]].copy()
        snap = snap.rename(columns={"pts_after": "pts_now", "con_pts_after": "con_pts_now",
                                    "pod_after": "pod_now", "races_after": "races_now",
                                    "dnf_after": "dnf_now"})
        snap["gap_to_leader"] = leader_pts - snap["pts_now"]
        snap["gap_frac"] = snap["gap_to_leader"] / max(leader_pts, 1)
        snap["champ_rank"] = snap["pts_now"].rank(method="min", ascending=False)
        snap["pod_rate"] = snap["pod_now"] / snap["races_now"].clip(lower=1)
        snap["dnf_rate"] = snap["dnf_now"] / snap["races_now"].clip(lower=1)
        snap["pts_per_race"] = snap["pts_now"] / snap["races_now"].clip(lower=1)
        snap["frac_round"] = rnd / tot
        snap["rounds_remaining"] = tot - rnd
        snap["max_pts_remaining"] = snap["rounds_remaining"] * 25
        snap["mathematically_possible"] = (snap["pts_now"] + snap["max_pts_remaining"]
                                           >= leader_pts).astype(int)
        # momentum_3: points over the last 3 races (round_N-2..N) per driver
        last3 = df[(df.season == season) & (df["round"] >= rnd - 2)
                   & (df["round"] <= rnd)].groupby("driver")["points"].sum()
        snap["momentum_3"] = snap["driver"].map(last3).fillna(0) / 75.0
        # H2H vs leader: mean (pos_driver - pos_leader) over the last 5 races both finished
        h5 = df[(df.season == season) & (df["round"] >= rnd - 4) & (df["round"] <= rnd)]
        lead_pos = h5[h5["driver"] == leader_drv].set_index("round")["position"]
        h2h_vals = {}
        for d, gd in h5.groupby("driver"):
            merged = gd.set_index("round")["position"].dropna()
            common = merged.index.intersection(lead_pos.index)
            if len(common) >= 1:
                h2h_vals[d] = (merged.loc[common] - lead_pos.loc[common]).mean()
            else:
                h2h_vals[d] = 0.0
        snap["h2h_vs_leader"] = snap["driver"].map(h2h_vals).fillna(0)
        snap["is_leader"] = (snap["driver"] == leader_drv).astype(int)
        snap["season"] = season; snap["round"] = rnd
        rows.append(snap)
    out = pd.concat(rows, ignore_index=True)
    return out


FEAT_COLS = [
    "pts_now", "champ_rank", "gap_to_leader", "gap_frac",
    "pod_now", "pod_rate", "dnf_now", "dnf_rate", "pts_per_race",
    "races_now", "frac_round", "rounds_remaining",
    "con_pts_now", "mathematically_possible",
    "momentum_3", "h2h_vs_leader", "is_leader", "is_def_champ",
]


def train_eval(ds, target_col, tag, predict_year=2026, predict_round=None):
    test_y = 2025; val_y = 2024; train_max = val_y - 1
    tr = ds[ds.season <= train_max]
    va = ds[ds.season == val_y]
    te = ds[ds.season == test_y]
    print(f"\n{'='*60}\nTARGET: {target_col} ({tag})")
    print(f"split: train<={train_max} ({len(tr):,}) val ({len(va):,}) test ({len(te):,})")
    Xtr, ytr = tr[FEAT_COLS].values, tr[target_col].values
    Xva, yva = va[FEAT_COLS].values, va[target_col].values
    Xte, yte = te[FEAT_COLS].values, te[target_col].values
    w = np.exp(-(train_max - tr["season"].values) / 10.0)
    m = CatBoostClassifier(
        iterations=2000, depth=5, learning_rate=0.03,
        boosting_type="Ordered", bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=20, loss_function="Logloss",
        random_seed=42, allow_writing_files=False, verbose=False,
        early_stopping_rounds=80, task_type="CPU",
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(m.predict_proba(Xva)[:, 1], yva)
    p_te = cal.predict(m.predict_proba(Xte)[:, 1])
    print(f"  AUC {roc_auc_score(yte, p_te):.3f}  Brier {brier_score_loss(yte, p_te):.4f}  "
          f"logloss {log_loss(yte, np.clip(p_te,1e-6,1-1e-6)):.4f}")
    te2 = te.copy(); te2["p"] = p_te
    if target_col == "is_champ":
        hits = tot = 0
        for rnd, g in te2.groupby("round"):
            top = g.sort_values("p", ascending=False).head(1)
            hits += int(top["is_champ"].iloc[0]); tot += 1
        print(f"  top-1 per round: {hits}/{tot} = {hits/tot:.3f}")
    else:
        hits = tot = 0
        for rnd, g in te2.groupby("round"):
            top3 = g.sort_values("p", ascending=False).head(3)
            hits += int(top3["is_top3"].sum()); tot += 3
        print(f"  top-3 hit rate per round: {hits}/{tot} = {hits/tot:.3f}")

    # predict target year
    pr_data = ds[ds.season == predict_year]
    if pr_data.empty:
        return
    pr_round = predict_round if predict_round else int(pr_data["round"].max())
    snap = pr_data[pr_data["round"] == pr_round].copy()
    if snap.empty:
        return
    p = cal.predict(m.predict_proba(snap[FEAT_COLS].values)[:, 1])
    snap["p"] = p
    snap["p_norm"] = snap["p"] / snap["p"].sum().clip(min=1e-9)
    print(f"\n  PREDICT {predict_year} dopo round {pr_round}:")
    show = snap.sort_values("p_norm", ascending=False)[
        ["driver", "champ_rank", "pts_now", "gap_to_leader", "p_norm"]].head(8)
    print(show.to_string(index=False, formatters={"p_norm": "{:.1%}".format}))
    return m, cal


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-year", type=int, default=2026)
    ap.add_argument("--predict-round", type=int, default=None)
    args = ap.parse_args()

    df = load_results()
    print(f"dati: {len(df):,} righe, {df.season.nunique()} stagioni {df.season.min()}-{df.season.max()}")
    ds = build_champ_dataset(df)
    print(f"dataset campionato: {len(ds):,} righe")

    m1, c1 = train_eval(ds, "is_champ", "P(vince campionato)",
                        args.predict_year, args.predict_round)
    m2, c2 = train_eval(ds, "is_top3", "P(top-3 standings finali)",
                        args.predict_year, args.predict_round)

    m1.save_model(str(PROCESSED_DIR / "f1_champ.cbm"))
    m2.save_model(str(PROCESSED_DIR / "f1_top3.cbm"))
    import joblib
    joblib.dump(c1, PROCESSED_DIR / "f1_champ_cal.pkl")
    joblib.dump(c2, PROCESSED_DIR / "f1_top3_cal.pkl")
    print(f"\nSalvato: f1_champ.cbm + f1_top3.cbm")


if __name__ == "__main__":
    main()
