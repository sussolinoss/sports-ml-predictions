"""Aggiungo 3 feat creative: team_change, age*log(exp), champ_pos_momentum (3-gare delta).
Test multi-seed FULL vs FULL+nuove (decay tau=1.5, CPU)."""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features


def add_new(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Compute team_change, age_x_exp, champ_momentum, anti-leakage."""
    drv_last_team = {}
    drv_races = defaultdict(int)
    drv_cpos = defaultdict(lambda: deque(maxlen=4))
    extras = []
    df_sorted = df.sort_values(["season", "round"]).reset_index(drop=True)
    for r in df_sorted.itertuples(index=False):
        last_t = drv_last_team.get(r.driver)
        team_chg = 0 if last_t is None else int(last_t != r.constructor)
        nr = drv_races[r.driver]
        ax = (r.age_years if not pd.isna(getattr(r, "age_years", np.nan)) else 27.0) * np.log1p(nr)
        cp_now = getattr(r, "champ_pos", np.nan)
        cps = list(drv_cpos[r.driver])
        if len(cps) >= 3 and not pd.isna(cp_now):
            cmom = cps[0] - cp_now  # cps[0] = piu' vecchia (3 gare fa), positivo = sale in classifica
        else:
            cmom = 0.0
        extras.append({"season": r.season, "round": r.round, "driver": r.driver,
                       "team_change": team_chg, "age_x_exp": ax, "champ_momentum": cmom})
        # update state DOPO read
        drv_last_team[r.driver] = r.constructor
        drv_races[r.driver] += 1
        if not pd.isna(cp_now):
            drv_cpos[r.driver].append(float(cp_now))
    ex = pd.DataFrame(extras)
    return feat.merge(ex, on=["season", "round", "driver"], how="left")


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    feat = add_new(df, feat)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    complete = [y for y in sorted(feat.season.unique()) if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1
    tr = feat[feat.season <= train_max]
    va = feat[feat.season == val_y]
    te = feat[feat.season == test_y].copy()
    te["neg_grid"] = -te["grid"]
    w = np.exp(-(train_max - tr["season"].values) / 1.5)

    NEW = ["team_change", "age_x_exp", "champ_momentum"]
    # diagnose new feat coverage in test
    print(f"new feat coverage test: team_change_nan {te['team_change'].isna().mean():.2f}  "
          f"age_x_exp_nan {te['age_x_exp'].isna().mean():.2f}  "
          f"champ_momentum_nonzero {(te['champ_momentum']!=0).mean():.2f}")

    for cols, tag in [(FEATURE_COLS, "FULL 25"),
                      (FEATURE_COLS + NEW, "FULL+new 28"),
                      (FEATURE_COLS + ["team_change"], "+team_change"),
                      (FEATURE_COLS + ["age_x_exp"], "+age_x_exp"),
                      (FEATURE_COLS + ["champ_momentum"], "+champ_momentum")]:
        Xtr = tr[cols].values; Xva = va[cols].values; Xte = te[cols].values
        ytr, yva = tr["podium"].values, va["podium"].values
        rows = []
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
            p_va = cal.fit_transform(m.predict_proba(Xva)[:, 1], yva)
            p_te = cal.predict(m.predict_proba(Xte)[:, 1])
            from sklearn.metrics import log_loss
            ll = log_loss(yva, np.clip(p_va, 1e-6, 1-1e-6))
            va_c = va.copy(); va_c["p"] = p_va
            te_c = te.copy(); te_c["p"] = p_te
            rows.append({"seed": seed, "val_ll": ll,
                         "val_p": _precision_at3(va_c, "p"),
                         "test_p": _precision_at3(te_c, "p")})
        arr = np.array([(r["val_ll"], r["val_p"], r["test_p"]) for r in rows])
        best_pv = max(rows, key=lambda x: (x["val_p"], -x["val_ll"]))
        print(f"  {tag:20s} mean {arr[:,2].mean():.3f}±{arr[:,2].std():.3f}  "
              f"max {arr[:,2].max():.3f}  best-on-val→test {best_pv['test_p']:.3f}")


if __name__ == "__main__":
    main()
