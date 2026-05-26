"""
F1 podium model: P(podium) per driver per race. Anti-leakage features (only data
from previous races), time split, calibrated CatBoost. Key metric: precision@3 —
of the three drivers with the highest predicted P(podium) in each race, how many
actually finish on the podium — compared to the "three on the front of the grid"
baseline.

In F1 the car (constructor) and the starting grid dominate, so the model has to
beat them to be worth anything.

Run:
    python -m f1_podium            # download, build features, train, evaluate
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import PROCESSED_DIR, download_all, load_results

# Neutral priors (grid / average position in a ~20-car field)
NEUTRAL_POS = 10.5
PODIUM_PRIOR = 0.15
DNF_PRIOR = 0.12

FEATURE_COLS = [
    "grid", "quali_gap_ms", "teammate_gap_ms",
    "driver_form", "driver_form_pts", "driver_podium_rate", "driver_dnf_rate",
    "constructor_form", "constructor_podium_rate", "driver_track_avg",
    "constructor_track_avg", "constructor_mech_dnf_rate", "round",
    "last1_pos", "last2_pos", "last3_pos",
    "last1_pod", "last2_pod", "last3_pod",
    "last1_dnf", "last2_dnf", "last3_dnf",
    "circuit_is_street", "driver_recovery",
    "constructor_track_type_avg", "driver_track_type_avg",
]

# Street circuits (Ergast circuitId)
STREET_CIRCUITS = {"monaco", "marina_bay", "baku", "jeddah", "miami",
                   "vegas", "albert_park"}


def _lag(dq, i):
    """element i positions back (0 = most recent), NaN if absent."""
    return dq[-1 - i] if len(dq) > i else float("nan")

# words in the 'status' field that indicate a MECHANICAL failure (not crash/disqualification)
_MECH = ("engine", "power unit", "gearbox", "hydraul", "transmission", "electric",
         "turbo", "fuel", "oil", "water", "mechanical", "clutch", "suspension",
         "brakes", "exhaust", "radiator", "driveshaft", "throttle", "overheating",
         "battery", "ers", "mgu", "vibrations", "pneumatic", "wheel")


def _is_mech(status: str) -> bool:
    s = str(status).lower()
    return any(k in s for k in _MECH)


def _avg(dq, default):
    return sum(dq) / len(dq) if dq else default


def build_features(df: pd.DataFrame, wet_map: dict | None = None,
                   pace_map: dict | None = None) -> pd.DataFrame:
    """Iterate races in chronological order; compute pre-race features, then update state.
    wet_map -> 'is_wet' + 'driver_wet_form'. pace_map -> 'driver_race_pace' +
    'driver_tyre_deg' (race pace and tyre degradation rolling from past races,
    anti-leakage)."""
    weather_on = wet_map is not None
    pace_on = pace_map is not None
    drv_wet = defaultdict(lambda: deque(maxlen=8))
    drv_pace = defaultdict(lambda: deque(maxlen=6))
    drv_deg = defaultdict(lambda: deque(maxlen=6))
    drv_lag_pos = defaultdict(lambda: deque(maxlen=3))   # lag features: last 3 races
    drv_lag_pod = defaultdict(lambda: deque(maxlen=3))
    drv_lag_dnf = defaultdict(lambda: deque(maxlen=3))
    drv_recov = defaultdict(lambda: deque(maxlen=10))    # (grid - finishing position)
    con_street = defaultdict(lambda: deque(maxlen=12))   # team on past street circuits
    con_perm = defaultdict(lambda: deque(maxlen=12))     # team on past permanent circuits
    drv_street = defaultdict(lambda: deque(maxlen=10))   # driver on past street circuits
    drv_perm = defaultdict(lambda: deque(maxlen=10))     # driver on past permanent circuits
    drv_pit_min = defaultdict(lambda: deque(maxlen=8))   # min pit-stationary time driver
    con_pit_min = defaultdict(lambda: deque(maxlen=12))  # min pit-stationary time team (crew)
    drv_con = defaultdict(lambda: deque(maxlen=10))      # (driver, constructor) -> positions
    drv_speed = defaultdict(lambda: deque(maxlen=8))     # fastest-lap km/h in past races
    drv_flrank = defaultdict(lambda: deque(maxlen=8))    # fastest-lap rank in past races
    drv_tgap = defaultdict(list)                         # (driver, circuit) -> gap to the winner (ms)
    drv_pos = defaultdict(lambda: deque(maxlen=5))
    drv_pts = defaultdict(lambda: deque(maxlen=5))
    drv_pod = defaultdict(lambda: deque(maxlen=10))
    drv_fin = defaultdict(lambda: deque(maxlen=10))
    con_pos = defaultdict(lambda: deque(maxlen=12))
    con_pod = defaultdict(lambda: deque(maxlen=12))
    con_mech = defaultdict(lambda: deque(maxlen=12))   # team mechanical failures
    track = defaultdict(list)        # (driver, circuit) -> positions
    con_track = defaultdict(list)    # (constructor, circuit) -> positions

    rows = []
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        # 1) read PRE-race state
        for r in g.itertuples(index=False):
            rows.append({
                "season": r.season, "round": r.round, "date": r.date,
                "driver": r.driver, "constructor": r.constructor,
                "grid": r.grid if r.grid > 0 else NEUTRAL_POS,
                "quali_gap_ms": getattr(r, "quali_gap_ms", float("nan")),
                "teammate_gap_ms": getattr(r, "teammate_gap_ms", float("nan")),
                "grid_penalty": getattr(r, "grid_penalty", 0.0),
                "sprint_position": getattr(r, "sprint_position", float("nan")),
                "driver_form": _avg(drv_pos[r.driver], NEUTRAL_POS),
                "driver_form_pts": _avg(drv_pts[r.driver], 0.0),
                "driver_podium_rate": _avg(drv_pod[r.driver], PODIUM_PRIOR),
                "driver_dnf_rate": 1.0 - _avg(drv_fin[r.driver], 1.0 - DNF_PRIOR),
                "constructor_form": _avg(con_pos[r.constructor], NEUTRAL_POS),
                "constructor_podium_rate": _avg(con_pod[r.constructor], PODIUM_PRIOR),
                "driver_track_avg": (np.mean(track[(r.driver, r.circuit)])
                                     if track[(r.driver, r.circuit)] else NEUTRAL_POS),
                "constructor_track_avg": (np.mean(con_track[(r.constructor, r.circuit)])
                                          if con_track[(r.constructor, r.circuit)] else NEUTRAL_POS),
                "constructor_mech_dnf_rate": _avg(con_mech[r.constructor], DNF_PRIOR),
                "last1_pos": _lag(drv_lag_pos[r.driver], 0),
                "last2_pos": _lag(drv_lag_pos[r.driver], 1),
                "last3_pos": _lag(drv_lag_pos[r.driver], 2),
                "last1_pod": _lag(drv_lag_pod[r.driver], 0),
                "last2_pod": _lag(drv_lag_pod[r.driver], 1),
                "last3_pod": _lag(drv_lag_pod[r.driver], 2),
                "last1_dnf": _lag(drv_lag_dnf[r.driver], 0),
                "last2_dnf": _lag(drv_lag_dnf[r.driver], 1),
                "last3_dnf": _lag(drv_lag_dnf[r.driver], 2),
                "circuit_is_street": int(r.circuit in STREET_CIRCUITS),
                "driver_recovery": _avg(drv_recov[r.driver], 0.0),
                "constructor_track_type_avg": _avg(
                    con_street[r.constructor] if r.circuit in STREET_CIRCUITS else con_perm[r.constructor],
                    NEUTRAL_POS),
                "driver_track_type_avg": _avg(
                    drv_street[r.driver] if r.circuit in STREET_CIRCUITS else drv_perm[r.driver],
                    NEUTRAL_POS),
                "age_years": getattr(r, "age_years", float("nan")),
                "home_race": int(getattr(r, "home_race", 0)),
                "champ_pos": getattr(r, "champ_pos", float("nan")),
                "champ_pts": getattr(r, "champ_pts", float("nan")),
                "k_champ_pos": getattr(r, "k_champ_pos", float("nan")),
                "k_champ_pts": getattr(r, "k_champ_pts", float("nan")),
                "circuit_chaos_rate": getattr(r, "circuit_chaos_rate", float("nan")),
                "podium": int(r.podium),
            })
            if weather_on:
                rows[-1]["is_wet"] = int(wet_map.get((r.season, r.round), 0))
                rows[-1]["driver_wet_form"] = _avg(drv_wet[r.driver], NEUTRAL_POS)
            if pace_on:
                rows[-1]["driver_race_pace"] = _avg(drv_pace[r.driver], 1.0)  # ~1s neutral gap
                rows[-1]["driver_tyre_deg"] = _avg(drv_deg[r.driver], 0.05)
        # 2) update state with the actual outcome
        for r in g.itertuples(index=False):
            drv_pos[r.driver].append(r.position)
            drv_pts[r.driver].append(r.points)
            drv_pod[r.driver].append(int(r.podium))
            drv_fin[r.driver].append(int(bool(r.finished)))
            con_pos[r.constructor].append(r.position)
            con_pod[r.constructor].append(int(r.podium))
            con_mech[r.constructor].append(int((not bool(r.finished)) and _is_mech(r.status)))
            drv_lag_pos[r.driver].append(r.position)
            drv_lag_pod[r.driver].append(int(r.podium))
            drv_lag_dnf[r.driver].append(int(not bool(r.finished)))
            pms = getattr(r, "pit_min_s", None)
            if pms is not None and not pd.isna(pms):
                drv_pit_min[r.driver].append(float(pms))
                con_pit_min[r.constructor].append(float(pms))
            if r.grid > 0:
                drv_recov[r.driver].append(int(r.grid) - int(r.position))
            if r.circuit in STREET_CIRCUITS:
                con_street[r.constructor].append(r.position)
                drv_street[r.driver].append(r.position)
            else:
                con_perm[r.constructor].append(r.position)
                drv_perm[r.driver].append(r.position)
            drv_con[(r.driver, r.constructor)].append(r.position)
            fls = getattr(r, "fl_speed_kph", None)
            if fls and not pd.isna(fls):
                drv_speed[r.driver].append(float(fls))
            flr = getattr(r, "fl_rank", None)
            if flr and not pd.isna(flr):
                drv_flrank[r.driver].append(int(flr))
            tg = getattr(r, "time_gap_ms", None)
            if tg is not None and not pd.isna(tg):
                drv_tgap[(r.driver, r.circuit)].append(float(tg))
            track[(r.driver, r.circuit)].append(r.position)
            con_track[(r.constructor, r.circuit)].append(r.position)
            if weather_on and wet_map.get((r.season, r.round), 0):
                drv_wet[r.driver].append(r.position)
            if pace_on:
                pv = pace_map.get((r.season, r.round, r.driver))
                if pv:
                    if pv.get("pace_gap") is not None:
                        drv_pace[r.driver].append(pv["pace_gap"])
                    if pv.get("deg") is not None:
                        drv_deg[r.driver].append(pv["deg"])
    return pd.DataFrame(rows)


def _precision_at3(df_eval: pd.DataFrame, prob_col: str) -> float:
    """Per race: of the 3 with highest predicted probability, how many reach the podium. Averaged over the podium size (3)."""
    hits = tot = 0
    for (_, _), g in df_eval.groupby(["season", "round"]):
        top3 = g.nlargest(3, prob_col)
        hits += top3["podium"].sum()
        tot += 3
    return hits / tot if tot else 0.0


def main():
    import argparse
    from sklearn.model_selection import KFold
    import f1_tcn_core as TC

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=None,
                    help="Anno di test (default: ultima stagione completa >=18 gare)")
    ap.add_argument("--val", type=int, default=None,
                    help="Anno di val (default: test-1)")
    ap.add_argument("--no-download", action="store_true",
                    help="Salta download_all (usa solo i file in cache)")
    args = ap.parse_args()

    if not args.no_download:
        print("Scarico dati F1...")
        download_all()
    else:
        print("Salto download_all (cache only)")
    df = load_results()
    print(f"  {len(df):,} risultati")

    print("Costruisco feature (anti-leakage)...")
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    seasons = sorted(feat["season"].unique())
    if len(seasons) < 4:
        raise RuntimeError("Servono almeno 4 stagioni per lo split.")
    races_per_season = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    if args.test is not None:
        test_y = args.test
        val_y = args.val if args.val is not None else test_y - 1
    else:
        complete = [y for y in seasons if races_per_season.get(y, 0) >= 18]
        if len(complete) < 2:
            raise RuntimeError("Servono >=2 stagioni complete (>=18 gare).")
        test_y = complete[-1]; val_y = complete[-2]
    train_max = val_y - 1
    n_test = races_per_season.get(test_y, 0)
    print(f"  train<={train_max}  val {val_y}  test {test_y}  ({n_test} races in test)")
    if n_test < 10:
        print(f"  WARNING: only {n_test} test races -> precision@3 SE ~±{int(50/n_test**0.5)}pt")

    # The TCN is trained and saved for the predict fallback, but its output is NOT
    # used in the final model: under the time-decay weighting it added nothing and
    # slightly hurt the ranking metric, so it is dropped (see paper).
    N_SEEDS = 5
    print(f"\nTCN multi-seed averaging (N={N_SEEDS})...")
    seqs, y_seq = TC.build_sequences(df)
    assert len(seqs) == len(feat)
    tr_mask = (feat.season <= train_max).to_numpy()
    va_mask = (feat.season == val_y).to_numpy()
    te_mask = (feat.season == test_y).to_numpy()
    tr_idx = np.where(tr_mask)[0]
    p_tcn_acc = np.zeros(len(df), dtype=np.float32)
    for old in PROCESSED_DIR.glob("f1_tcn_seed*.pt"):
        old.unlink()
    for seed_index, seed in enumerate(range(42, 42 + N_SEEDS), 1):
        p_tcn = np.zeros(len(df), dtype=np.float32)
        for train_fold, val_fold in KFold(3, shuffle=True, random_state=seed).split(tr_idx):
            a, b = tr_idx[train_fold], tr_idx[val_fold]
            fold_model = TC.train_tcn(seqs[a], y_seq[a], seqs[b], y_seq[b], seed=seed)
            p_tcn[b] = TC.predict_tcn(fold_model, seqs[b])
        full_model = TC.train_tcn(seqs[tr_mask], y_seq[tr_mask],
                                  seqs[va_mask], y_seq[va_mask], seed=seed)
        p_tcn[va_mask] = TC.predict_tcn(full_model, seqs[va_mask])
        p_tcn[te_mask] = TC.predict_tcn(full_model, seqs[te_mask])
        TC.save(full_model, PROCESSED_DIR / f"f1_tcn_seed{seed_index}.pt")
        p_tcn_acc += p_tcn
        print(f"  seed {seed_index}/{N_SEEDS} done")
    p_tcn = p_tcn_acc / N_SEEDS
    TC.save(full_model, PROCESSED_DIR / "f1_tcn.pt")
    print(f"  TCN(avg{N_SEEDS}) AUC train(OOF) {roc_auc_score(y_seq[tr_mask], p_tcn[tr_mask]):.3f}  "
          f"val {roc_auc_score(y_seq[va_mask], p_tcn[va_mask]):.3f}  "
          f"test {roc_auc_score(y_seq[te_mask], p_tcn[te_mask]):.3f}")

    feat = feat.copy(); feat["p_tcn"] = p_tcn
    train = feat[tr_mask]; val = feat[va_mask]; test = feat[te_mask].copy()

    from catboost import CatBoostClassifier
    cols = FEATURE_COLS  # p_tcn intentionally excluded (see comment above)
    # CatBoost head on CPU: GPU produces different splits on this small dataset
    # (a couple of points), and the head is fast on CPU anyway.
    use_gpu = False
    Xtr, ytr = train[cols].values, train["podium"].values
    Xva, yva = val[cols].values, val["podium"].values
    Xte, yte = test[cols].values, test["podium"].values
    # Ordered boosting limits overfitting on a small dataset; l2_leaf_reg and a
    # Bernoulli subsample add regularisation.
    model = CatBoostClassifier(
        iterations=1500, depth=5, learning_rate=0.03,
        boosting_type="Ordered",
        bootstrap_type="Bernoulli", subsample=0.85,
        l2_leaf_reg=3.0, min_data_in_leaf=5,
        loss_function="Logloss", eval_metric="Logloss",
        random_seed=42, allow_writing_files=False, verbose=False,
        early_stopping_rounds=80,
        task_type="GPU" if use_gpu else "CPU", devices="0" if use_gpu else None,
    )
    # Time-decay sample weights: recent races count more. F1 changes a lot season
    # to season (rules, cars, tyres), so weight each race by exp(-age/tau).
    _tau = 1.5
    _age = train_max - train["season"].values
    _w = np.exp(-_age / _tau)
    model.fit(Xtr, ytr, sample_weight=_w, eval_set=(Xva, yva))
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(model.predict_proba(Xva)[:, 1], yva)

    test["p_podium"] = cal.predict(model.predict_proba(Xte)[:, 1])
    print(f"\n--- RESULTS test {test_y} ---")
    y = test["podium"].values; p = test["p_podium"].values
    print(f"  logloss {log_loss(y, np.clip(p,1e-6,1-1e-6)):.4f}  brier {brier_score_loss(y,p):.4f}  "
          f"AUC {roc_auc_score(y,p):.4f}")
    prec_model = _precision_at3(test, "p_podium")
    test["neg_grid"] = -test["grid"]
    prec_grid = _precision_at3(test, "neg_grid")
    print(f"  precision@3:  model {prec_model:.3f}  vs grid {prec_grid:.3f}  "
          f"delta {(prec_model-prec_grid)*100:+.1f} pt")

    print("\n--- Top features (importance) ---")
    importances = sorted(zip(cols, model.get_feature_importance()), key=lambda x: -x[1])[:10]
    for name, importance in importances:
        print(f"  {name:24s} {importance:.2f}")

    model.save_model(str(PROCESSED_DIR / "f1_podium.cbm"))
    import joblib
    joblib.dump(cal, PROCESSED_DIR / "f1_calibrator.pkl")
    print(f"\nSaved: f1_podium.cbm + f1_calibrator.pkl + f1_tcn.pt")


if __name__ == "__main__":
    main()
