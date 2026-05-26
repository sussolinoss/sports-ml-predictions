"""
TCN (Temporal Convolutional Network) + XGBoost: stacking.

Per ogni (pilota, gara) costruisco la sequenza delle ULTIME K=8 gare (pos/podio/
dnf/punti, normalizzati) -> TCN dilated (kernel 3, dil 1/2/4) -> P(podio).
Anti-leakage: per i train rows uso predizioni OOF (3-fold); val/test predette
dal TCN allenato su tutto il train. Aggiungo p_tcn come 23a feature all'XGBoost
finale (FEATURE_COLS + p_tcn) e misuro precision@3.

Aspettativa onesta: marginale (le lag features gia' catturano parte del temporale).
Vale come 8a prova rigorosa per il paper.

Uso:  python -m f1_tcn --test 2025
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_podium as F
from f1_data import load_results

K, CH = 8, 4
SEED = 42
torch.manual_seed(SEED)


def build_sequences(df: pd.DataFrame):
    """Per ogni riga, sequenza (K, CH) delle ultime K gare del pilota (anti-leakage)."""
    state = defaultdict(lambda: deque(maxlen=K))
    seqs = np.zeros((len(df), K, CH), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.float32)
    keys = []
    for i, r in enumerate(df.itertuples(index=False)):
        d = state[r.driver]
        existing = list(d)
        pad = [(0.0, 0.0, 0.0, 0.0)] * (K - len(existing))
        seq = pad + existing  # (K, 4) oldest first
        for t in range(K):
            seqs[i, t] = seq[t]
        y[i] = int(r.podium)
        keys.append((r.season, r.round, r.driver))
        # update DOPO la lettura
        d.append((min(r.position, 20) / 20.0, int(r.podium),
                  int(not bool(r.finished)), min(r.points, 25) / 25.0))
    return seqs, y, keys


class TCN(nn.Module):
    def __init__(self, ci=CH, ch=32):
        super().__init__()
        self.c1 = nn.Conv1d(ci, ch, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(ch, ch, 3, padding=4, dilation=4)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(ch, 1)

    def forward(self, x):           # x: (N, K, F)
        x = x.transpose(1, 2)        # (N, F, K)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = x.mean(dim=2)            # global avg pool sul tempo
        return torch.sigmoid(self.fc(self.drop(x))).squeeze(-1)


def train_tcn(Xtr, ytr, Xva, yva, epochs=60, batch=128, lr=1e-3, patience=6):
    m = TCN()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva)
    best_loss, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            ix = perm[i:i+batch]
            opt.zero_grad()
            p = m(Xtr_t[ix])
            loss = bce(p.clamp(1e-6, 1-1e-6), ytr_t[ix])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = m(Xva_t).clamp(1e-6, 1-1e-6)
            vloss = bce(vp, yva_t).item()
        if vloss < best_loss - 1e-4:
            best_loss = vloss; best_state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(X)).numpy()


def _prec3(d, col):
    h = t = 0
    for _, g in d.groupby(["season", "round"]):
        h += (g.nlargest(3, col)["podium"] == 1).sum()
        t += 3
    return h / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=2025)
    args = ap.parse_args()

    df = load_results()
    feat = F.build_features(df)
    seqs, y_all, _ = build_sequences(df)
    # allinea: feat ha l'ordine cronologico di df (load_results sorta), seqs idem
    assert len(feat) == len(seqs) == len(df), f"misalign {len(feat)} {len(seqs)} {len(df)}"

    tr_mask = (feat.season <= args.test - 2).to_numpy()
    va_mask = (feat.season == args.test - 1).to_numpy()
    te_mask = (feat.season == args.test).to_numpy()
    print(f"train {tr_mask.sum()}  val {va_mask.sum()}  test {te_mask.sum()}")

    # OOF p_tcn sul train (3-fold, no leakage)
    p_tcn = np.zeros(len(df), dtype=np.float32)
    tr_idx = np.where(tr_mask)[0]
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    for fi, (i_a, i_b) in enumerate(kf.split(tr_idx), 1):
        a = tr_idx[i_a]; b = tr_idx[i_b]
        m = train_tcn(seqs[a], y_all[a], seqs[b], y_all[b])
        p_tcn[b] = predict(m, seqs[b])
        print(f"  fold {fi}/3 done")

    # TCN finale su tutto il train -> predice val + test
    m_full = train_tcn(seqs[tr_mask], y_all[tr_mask], seqs[va_mask], y_all[va_mask])
    p_tcn[va_mask] = predict(m_full, seqs[va_mask])
    p_tcn[te_mask] = predict(m_full, seqs[te_mask])

    print(f"\nTCN AUC train(OOF) {roc_auc_score(y_all[tr_mask], p_tcn[tr_mask]):.3f}  "
          f"val {roc_auc_score(y_all[va_mask], p_tcn[va_mask]):.3f}  "
          f"test {roc_auc_score(y_all[te_mask], p_tcn[te_mask]):.3f}")

    feat = feat.copy()
    feat["p_tcn"] = p_tcn
    tr = feat[tr_mask]; va = feat[va_mask]; te = feat[te_mask].copy()

    p = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
         "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
         "min_child_weight": 5, "tree_method": "hist", "seed": SEED}

    def fit_xgb(cols):
        m = xgb.train(p, xgb.DMatrix(tr[cols], label=tr.podium, feature_names=cols),
                      600, evals=[(xgb.DMatrix(va[cols], label=va.podium, feature_names=cols), "v")],
                      early_stopping_rounds=40, verbose_eval=False)
        c = IsotonicRegression(out_of_bounds="clip")
        c.fit(m.predict(xgb.DMatrix(va[cols], feature_names=cols)), va.podium.values)
        return m, c

    BASE = F.FEATURE_COLS
    FULL = BASE + ["p_tcn"]
    m1, c1 = fit_xgb(BASE);  te["p_base"] = c1.predict(m1.predict(xgb.DMatrix(te[BASE], feature_names=BASE)))
    m2, c2 = fit_xgb(FULL);  te["p_full"] = c2.predict(m2.predict(xgb.DMatrix(te[FULL], feature_names=FULL)))
    te["ng"] = -te.grid

    print(f"\nbaseline griglia            precision@3 {_prec3(te,'ng'):.4f}")
    print(f"XGB lag (22 feat) SENZA TCN  precision@3 {_prec3(te,'p_base'):.4f}  "
          f"AUC {roc_auc_score(te.podium, te.p_base):.4f}")
    print(f"XGB lag + p_tcn (23 feat)    precision@3 {_prec3(te,'p_full'):.4f}  "
          f"AUC {roc_auc_score(te.podium, te.p_full):.4f}")
    print(f"\ngain p_tcn nello XGB: {m2.get_score(importance_type='gain').get('p_tcn', 0):.1f}")


if __name__ == "__main__":
    main()
