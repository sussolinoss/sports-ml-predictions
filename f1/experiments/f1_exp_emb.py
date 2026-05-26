"""
MLP con nn.Embedding(driver) + nn.Embedding(constructor) + tabular features.
Cattura identita' pilota/team oltre alle feature aggregate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1_data import load_results
from f1_podium import FEATURE_COLS, _precision_at3, build_features

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class EmbMLP(nn.Module):
    def __init__(self, n_drv, n_con, n_feat, emb=8, h=64):
        super().__init__()
        self.drv = nn.Embedding(n_drv, emb)
        self.con = nn.Embedding(n_con, emb)
        self.net = nn.Sequential(
            nn.Linear(2 * emb + n_feat, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, d, c, x):
        return self.net(torch.cat([self.drv(d), self.con(c), x], dim=1)).squeeze(-1)


def main():
    df = load_results()
    feat = build_features(df)
    feat = feat.sort_values(["season", "round"]).reset_index(drop=True)
    rps = feat.groupby("season").apply(lambda g: g.groupby("round").ngroups)
    seasons = sorted(feat.season.unique())
    complete = [y for y in seasons if rps.get(y, 0) >= 18]
    test_y, val_y = complete[-1], complete[-2]
    train_max = val_y - 1

    # encode driver/constructor IDs
    drv_ids = {d: i for i, d in enumerate(feat["driver"].unique())}
    con_ids = {c: i for i, c in enumerate(feat["constructor"].unique())}
    feat["d_id"] = feat["driver"].map(drv_ids).astype(int)
    feat["c_id"] = feat["constructor"].map(con_ids).astype(int)
    print(f"drivers {len(drv_ids)}  constructors {len(con_ids)}")

    cols = FEATURE_COLS
    sc = StandardScaler()
    X_all = sc.fit_transform(feat[cols].fillna(feat[cols].median()).values)
    feat_x = X_all

    tr_m = feat.season <= train_max
    va_m = feat.season == val_y
    te_m = feat.season == test_y

    def to_t(m, dtype=torch.float32):
        return torch.tensor(m, dtype=dtype, device=DEV)

    Xtr = to_t(feat_x[tr_m]); Xva = to_t(feat_x[va_m]); Xte = to_t(feat_x[te_m])
    Dtr = to_t(feat.loc[tr_m, "d_id"].values, torch.long)
    Dva = to_t(feat.loc[va_m, "d_id"].values, torch.long)
    Dte = to_t(feat.loc[te_m, "d_id"].values, torch.long)
    Ctr = to_t(feat.loc[tr_m, "c_id"].values, torch.long)
    Cva = to_t(feat.loc[va_m, "c_id"].values, torch.long)
    Cte = to_t(feat.loc[te_m, "c_id"].values, torch.long)
    ytr = to_t(feat.loc[tr_m, "podium"].values)
    yva = to_t(feat.loc[va_m, "podium"].values)

    # multi-seed averaging
    probs_va = np.zeros(yva.shape[0]); probs_te = np.zeros(Xte.shape[0])
    N_SEEDS = 5
    for seed in range(42, 42 + N_SEEDS):
        torch.manual_seed(seed)
        m = EmbMLP(len(drv_ids), len(con_ids), Xtr.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        best_val = 1e9; best_state = None; patience = 0
        for ep in range(120):
            m.train()
            idx = torch.randperm(Xtr.shape[0], device=DEV)
            for i in range(0, len(idx), 512):
                b = idx[i:i+512]
                logits = m(Dtr[b], Ctr[b], Xtr[b])
                loss = loss_fn(logits, ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = loss_fn(m(Dva, Cva, Xva), yva).item()
            if vl < best_val - 1e-4:
                best_val = vl; best_state = {k: v.clone() for k, v in m.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience >= 15: break
        m.load_state_dict(best_state)
        with torch.no_grad():
            probs_va += torch.sigmoid(m(Dva, Cva, Xva)).cpu().numpy()
            probs_te += torch.sigmoid(m(Dte, Cte, Xte)).cpu().numpy()
        print(f"  seed {seed} best_val_loss {best_val:.4f}")
    probs_va /= N_SEEDS; probs_te /= N_SEEDS

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs_va, feat.loc[va_m, "podium"].values)
    p_te = cal.predict(probs_te)
    te = feat[te_m].copy(); te["p_emb"] = p_te
    prec = _precision_at3(te, "p_emb")
    te["neg_grid"] = -te["grid"]
    prec_g = _precision_at3(te, "neg_grid")
    print(f"\nEmbMLP test {test_y}: precision@3 {prec:.3f}  vs grid {prec_g:.3f}  delta {(prec-prec_g)*100:+.1f}pt")


if __name__ == "__main__":
    main()
