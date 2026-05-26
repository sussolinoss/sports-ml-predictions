"""
Reusable TCN core (model, sequences, train, save/load).
Imported by f1_podium (training+save), f1_predict (load), f1_tcn (eval).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

K, CH = 8, 4
SEED = 42
import os
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and
                                 os.environ.get("F1_GPU", "1") == "1") else "cpu")


def build_sequences(df: pd.DataFrame):
    """For each row: the (K, CH) sequence of the driver's last K races (anti-leakage).
    Channels: [pos/20, podium, dnf, points/25]."""
    state = defaultdict(lambda: deque(maxlen=K))
    seqs = np.zeros((len(df), K, CH), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        d = state[r.driver]
        existing = list(d)
        pad = [(0.0, 0.0, 0.0, 0.0)] * (K - len(existing))
        seq = pad + existing
        for t in range(K):
            seqs[i, t] = seq[t]
        y[i] = int(r.podium)
        d.append((min(r.position, 20) / 20.0, int(r.podium),
                  int(not bool(r.finished)), min(r.points, 25) / 25.0))
    return seqs, y


class TCN(nn.Module):
    def __init__(self, ci=CH, ch=32):
        super().__init__()
        self.c1 = nn.Conv1d(ci, ch, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(ch, ch, 3, padding=4, dilation=4)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(ch, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = x.mean(dim=2)
        return torch.sigmoid(self.fc(self.drop(x))).squeeze(-1)


def train_tcn(Xtr, ytr, Xva, yva, epochs=60, batch=128, lr=1e-3, patience=6, seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m = TCN().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE); yva_t = torch.from_numpy(yva).to(DEVICE)
    best_loss, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            ix = perm[i:i+batch]
            opt.zero_grad()
            p = m(Xtr_t[ix]).clamp(1e-6, 1-1e-6)
            loss = bce(p, ytr_t[ix])
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
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def predict_tcn(m: TCN, X: np.ndarray) -> np.ndarray:
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(X).to(DEVICE)).cpu().numpy()


def save(m: TCN, path: Path):
    torch.save(m.state_dict(), str(path))


def load(path: Path) -> TCN:
    m = TCN().to(DEVICE)
    m.load_state_dict(torch.load(str(path), map_location=DEVICE, weights_only=True))
    m.eval()
    return m
