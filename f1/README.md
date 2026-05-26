<h1 align="center">F1 Prediction</h1>

<p align="center">
  Pre-event Formula 1 models for race podiums and the season champion.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/status-research-orange.svg" alt="Status">
</p>

---

## Overview

Two models, both using only information available before the event:

- **Podium** (`f1_podium.py`): P(podium) per driver per race. CatBoost on 25
  anti-leakage features with time-decay sample weights and isotonic calibration.
  The headline metric is precision@3 (of the three highest-probability drivers,
  how many actually reach the podium), compared against the qualifying grid.
- **Season champion** (`f1_champ_v2.py`): P(title) per driver given the standings
  after round N. CatBoost scores are turned into probabilities with a softmax over
  the drivers plus temperature scaling, so they sum to one and are not overconfident.

## 🏁 Run

```bash
# from the f1/ directory
PYTHONPATH=. ../.venv/bin/python -m f1_podium       # podium model
PYTHONPATH=. ../.venv/bin/python -m f1_champ_v2     # season champion (calibrated)
PYTHONPATH=. ../.venv/bin/python -m f1_predict --year 2026 --round 6   # live race
```

## 📊 Results (2025 test)

| Metric | Value |
|--------|-------|
| Precision@3 (podium model) | **0.778** |
| Precision@3 (qualifying grid baseline) | 0.750 |
| Precision@3 (logistic regression, same features) | 0.639 |
| NDCG@3 | 0.826 |
| Bootstrap 95% CI (precision@3) | [0.708, 0.847] |
| Best seed on validation | 0.833 |
| 2026 held-out (first five races) | 0.600 |

On the first five unseen 2026 races the podium model scored 0.600 and never missed
a podium completely. Full write-up: [`documentation/f1_paper.pdf`](documentation/f1_paper.pdf).

> Note: this is a research project, not a tipping service. The numbers above are the
> only ones I have measured; I have not tuned them to look better than they are.

## Data

Results come from the public Ergast / Jolpica F1 API (1950–present), downloaded by
`f1_data.py`. Nothing is redistributed here; the scripts fetch it on first run.

## Key files

| File | Role |
|------|------|
| `f1_data.py` | download + load results, build raw table |
| `f1_podium.py` | podium model (production) |
| `f1_champ.py` / `f1_champ_v2.py` | season-champion dataset + calibrated model |
| `f1_predict.py` | predict an upcoming race |
| `f1_multiclass.py` | winner/points/no-points benchmark |
| `f1_tcn_core.py` | TCN (tried, dropped from the final model — see paper) |
| `experiments/` | research scratch scripts |

## Trained weights

The trained weights are not in this repo. They ship as assets on the
**GitHub release v1.0**.
