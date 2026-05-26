# F1 prediction

Two models, both pre-event only:

- **Podium** (`f1_podium.py`): P(podium) per driver per race. CatBoost on 25
  anti-leakage features + time-decay sample weights, isotonic calibration. Metric:
  precision@3 (of the three highest-probability drivers, how many reach the
  podium) vs the qualifying grid.
- **Season champion** (`f1_champ_v2.py`): P(title) per driver given the standings
  after round N. CatBoost scores turned into probabilities with a softmax over the
  drivers + temperature scaling, so they sum to one and are not overconfident.

## Run
```bash
# from the f1/ directory
PYTHONPATH=. ../.venv/bin/python -m f1_podium       # podium model
PYTHONPATH=. ../.venv/bin/python -m f1_champ_v2     # season champion (calibrated)
PYTHONPATH=. ../.venv/bin/python -m f1_predict --year 2026 --round 6   # live race
```

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

## Results (2025 test)
precision@3 0.778 vs grid 0.750. Logistic regression on the same features 0.639.
Best seed on validation reaches 0.833. On the first five unseen 2026 races: 0.600,
never missing a podium completely. Full write-up: `documentation/f1_paper.pdf`.

Trained weights are GitHub release assets, not in the repo.
