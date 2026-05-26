# Sports prediction — Tennis & F1

Two independent ML projects that share the same method: anti-leakage feature
engineering, calibrated models, strict time-based validation. I built the tennis
one first, then applied the same way of working to Formula 1.

Both predict only from data known **before** the event. The output is a
probability, not a bet — I never placed real money on it.

```
.
├── tennis/   ATP/WTA: ELO + serve stats + closing odds, pre-match model,
│             betting backtest, walk-forward validation. See tennis/README.md
├── f1/       F1 podium model + season-champion model. CatBoost, anti-leakage
│             features, precision@3 vs the qualifying grid. See f1/README.md
│   └── experiments/   research scratch scripts (not the final pipeline)
├── docs/     paper_tennis (does the model beat the market?)
├── f1/documentation/   f1_paper (race podium + championship)
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Tennis:
```bash
cd tennis
python run_full_pipeline.py                   # ATP (default)
TENNIS_TOUR=wta python run_full_pipeline.py    # WTA
```

F1:
```bash
cd f1
python -m f1_podium       # download data, train podium model, evaluate vs grid
python -m f1_champ_v2     # season-champion model (softmax + temperature)
```

## Trained models

The trained weights (`.cbm`, `.pkl`, `.pt`) are not in the repo. They are
attached to the GitHub **release** (tag `v1.0`). Download them into the matching
`data/.../processed/` folder, or just re-run the scripts above to regenerate
them from the public data.

## Results in one line

- **Tennis**: ~67% calibrated win predictor, but **no profitable edge** against
  the closing odds on free data (shown on 8000+ out-of-sample bets). The finding —
  *where there is no value* — is the point of the tennis paper.
- **F1**: podium model above the qualifying grid (precision@3 0.778 vs 0.750 on
  the 2025 test, and 0.600 on the first five unseen 2026 races); season-champion
  model calibrated with softmax + temperature. Details in `f1/documentation`.

Both papers are written up honestly, including the parts that did not work.
