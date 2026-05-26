# Tennis prediction

Pre-match ATP/WTA win prediction. The model sees only what was known before the
first ball: rankings, surface, recent form, serve stats, head-to-head, and the
closing odds (also fixed before the match). The output is a probability, never a
bet.

## Pipeline
- `data_loader.py` — load Jeff Sackmann's match data (2005–present).
- `elo_system.py` — general + surface-specific ELO ratings.
- `feature_engineering.py` — ELO, ranking, form, serve stats, fatigue,
  head-to-head, built with the state-before-match rule (no leakage).
- `train_model.py` — calibrated XGBoost (isotonic on validation).
- `meta_model.py` — optional pre-match stacking (XGBoost + ELO).
- `evaluate.py` — time-based forward test.
- `odds_loader.py` + `backtest.py` — closing odds from tennis-data.co.uk and a
  betting ROI backtest (evaluation only).
- `walkforward.py` — walk-forward + bootstrap honesty check.

## Run
```bash
# from the tennis/ directory
python run_full_pipeline.py                   # ATP (default)
TENNIS_TOUR=wta python run_full_pipeline.py    # WTA
python -m backtest --start 2025-01-01 --end 2025-12-31
```

## Result
~67% accuracy on the held-out test, well calibrated, but the accuracy and the
calibration match the closing odds almost exactly: **no profitable edge** against
the market on free public data (walk-forward over 8000+ out-of-sample bets,
ROI ≈ 0 before costs, negative after). The paper documents this honestly.

Credentials for the optional live/betting modules (Betfair, The Odds API) are read
from environment variables, never hardcoded. Nothing runs against real money.
