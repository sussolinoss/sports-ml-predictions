<h1 align="center">Sports Prediction — Tennis &amp; F1</h1>

<p align="center">
  <em>Two pre-event ML projects, one honest method: predict before the event, calibrate the probability, and never hide what doesn't work.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/models-CatBoost%20%7C%20XGBoost-512BD4.svg" alt="Models">
  <img src="https://img.shields.io/badge/license-CC%20BY%204.0-green.svg" alt="License: CC BY 4.0">
  <a href="https://zenodo.org/records/20402346"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20402346-blue.svg" alt="DOI"></a>
</p>

<p align="center">
  <a href="#whats-inside">What's inside</a> •
  <a href="#results-in-one-glance">Results</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#method">Method</a> •
  <a href="#papers">Papers</a>
</p>

---

I built the tennis model first to learn win prediction, then applied the same way
of working to Formula 1. Both predict **only from data known before the event**.
The output is a probability, not a bet — I never placed real money on it.

The thing I care about most is method, not hype: time-based splits, an explicit
anti-leakage rule, calibrated probabilities, and results reported with their
limits (confidence intervals, ablations, and the parts that failed).

## What's inside

| Project | What it predicts | Model | Headline |
|---------|------------------|-------|----------|
| **`f1/`** | Race podium + season champion | CatBoost + time-decay + calibration | precision@3 **0.778** vs grid 0.750 |
| **`tennis/`** | ATP/WTA match winner | Calibrated XGBoost + surface ELO | **~67%** accuracy, market-level |

```
.
├── f1/          podium model, season-champion model, live predict
│   ├── documentation/   f1_paper (LaTeX + PDF)
│   └── experiments/     research scratch scripts (what was tried + dropped)
├── tennis/      ATP/WTA model, betting backtest, walk-forward validation
├── docs/        paper_tennis, paper_f1 (LaTeX sources)
├── style.md     code style guide
└── requirements.txt
```

## Results in one glance

**F1 — podium (2025 held-out, 24 races)**

| Rule | precision@3 |
|------|:-----------:|
| Recent podium-rate (last 10) | 0.639 |
| Logistic regression (same features) | 0.639 |
| Qualifying grid | 0.750 |
| **This model** (NDCG@3 0.826) | **0.778** |

Bootstrap 95% CI is **[0.708, 0.847]** — the grid sits inside it, so the gain is
*suggestive, not yet significant on one season*. On the first five unseen 2026
races the model scored **0.600** and never missed a podium completely (3/3 once,
2/3 twice, at least 1/3 every time).

**Tennis — match winner**

~67% accuracy, well calibrated, but it matches the closing odds almost exactly:
**no profitable edge** against the market on free data (walk-forward over 8000+
out-of-sample bets, ROI ≈ 0 before costs). The finding — *where there is no
value* — is the point of the tennis paper.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# F1
cd f1
PYTHONPATH=. ../.venv/bin/python -m f1_podium       # podium model
PYTHONPATH=. ../.venv/bin/python -m f1_champ_v2     # season champion

# Tennis
cd ../tennis
python run_full_pipeline.py                          # ATP (default)
```

Trained weights are not in the repo — they ship as assets on the
**[GitHub release v1.0](../../releases/tag/v1.0)** (F1 + tennis models + a small
F1 test dataset). Or just run the scripts to regenerate them from the public API.

## Method

The same pipeline applied to both sports:

- **State-before-event rule** — features are computed from the world *before* the
  event, and the rolling stats are updated only after. Wrong order = the result
  leaks into the features and the accuracy is fake. This is the single most
  important step.
- **Time-based splits** — old seasons train, one recent season validates, the most
  recent is the held-out test. Never a random split on time-series data.
- **Calibration** — isotonic regression (tennis, F1 podium) and softmax +
  temperature (F1 season), so a "70%" really means about 70%.
- **Honest evaluation** — baselines, ablation, bootstrap confidence intervals, and
  a comparison to the bookmaker as a *calibration check*, not a claim of beating
  the market.

## Papers

- **F1**: [`f1/documentation/f1_paper.pdf`](f1/documentation/f1_paper.pdf) — race
  podium + season champion, ablation, what would break the model.
- **Tennis**: [`tennis/documentation/tennis_paper.pdf`](tennis/documentation/tennis_paper.pdf)
  — does the model beat the market? (Spoiler: it matches it.)

## Citation

The paper is archived on Zenodo with a DOI:
[10.5281/zenodo.20402346](https://zenodo.org/records/20402346). Please cite it if
you use this work.

## License

Released under the **Creative Commons Attribution 4.0 (CC BY 4.0)** — see
[`LICENSE`](LICENSE). You can share, adapt and reuse it for any purpose, even
commercially, as long as you give credit.
