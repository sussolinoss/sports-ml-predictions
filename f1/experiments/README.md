# Experiments

Research scratch scripts from building the F1 models. These are **not** part of
the final pipeline and are kept only to show the process: what was tried, what
worked, what was dropped.

Each script tests one idea against the production podium model (precision@3):

| Script | Idea tested | Outcome |
|--------|-------------|---------|
| `f1_exp_decay.py`, `f1_exp_decay_verify.py` | time-decay sample weights, tau sweep | kept (best lever) |
| `f1_exp_bestseed.py` | seed selection on validation | kept |
| `f1_exp_tcn_ablate.py` | TCN with/without decay | TCN dropped |
| `f1_exp_ranker.py`, `f1_exp_xgbrank.py` | LambdaRank / XGBoost ranker | dropped |
| `f1_exp_emb.py` | driver/constructor embedding MLP | dropped |
| `f1_exp_bagging.py`, `f1_exp_avg.py` | multi-seed averaging | dropped (hurts ranking metric) |
| `f1_exp_blend.py`, `f1_exp_stack.py` | blend / meta-LR stacking | dropped |
| `f1_exp_more.py`, `f1_exp_noform.py`, `f1_exp_newfeat.py` | extra/feature ablations | mostly dropped |
| `f1_exp_ablation.py` | feature-group ablation | analysis only |
| `f1_exp_tau_fine.py` | fine tau grid | kept tau in [1.0, 1.5] |

They expect to be run from the `f1/` directory, e.g.
`cd f1 && PYTHONPATH=. ../.venv/bin/python -m experiments.f1_exp_decay`
(some may need small path fixes after the move). They are scratch quality on
purpose; the clean code is in `f1/`.
