#!/usr/bin/env bash
# Bot pre-match giornaliero. Chiavi da ../.env (non committato).
set -euo pipefail
HERE="$(dirname "$0")"
set -a; [ -f "$HERE/../.env" ] && . "$HERE/../.env"; set +a
cd "$HERE/../tennis"
../.venv/bin/python -m prematch_bot --bankroll "${BANKROLL:-100}" --min_edge "${MIN_EDGE:-0.04}"
