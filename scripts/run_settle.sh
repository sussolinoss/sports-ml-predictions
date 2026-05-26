#!/usr/bin/env bash
# Settle giornaliero delle bet paper. Chiavi da ../.env.
set -euo pipefail
HERE="$(dirname "$0")"
set -a; [ -f "$HERE/../.env" ] && . "$HERE/../.env"; set +a
cd "$HERE/../tennis"
../.venv/bin/python -m settle_paper
