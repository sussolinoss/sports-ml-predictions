#!/usr/bin/env bash
# Retrain settimanale tennis (dati freschi Sackmann). Niente chiavi API.
set -euo pipefail
cd "$(dirname "$0")/../tennis"
echo "=== weekly refresh $(date -Is) ==="
../.venv/bin/python run_full_pipeline.py
echo "=== done $(date -Is) ==="
