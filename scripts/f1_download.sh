#!/usr/bin/env bash
# Scarica/aggiorna dati F1 (Ergast/jolpica): risultati gara + qualifiche + sprint + standings.
# Usa retry su rate-limit. Riparte da dove era (cache).
# Sovrascrive solo l'ultimo anno (in corso) per dati freschi.
set -euo pipefail
cd "$(dirname "$0")/../f1"
../.venv/bin/python -c "
import f1_data
print('Scarico F1 (stagione corrente in overwrite)...')
years = f1_data.SEASONS
refresh = {years[-1]}  # solo anno corrente
for y in years:
    try:
        f1_data.download_season(y, overwrite=(y in refresh))
        f1_data.download_quali(y, overwrite=(y in refresh))
        f1_data.download_sprint(y, overwrite=(y in refresh))
        f1_data.download_standings(y, overwrite=(y in refresh))
        print(f' {y} ok')
    except Exception as e:
        print(f' {y} fail: {e}')
print('done')
"
