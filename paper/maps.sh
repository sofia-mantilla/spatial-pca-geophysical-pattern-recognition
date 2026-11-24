#!/bin/zsh
set -u
cd "$(dirname "$0")"
SP="$1"
echo "=== lithology, both cases ==="
../.venv/bin/python -u make_fig_litho_combined.py --outdir $SP/geomaps --name litho_both --colour lithology > $SP/logs/map_litho.log 2>&1
echo "litho exit $?"
echo "=== age/era, both cases ==="
../.venv/bin/python -u make_fig_litho_combined.py --outdir $SP/geomaps --name age_both --colour age > $SP/logs/map_age.log 2>&1
echo "age exit $?"
echo "=== with structures ==="
../.venv/bin/python -u make_fig_litho_combined.py --outdir $SP/geomaps --name litho_struct --colour lithology --structures > $SP/logs/map_struct.log 2>&1
echo "struct exit $?"
echo MAPSDONE
