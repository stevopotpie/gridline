#!/usr/bin/env bash
# Rebuild Gridline from scratch: download nflverse data, compute features, refit, project the season, build both pages.
# Usage: bash run_all.sh WORK_DIR   (WORK_DIR holds these scripts; data goes in WORK_DIR/data)
set -euo pipefail
W="${1:-$(pwd)}"; D="$W/data"; mkdir -p "$D"
pip install -q --break-system-packages pyarrow scikit-learn pandas scipy >/dev/null 2>&1 || true
B=https://github.com/nflverse/nflverse-data/releases/download
curl -sSLf -o "$D/games.csv" "$B/schedules/games.csv"
for y in $(seq 2015 2026); do
  curl -sSLf -o "$D/injuries_$y.csv" "$B/injuries/injuries_$y.csv"
  curl -sSLf -o "$D/dc_$y.csv" "$B/depth_charts/depth_charts_$y.csv"
  curl -sSLf -o "$D/pbp_$y.parquet" "$B/pbp/play_by_play_$y.parquet"
done
curl -sSLf -o "$D/players.csv" "$B/players/players.csv"
cd "$W"
# live sportsbook odds pulled on Steven's PC (github.com/stevopotpie/gridline odds/): patch unplayed games with the books' middle line
if [ -f odds.py ]; then python3 -I -W ignore odds.py "$D" "$W" || echo "odds: step failed, using nflverse lines only"; else echo "odds: odds.py missing, using nflverse lines only"; fi
python3 -I -W ignore features.py "$D" "$W/games_features.csv"
python3 -I -W ignore special_teams.py "$D" "$W/games_features.csv" "$W/games_features_st.csv"
python3 -I -W ignore candidates.py "$D" "$W/games_features_st.csv" "$W/games_features_all.csv"
python3 -I -W ignore improvements.py "$D" "$W/games_features_all.csv" "$W/games_features_plus.csv"
python3 -I -W ignore final_model.py "$W/games_features_plus.csv" "$W" | head -12
python3 -I -W ignore build_pages.py calc "$W"
python3 -I -W ignore season.py "$W" | head -3
python3 -I -W ignore comebacks.py "$D" "$W" | head -1
python3 -I -W ignore comeback_model.py "$W" | head -2
python3 -I -W ignore blowouts.py "$W" | head -3
python3 -I -W ignore volatility.py "$D" "$W" | head -5
python3 -I -W ignore build_pages.py calc "$W"
python3 -I -W ignore player_stats.py "$D" "$W"
python3 -I -W ignore player_props.py "$D" "$W" | head -1
if [ -f player_quarters.py ]; then python3 -I -W ignore player_quarters.py "$D" "$W"; fi
python3 -I -W ignore fourth_quarter.py "$D" "$W" | head -8
python3 -I -W ignore build_pages.py season "$W"
python3 -I -W ignore build_pages.py betlog "$W"
