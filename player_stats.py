"""Gridline players, step 1: per player, per game box scores from nflverse play by play.

Usage: python player_stats.py DATA_DIR OUT_DIR
Writes OUT_DIR/player_games.csv (one row per player per game, regular season and playoffs)
and OUT_DIR/team_games.csv (team totals and opponent context used by the projections).
"""
import sys

import numpy as np
import pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
COLS = ["game_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team", "away_team",
        "play_type", "pass_attempt", "rush_attempt", "sack", "complete_pass", "interception", "two_point_attempt",
        "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name", "passing_yards", "rushing_yards", "receiving_yards",
        "pass_touchdown", "rush_touchdown", "touchdown", "td_player_id", "td_player_name", "td_team",
        "air_yards", "yardline_100"]

frames = []
for y in range(2015, 2027):
    p = pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=COLS)
    frames.append(p[(p.two_point_attempt != 1) & p.posteam.notna()])
pbp = pd.concat(frames, ignore_index=True)
key = ["game_id", "season", "week", "season_type", "game_date"]

# Passing: dropbacks that were real attempts (sacks excluded from attempts)
pa = pbp[(pbp.pass_attempt == 1) & (pbp.sack != 1) & pbp.passer_player_id.notna()]
passing = pa.groupby(key + ["posteam", "defteam", "passer_player_id"]).agg(
    name=("passer_player_name", "last"), pass_att=("pass_attempt", "sum"), completions=("complete_pass", "sum"),
    pass_yds=("passing_yards", "sum"), pass_td=("pass_touchdown", "sum"), ints=("interception", "sum")
).reset_index().rename(columns={"passer_player_id": "player_id"})

ru = pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()]
rushing = ru.groupby(key + ["posteam", "defteam", "rusher_player_id"]).agg(
    name=("rusher_player_name", "last"), carries=("rush_attempt", "sum"), rush_yds=("rushing_yards", "sum"),
    rush_td=("rush_touchdown", "sum"), rz_carries=("yardline_100", lambda s: (s <= 10).sum())
).reset_index().rename(columns={"rusher_player_id": "player_id"})

rc = pa[pa.receiver_player_id.notna()]
receiving = rc.groupby(key + ["posteam", "defteam", "receiver_player_id"]).agg(
    name=("receiver_player_name", "last"), targets=("pass_attempt", "sum"), receptions=("complete_pass", "sum"),
    rec_yds=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"), air_yds=("air_yards", "sum"),
    rz_targets=("yardline_100", lambda s: (s <= 10).sum())
).reset_index().rename(columns={"receiver_player_id": "player_id"})

# Anytime touchdown = the player scored, any way (rushing, receiving, returns, fumble recoveries)
td = pbp[(pbp.touchdown == 1) & pbp.td_player_id.notna()]
tds = td.groupby(["game_id", "td_player_id"]).size().rename("tds").reset_index().rename(columns={"td_player_id": "player_id"})

pg = passing.merge(rushing, on=key + ["posteam", "defteam", "player_id"], how="outer", suffixes=("", "_r"))
pg["name"] = pg["name"].fillna(pg.pop("name_r"))
pg = pg.merge(receiving, on=key + ["posteam", "defteam", "player_id"], how="outer", suffixes=("", "_c"))
pg["name"] = pg["name"].fillna(pg.pop("name_c"))
pg = pg.merge(tds, on=["game_id", "player_id"], how="left")
num = [c for c in pg.columns if c not in key + ["posteam", "defteam", "player_id", "name"]]
pg[num] = pg[num].fillna(0)

players = pd.read_csv(f"{D}/players.csv", usecols=["gsis_id", "display_name", "position", "headshot"], low_memory=False)
pg = pg.merge(players.rename(columns={"gsis_id": "player_id"}), on="player_id", how="left")
pg["display_name"] = pg["display_name"].fillna(pg["name"])
pg["position"] = pg["position"].fillna(pd.Series(np.where(pg.pass_att > 5, "QB", np.where(pg.carries > pg.targets, "RB", "WR")), index=pg.index))
pg = pg.rename(columns={"posteam": "team", "defteam": "opp"})

# Team totals per game: the denominators for shares
tg = pbp.groupby(key + ["posteam", "defteam"]).agg(
    team_dropbacks=("pass_attempt", "sum"), team_carries=("rush_attempt", "sum"),
    team_pass_yds=("passing_yards", "sum"), team_rush_yds=("rushing_yards", "sum"),
).reset_index()
tt = pa.groupby(["game_id", "posteam"]).size().rename("team_targets").reset_index()
tg = tg.merge(tt, on=["game_id", "posteam"], how="left").rename(columns={"posteam": "team", "defteam": "opp"})
g = pd.read_csv(f"{D}/games.csv")[["game_id", "home_team", "away_team", "home_score", "away_score", "spread_line", "total_line"]]
tg = tg.merge(g, on="game_id", how="left")
home = tg.team == tg.home_team
tg["points"] = np.where(home, tg.home_score, tg.away_score)
# Market implied team total: total / 2 plus or minus half the spread (spread_line = home margin)
tg["implied_pts"] = tg.total_line / 2 + np.where(home, 1, -1) * tg.spread_line / 2
tg["is_home"] = home.astype(int)
tg = tg.drop(columns=["home_team", "away_team", "home_score", "away_score"])

pg = pg.sort_values(["game_date", "game_id"]).reset_index(drop=True)
pg.to_csv(f"{OUT}/player_games.csv", index=False)
tg.to_csv(f"{OUT}/team_games.csv", index=False)
print(f"player games: {len(pg):,}  players: {pg.player_id.nunique():,}  team games: {len(tg):,}  through {pg.game_date.max()}")
