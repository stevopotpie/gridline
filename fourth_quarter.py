"""Fourth quarter scoring vs the point differential entering the fourth quarter.

League view (2015 to now): how many points teams score and allow in the 4th quarter, by margin after three quarters.
Team view (2026): each team's 4th quarter points for and against, overall and split by leading, close or trailing,
plus how each team's 4th quarter compares with what the league does from the same margin.
Usage: python fourth_quarter.py DATA_DIR WORK_DIR
"""
import json
import sys

import numpy as np
import pandas as pd

D, W = sys.argv[1], sys.argv[2]
frames = []
for y in range(2015, 2027):
    p = pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=["game_id", "season", "season_type", "home_team", "away_team",
                                                           "qtr", "total_home_score", "total_away_score", "play_id"])
    frames.append(p)
pbp = pd.concat(frames).sort_values(["game_id", "play_id"])
end3 = pbp[pbp.qtr <= 3].groupby("game_id").tail(1).set_index("game_id")
end4 = pbp[pbp.qtr <= 4].groupby("game_id").tail(1).set_index("game_id")
gm = end4[["season", "season_type", "home_team", "away_team"]].join(
    end3[["total_home_score", "total_away_score"]].rename(columns=lambda c: c + "_3")).join(
    end4[["total_home_score", "total_away_score"]].rename(columns=lambda c: c + "_4")).dropna()
rows = []
for side, opp in (("home", "away"), ("away", "home")):
    rows.append(pd.DataFrame({
        "game_id": gm.index, "season": gm.season.values, "season_type": gm.season_type.values,
        "team": gm[f"{side}_team"].values, "opp": gm[f"{opp}_team"].values,
        "diff_q3": (gm[f"total_{side}_score_3"] - gm[f"total_{opp}_score_3"]).values,
        "q4_for": (gm[f"total_{side}_score_4"] - gm[f"total_{side}_score_3"]).values,
        "q4_against": (gm[f"total_{opp}_score_4"] - gm[f"total_{opp}_score_3"]).values}))
q = pd.concat(rows, ignore_index=True)
q["q4_net"] = q.q4_for - q.q4_against
BINS = [-100, -15, -9, -1, 0, 8, 14, 100]
LABELS = ["Down 15+", "Down 9 to 14", "Down 1 to 8", "Tied", "Up 1 to 8", "Up 9 to 14", "Up 15+"]
q["margin"] = pd.cut(q.diff_q3, BINS, labels=LABELS)

league = q.groupby("margin", observed=True).agg(games=("game_id", "size"), q4_for=("q4_for", "mean"),
                                                 q4_against=("q4_against", "mean"), q4_net=("q4_net", "mean"),
                                                 scored_any=("q4_for", lambda s: (s > 0).mean())).reset_index()
# Expected 4th quarter for a team given its margin, from the league table
exp = q.groupby("margin", observed=True)[["q4_for", "q4_against"]].mean()
q["exp_for"] = q.margin.map(exp.q4_for).astype(float)
q["exp_against"] = q.margin.map(exp.q4_against).astype(float)

t = q[(q.season == 2026) & (q.season_type == "REG")].copy()
t["state"] = np.select([t.diff_q3 >= 9, t.diff_q3 <= -9], ["leading", "trailing"], "close")
team = t.groupby("team").agg(games=("game_id", "size"), q4_for=("q4_for", "mean"), q4_against=("q4_against", "mean"),
                             q4_net=("q4_net", "mean"), vs_expected=("q4_net", "sum"), avg_diff_q3=("diff_q3", "mean"))
team["vs_expected"] = (t.groupby("team").q4_for.sum() - t.groupby("team").exp_for.sum()
                       - (t.groupby("team").q4_against.sum() - t.groupby("team").exp_against.sum())) / team.games
for st in ["leading", "close", "trailing"]:
    s = t[t.state == st].groupby("team")
    team[f"{st}_g"] = s.size()
    team[f"{st}_net"] = s.q4_net.mean()
team = team.reset_index().fillna({"leading_g": 0, "close_g": 0, "trailing_g": 0})
games_list = t.sort_values("game_id")[["team", "game_id", "opp", "diff_q3", "q4_for", "q4_against"]]

league.to_csv(f"{W}/q4_league.csv", index=False)
team.to_csv(f"{W}/q4_team_2026.csv", index=False)
json.dump({"league": league.round(3).to_dict("records"),
           "team": team.round(2).to_dict("records"),
           "games": games_list.to_dict("records")}, open(f"{W}/q4_data.json", "w"))
print(league.round(2).to_string(index=False))
print(team.sort_values("vs_expected").round(2)[["team", "games", "q4_for", "q4_against", "q4_net", "vs_expected"]].to_string(index=False))
