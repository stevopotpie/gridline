"""Per team-game special teams and field position stats from nflverse play-by-play,
turned into pregame form (exponentially weighted, no look-ahead).
Usage: python3 -I special_teams.py DATA_DIR FEATURES_CSV OUT_CSV"""
import sys, glob, os
import numpy as np
import pandas as pd

DATA, FEAT, OUT = sys.argv[1:4]
COLS = ["game_id", "season", "posteam", "defteam", "home_team", "away_team", "yardline_100", "play_type",
        "punt_attempt", "punt_blocked", "kick_distance", "return_yards", "touchback", "return_team",
        "kickoff_attempt", "field_goal_attempt", "field_goal_result", "fixed_drive", "drive_start_yard_line",
        "drive_start_transition", "extra_point_attempt", "season_type"]
frames = []
for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet"))):
    frames.append(pd.read_parquet(f, columns=COLS))
p = pd.concat(frames, ignore_index=True)
print("plays", len(p))

# ---------- Punting (credited to the punting team) ----------
pu = p[(p.punt_attempt == 1) & (p.punt_blocked != 1) & p.kick_distance.notna()].copy()
pu["ret"] = pu.return_yards.fillna(0)
pu["net"] = np.where(pu.touchback == 1, pu.kick_distance - 20, pu.kick_distance - pu.ret)
pu["land"] = np.where(pu.touchback == 1, 20, pu.yardline_100 - pu.kick_distance + pu.ret)  # receiver's own yard line
pu["pin_opp"] = (pu.yardline_100 <= 60).astype(int)                                        # punting from plus territory or near midfield
pu["in10"] = ((pu.land <= 10) & (pu.touchback != 1) & (pu.pin_opp == 1)).astype(int)
punt = pu.groupby(["game_id", "posteam"]).agg(punts=("net", "size"), punt_net_sum=("net", "sum"),
                                              pin_opps=("pin_opp", "sum"), in10=("in10", "sum")).reset_index()
punt = punt.rename(columns={"posteam": "team"})

# ---------- Returns (credited to the returning team) ----------
pr = pu.groupby(["game_id", "defteam"]).agg(pr_n=("ret", "size"), pr_yds=("ret", "sum")).reset_index().rename(columns={"defteam": "team"})

# ---------- Field goals over expected (make probability by distance, fit on all seasons) ----------
fg = p[(p.field_goal_attempt == 1) & p.kick_distance.notna()].copy()
fg["made"] = (fg.field_goal_result == "made").astype(int)
bins = np.arange(15, 70, 3)
fg["dbin"] = pd.cut(fg.kick_distance, bins)
exp_by_bin = fg.groupby("dbin", observed=True).made.mean()
fg["exp"] = fg.dbin.map(exp_by_bin).astype(float).fillna(fg.made.mean())
fg["fg_poe"] = 3 * (fg.made - fg.exp)                     # points over expected
fgt = fg.groupby(["game_id", "posteam"]).agg(fg_poe=("fg_poe", "sum"), fga=("made", "size")).reset_index().rename(columns={"posteam": "team"})

# ---------- Drive starting field position ----------
d = p[p.fixed_drive.notna() & p.posteam.notna() & p.drive_start_yard_line.notna()]
d = d.groupby(["game_id", "fixed_drive"]).first().reset_index()
def own_yard(row_line, team):
    s = str(row_line).split()
    if len(s) == 1:
        return 50.0 if s[0] == "50" or s[0] == "MID" else np.nan
    side, yd = s[0], float(s[1])
    return 50.0 if yd == 50 else (yd if side == team else 100 - yd)
d["start"] = [own_yard(a, b) for a, b in zip(d.drive_start_yard_line, d.posteam)]
d = d.dropna(subset=["start"])
d["after_kick"] = d.drive_start_transition.isin(["KICKOFF"]).astype(int)
d["after_punt"] = d.drive_start_transition.isin(["PUNT"]).astype(int)
off = d.groupby(["game_id", "posteam"]).agg(drives=("start", "size"), start_sum=("start", "sum"),
        ko_n=("after_kick", "sum"), ko_start_sum=("start", lambda s: s[d.loc[s.index, "after_kick"] == 1].sum()),
        pt_n=("after_punt", "sum"), pt_start_sum=("start", lambda s: s[d.loc[s.index, "after_punt"] == 1].sum())).reset_index().rename(columns={"posteam": "team"})
dfn = d.groupby(["game_id", "defteam"]).agg(opp_drives=("start", "size"), opp_start_sum=("start", "sum")).reset_index().rename(columns={"defteam": "team"})

st = off.merge(dfn, on=["game_id", "team"], how="outer").merge(punt, on=["game_id", "team"], how="left") \
        .merge(pr, on=["game_id", "team"], how="left").merge(fgt, on=["game_id", "team"], how="left").fillna(0)

# ---------- Pregame form: exponentially weighted ratios, carried across seasons ----------
g = pd.read_csv(FEAT, parse_dates=["gameday"])
order = g[["game_id", "gameday"]]
st = st.merge(order, on="game_id").sort_values(["gameday", "game_id"])
HALF = 8.0; A = 1 - 0.5 ** (1 / HALF)
NUM = ["start_sum", "opp_start_sum", "ko_start_sum", "pt_start_sum", "punt_net_sum", "in10", "pr_yds", "fg_poe"]
DEN = ["drives", "opp_drives", "ko_n", "pt_n", "punts", "pin_opps", "pr_n", "one"]
st["one"] = 1
PRIOR = {"start_sum": (28.5, 10), "opp_start_sum": (28.5, 10), "ko_start_sum": (26, 4), "pt_start_sum": (30, 4),
         "punt_net_sum": (41, 4), "in10": (0.15, 2), "pr_yds": (8.5, 2), "fg_poe": (0.0, 1)}
state = {}
pre = []
for r in st.itertuples(index=False):
    s = state.setdefault(r.team, {k: [PRIOR[k][0] * PRIOR[k][1], PRIOR[k][1]] for k in NUM})
    pre.append({"game_id": r.game_id, "team": r.team, **{k: s[k][0] / s[k][1] for k in NUM}})
    for k, dk in zip(NUM, DEN):
        s[k][0] = (1 - A) * s[k][0] + getattr(r, k)
        s[k][1] = (1 - A) * s[k][1] + getattr(r, dk)
pre = pd.DataFrame(pre).rename(columns={"start_sum": "fp_off", "opp_start_sum": "fp_def", "ko_start_sum": "ko_ret_start",
        "pt_start_sum": "pr_start", "punt_net_sum": "punt_net", "in10": "punt_in10", "pr_yds": "pr_avg", "fg_poe": "fg_poe_pg"})
# latest form for every team (after its most recent game) for the calculator
latest = {t: {k: v[0] / v[1] for k, v in s.items()} for t, s in state.items()}
pd.DataFrame(latest).T.rename(columns={"start_sum": "fp_off", "opp_start_sum": "fp_def", "ko_start_sum": "ko_ret_start",
        "pt_start_sum": "pr_start", "punt_net_sum": "punt_net", "in10": "punt_in10", "pr_yds": "pr_avg", "fg_poe": "fg_poe_pg"}) \
    .rename_axis("team").reset_index().to_csv(os.path.join(os.path.dirname(OUT), "st_form.csv"), index=False)

for side in ("home", "away"):
    m = g[["game_id", f"{side}_team"]].merge(pre, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"], how="left")
    for c in ["fp_off", "fp_def", "ko_ret_start", "pr_start", "punt_net", "punt_in10", "pr_avg", "fg_poe_pg"]:
        g[f"{side}_{c}"] = m[c].values
g["st_data"] = g.home_fp_off.notna() & g.away_fp_off.notna()
g.to_csv(OUT, index=False)
print("games with special teams form:", int(g.st_data.sum()))
print(pd.DataFrame(latest).T.round(2).sort_values("in10", ascending=False).head(8))
