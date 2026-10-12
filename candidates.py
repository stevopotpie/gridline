"""Candidate predictive factors from play-by-play, turned into pregame team form (no look-ahead).
Usage: python3 -I candidates.py DATA_DIR FEATURES_ST_CSV OUT_CSV"""
import sys, glob, os
import numpy as np
import pandas as pd

DATA, FEAT, OUT = sys.argv[1:4]
COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "epa", "success", "qb_dropback", "rush", "pass",
        "sack", "interception", "fumble_lost", "yards_gained", "pass_oe", "qb_epa", "passer_player_id", "wp", "qtr",
        "game_seconds_remaining", "fixed_drive", "fixed_drive_result", "drive_inside20", "penalty", "penalty_team",
        "penalty_yards", "qb_hit"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet")))], ignore_index=True)
p = p[p.posteam.notna()]

plays = p[p.play_type.isin(["pass", "run"]) & p.epa.notna()].copy()
plays["db"] = (plays.qb_dropback == 1).astype(int)
plays["ru"] = 1 - plays.db
plays["pass_epa"] = plays.epa * plays.db
plays["rush_epa"] = plays.epa * plays.ru
plays["expl"] = (((plays.db == 1) & (plays.yards_gained >= 20)) | ((plays.ru == 1) & (plays.yards_gained >= 10))).astype(int)
plays["sk"] = plays.sack.fillna(0)
plays["hit"] = plays.qb_hit.fillna(0)
plays["to"] = plays.interception.fillna(0) + plays.fumble_lost.fillna(0)
neutral = plays.wp.between(0.2, 0.8) & (plays.qtr <= 3)
plays["proe_n"] = neutral.astype(int) * plays.pass_oe.notna()
plays["proe_sum"] = np.where(plays.proe_n == 1, plays.pass_oe.fillna(0), 0)
# neutral pace: seconds until the offense's next snap in the same drive
plays = plays.sort_values(["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False])
nxt = plays.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
gap = plays.game_seconds_remaining - nxt
ok = neutral & gap.between(5, 60)
plays["pace_n"] = ok.astype(int)
plays["pace_sum"] = np.where(ok, gap, 0)

agg = dict(n=("epa", "size"), db=("db", "sum"), ru=("ru", "sum"), pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum"),
           succ=("success", "sum"), expl=("expl", "sum"), sk=("sk", "sum"), hit=("hit", "sum"), to=("to", "sum"),
           proe_sum=("proe_sum", "sum"), proe_n=("proe_n", "sum"), pace_sum=("pace_sum", "sum"), pace_n=("pace_n", "sum"))
off = plays.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
dfn = plays.groupby(["game_id", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})

# red zone: share of drives reaching the 20 that end in a touchdown
dr = p[p.fixed_drive.notna()].groupby(["game_id", "fixed_drive"]).agg(team=("posteam", "first"), opp=("defteam", "first"),
        rz=("drive_inside20", "max"), res=("fixed_drive_result", "first")).reset_index()
dr["rz"] = dr.rz.fillna(0)
dr["rztd"] = ((dr.rz == 1) & (dr.res == "Touchdown")).astype(int)
rz_o = dr.groupby(["game_id", "team"]).agg(rz=("rz", "sum"), rztd=("rztd", "sum")).reset_index()
rz_d = dr.groupby(["game_id", "opp"]).agg(rz=("rz", "sum"), rztd=("rztd", "sum")).reset_index().rename(columns={"opp": "team"})
# penalties against
pen = p[(p.penalty == 1) & p.penalty_team.notna()].groupby(["game_id", "penalty_team"]).penalty_yards.sum().reset_index() \
       .rename(columns={"penalty_team": "team", "penalty_yards": "pen_yds"})

tg = off.add_prefix("o_").rename(columns={"o_game_id": "game_id", "o_team": "team"}) \
    .merge(dfn.add_prefix("d_").rename(columns={"d_game_id": "game_id", "d_team": "team"}), on=["game_id", "team"]) \
    .merge(rz_o.rename(columns={"rz": "o_rz", "rztd": "o_rztd"}), on=["game_id", "team"], how="left") \
    .merge(rz_d.rename(columns={"rz": "d_rz", "rztd": "d_rztd"}), on=["game_id", "team"], how="left") \
    .merge(pen, on=["game_id", "team"], how="left").fillna(0)
tg["one"] = 1

# metric: (numerator, denominator, prior value, prior weight in denominator units)
M = {
    "off_pass_epa": ("o_pass_epa", "o_db", 0.0, 150), "off_rush_epa": ("o_rush_epa", "o_ru", -0.08, 100),
    "def_pass_epa": ("d_pass_epa", "d_db", 0.0, 150), "def_rush_epa": ("d_rush_epa", "d_ru", -0.08, 100),
    "off_sr": ("o_succ", "o_n", 0.44, 200), "def_sr": ("d_succ", "d_n", 0.44, 200),
    "off_expl": ("o_expl", "o_n", 0.08, 200), "def_expl": ("d_expl", "d_n", 0.08, 200),
    "sack_rate_allowed": ("o_sk", "o_db", 0.065, 150), "sack_rate_made": ("d_sk", "d_db", 0.065, 150),
    "hit_rate_made": ("d_hit", "d_db", 0.15, 150),
    "giveaways_pg": ("o_to", "one", 1.3, 4), "takeaways_pg": ("d_to", "one", 1.3, 4),
    "rz_td_off": ("o_rztd", "o_rz", 0.56, 15), "rz_td_def": ("d_rztd", "d_rz", 0.56, 15),
    "proe": ("o_proe_sum", "o_proe_n", 0.0, 100), "pace": ("o_pace_sum", "o_pace_n", 30.0, 100),
    "pen_yds_pg": ("pen_yds", "one", 50.0, 4),
}
g = pd.read_csv(FEAT, parse_dates=["gameday"])
tg = tg.merge(g[["game_id", "gameday", "season"]], on="game_id").sort_values(["gameday", "game_id"])
HALF = 8.0; A = 1 - 0.5 ** (1 / HALF)
state, last, pre = {}, {}, []
for r in tg.itertuples(index=False):
    s = state.setdefault(r.team, {k: [v[2] * v[3], v[3]] for k, v in M.items()})
    if last.get(r.team, r.season) != r.season:      # offseason: keep 60% of the evidence, refill with the prior
        for k, v in M.items():
            s[k][0] = 0.6 * s[k][0] + v[2] * v[3]; s[k][1] = 0.6 * s[k][1] + v[3]
    last[r.team] = r.season
    pre.append({"game_id": r.game_id, "team": r.team, **{k: s[k][0] / s[k][1] for k in M}})
    for k, (num, den, _, _) in M.items():
        s[k][0] = (1 - A) * s[k][0] + getattr(r, num); s[k][1] = (1 - A) * s[k][1] + getattr(r, den)
pre = pd.DataFrame(pre)
latest = pd.DataFrame({t: {k: v[0] / v[1] for k, v in s.items()} for t, s in state.items()}).T.rename_axis("team").reset_index()
latest.to_csv(os.path.join(os.path.dirname(OUT), "cand_form.csv"), index=False)

# ---------- QB-specific form: EPA per dropback for the actual starter ----------
qb = plays[(plays.db == 1) & plays.passer_player_id.notna()].groupby(["game_id", "passer_player_id"]) \
       .agg(qepa=("qb_epa", "sum"), qdb=("db", "sum")).reset_index()
qb = qb.merge(g[["game_id", "gameday"]], on="game_id").sort_values("gameday")
QPRIOR, QW = -0.10, 120          # unknown QBs start below average, worth about 3 games of evidence
QA = 1 - 0.5 ** (1 / 12)        # half life of 12 starts
qstate, qpre = {}, {}
for r in qb.itertuples(index=False):
    s = qstate.setdefault(r.passer_player_id, [QPRIOR * QW, QW])
    qpre[(r.game_id, r.passer_player_id)] = s[0] / s[1]
    s[0] = (1 - QA) * s[0] + r.qepa; s[1] = (1 - QA) * s[1] + r.qdb
def qb_val(gid, pid):
    if pd.isna(pid): return np.nan
    if (gid, pid) in qpre: return qpre[(gid, pid)]
    s = qstate.get(pid); return s[0] / s[1] if s else QPRIOR
for side in ("home", "away"):
    g[f"{side}_qb_epa"] = [qb_val(a, b) for a, b in zip(g.game_id, g[f"{side}_qb_id"])]
pd.DataFrame([{"qb_id": k, "qb_epa": v[0] / v[1], "dropbacks_weight": v[1]} for k, v in qstate.items()]) \
    .to_csv(os.path.join(os.path.dirname(OUT), "qb_form.csv"), index=False)

for side in ("home", "away"):
    m = g[["game_id", f"{side}_team"]].merge(pre, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"], how="left")
    for k in M:
        g[f"{side}_{k}"] = m[k].values

# ---------- Schedule spots ----------
TZ = {**{t: -8 for t in ["SEA", "SF", "LA", "LAC", "LV", "OAK", "SD", "STL_"]}, **{t: -7 for t in ["DEN", "ARI"]},
      **{t: -6 for t in ["KC", "CHI", "GB", "MIN", "DAL", "HOU", "NO", "TEN", "STL"]}}
tz = lambda t: TZ.get(t, -5)
g["tz_shift_away"] = [tz(h) - tz(a) for h, a in zip(g.home_team, g.away_team)]       # hours the away team travels east (+) or west (-)
g["west_early"] = [int(tz(a) == -8 and tz(h) == -5 and str(t)[:2] in ("09", "12", "13")) for a, h, t in zip(g.away_team, g.home_team, g.gametime)]
g["home_bye"] = (g.home_rest >= 13).astype(int); g["away_bye"] = (g.away_rest >= 13).astype(int)
g["short_week_away"] = (g.away_rest <= 4).astype(int); g["short_week_home"] = (g.home_rest <= 4).astype(int)
g["altitude"] = ((g.home_team == "DEN") & (g.neutral == 0)).astype(int)
g["cand_data"] = g.home_off_pass_epa.notna() & g.away_off_pass_epa.notna() & g.home_qb_epa.notna() & g.away_qb_epa.notna()
g.to_csv(OUT, index=False)
print("games with candidate data:", int(g.cand_data.sum()))
print(latest.set_index("team")[["off_pass_epa", "def_pass_epa", "pace", "proe"]].round(3).sort_values("off_pass_epa").tail(5))
