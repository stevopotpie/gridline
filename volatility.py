"""Game-to-game volatility: how much each team's own scores and stats swing from one game to the next.
Writes volatility_2026.csv and prints a check of whether early-season volatility carries over.
Usage: python3 -I volatility.py DATA_DIR MODEL_DIR"""
import sys, glob, os
import numpy as np
import pandas as pd

DATA, M = sys.argv[1:3]
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "success", "yards_gained"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet")))], ignore_index=True)
p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"]) & p.epa.notna()]
off = p.groupby(["game_id", "season", "week", "posteam"]).agg(off_epa=("epa", "mean"), off_sr=("success", "mean"), yds=("yards_gained", "sum")).reset_index().rename(columns={"posteam": "team"})
dfn = p.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean"), yds_allowed=("yards_gained", "sum")).reset_index().rename(columns={"defteam": "team"})
tg = off.merge(dfn, on=["game_id", "team"])

g = pd.read_csv(os.path.join(DATA, "games.csv"))
g = g[g.game_type == "REG"]
sc = pd.concat([g[["game_id", "home_team", "home_score", "away_score"]].set_axis(["game_id", "team", "pf", "pa"], axis=1),
                g[["game_id", "away_team", "away_score", "home_score"]].set_axis(["game_id", "team", "pf", "pa"], axis=1)])
tg = tg.merge(sc, on=["game_id", "team"]).dropna(subset=["pf"])
STATS = {"pf": "Points scored", "pa": "Points allowed", "off_epa": "Offense EPA/play", "def_epa": "Defense EPA/play allowed",
         "yds": "Yards gained", "yds_allowed": "Yards allowed"}

def vol_table(df):
    sd = df.groupby("team")[list(STATS)].std()
    z = (sd - sd.mean()) / sd.std()
    sd["index"] = z.mean(axis=1)
    sd["games"] = df.groupby("team").size()
    return sd

# does it carry over? weeks 1-6 vs weeks 7-18, 2015-2025, plus whether early volatility predicts later misses vs the line
lines = pd.concat([g[["game_id", "home_team", "result", "spread_line"]].assign(sg=1).set_axis(["game_id", "team", "result", "line", "sg"], axis=1),
                   g[["game_id", "away_team", "result", "spread_line"]].assign(sg=-1).set_axis(["game_id", "team", "result", "line", "sg"], axis=1)])
lines["miss"] = (lines.sg * (lines.result - lines.line)).abs()
tg = tg.merge(lines[["game_id", "team", "miss"]], on=["game_id", "team"], how="left")
rows = []
for s in range(2015, 2026):
    e, l = tg[(tg.season == s) & (tg.week <= 6)], tg[(tg.season == s) & (tg.week >= 7)]
    ve, vl = vol_table(e), vol_table(l)
    rows.append(pd.DataFrame({"early": ve["index"], "late": vl["index"], "late_miss": l.groupby("team").miss.mean(),
                              "late_pf_sd": vl.pf, "early_pf_sd": ve.pf}).assign(season=s))
c = pd.concat(rows).dropna()
print(f"Carry-over check, {len(c)} team-seasons (2015-2025):")
print(f"  volatility index, weeks 1-6 vs weeks 7-18: correlation {c.early.corr(c.late):.2f}")
print(f"  points scored swings only: correlation {c.early_pf_sd.corr(c.late_pf_sd):.2f}")
print(f"  early volatility vs later average miss against the line: correlation {c.early.corr(c.late_miss):.2f}")
top, rest = c[c.early >= c.early.quantile(.75)], c[c.early < c.early.quantile(.75)]
print(f"  most volatile quarter early: later index {top.late.mean():+.2f}, later line miss {top.late_miss.mean():.1f} | others {rest.late.mean():+.2f}, {rest.late_miss.mean():.1f}")

cur = tg[tg.season == 2026]
v = vol_table(cur)
v["label"] = np.where(v["index"] >= 0.6, "High", np.where(v["index"] <= -0.6, "Steady", "Normal"))
v = v.round(3).sort_values("index", ascending=False)
pd.set_option("display.width", 200)
print(v.to_string())
v.rename_axis("team").reset_index().to_csv(os.path.join(M, "volatility_2026.csv"), index=False)
