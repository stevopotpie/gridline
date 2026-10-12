"""Build pregame features for every NFL game: team scoring form (PPG), weather,
starter injuries and starting QB changes. Usage: python3 -I features.py DATA_DIR OUT_CSV"""
import sys, glob, os
import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2]

g = pd.read_csv(os.path.join(DATA, "games.csv"))
g = g[(g.season >= 2009)].copy()
g["gameday"] = pd.to_datetime(g.gameday)
g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
LEAGUE_PPG = 22.0

# ---------- 1. Team scoring form (exponentially weighted PPG for and against) ----------
HALF_LIFE = 6.0          # games
ALPHA = 1 - 0.5 ** (1 / HALF_LIFE)
OFFSEASON_REGRESS = 0.35  # pull 35% toward league average between seasons

off = {}; dfn = {}; last_season = {}
rows = []
for r in g.itertuples():
    feats = {}
    for side, team in (("home", r.home_team), ("away", r.away_team)):
        if team not in off:
            off[team] = LEAGUE_PPG; dfn[team] = LEAGUE_PPG; last_season[team] = r.season
        if last_season[team] != r.season:
            off[team] += OFFSEASON_REGRESS * (LEAGUE_PPG - off[team])
            dfn[team] += OFFSEASON_REGRESS * (LEAGUE_PPG - dfn[team])
            last_season[team] = r.season
        feats[f"{side}_ppg"] = off[team]
        feats[f"{side}_papg"] = dfn[team]
    rows.append(feats)
    if pd.notna(r.home_score):
        for team, pf, pa in ((r.home_team, r.home_score, r.away_score), (r.away_team, r.away_score, r.home_score)):
            off[team] += ALPHA * (pf - off[team])
            dfn[team] += ALPHA * (pa - dfn[team])
g = pd.concat([g, pd.DataFrame(rows)], axis=1)
pd.DataFrame({"team": list(off), "ppg": [off[t] for t in off], "papg": [dfn[t] for t in off]}).to_csv(os.path.join(os.path.dirname(OUT), "team_form.csv"), index=False)

# ---------- 2. Weather ----------
g["dome"] = g.roof.isin(["dome", "closed"]).astype(int)
g["wx_missing"] = ((g.dome == 0) & g.temp.isna()).astype(int)
g["temp_f"] = np.where(g.dome == 1, 70, g.temp)
g["wind_mph"] = np.where(g.dome == 1, 0, g.wind)
g["temp_f"] = g.temp_f.fillna(g.groupby(g.gameday.dt.month).temp.transform("mean")).fillna(60)
g["wind_mph"] = g.wind_mph.fillna(8)
g["freezing"] = (g.temp_f <= 32).astype(int)
g["wind15"] = np.clip(g.wind_mph - 15, 0, None)   # wind only hurts scoring past ~15 mph

# ---------- 3. Starting QB change ----------
qb_hist = {}
for side in ("home", "away"):
    g[f"{side}_backup_qb"] = 0
for i, r in g.iterrows():
    for side in ("home", "away"):
        team = r[f"{side}_team"]; qb = r[f"{side}_qb_id"]
        hist = qb_hist.setdefault((team, r.season), [])
        # Week 1 starter is treated as the planned starter; after that, the primary QB is
        # whoever has started the most games this season (ties go to the opening day starter).
        if hist and pd.notna(qb):
            vc = pd.Series(hist).value_counts()
            top = vc[vc == vc.max()].index
            primary = hist[0] if hist[0] in top else top[0]
            g.at[i, f"{side}_backup_qb"] = int(qb != primary)
        if pd.notna(qb):
            hist.append(qb)

# ---------- 4. Starters ruled Out / Doubtful ----------
GROUP = {"QB": "QB", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "LT": "OL", "LG": "OL", "RG": "OL", "RT": "OL",
         "WR": "WR", "TE": "WR", "RB": "RB", "FB": "RB",
         "DE": "DL", "DT": "DL", "NT": "DL", "LDE": "DL", "RDE": "DL", "LDT": "DL", "RDT": "DL",
         "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB", "WLB": "LB", "SLB": "LB", "LILB": "LB", "RILB": "LB",
         "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "LCB": "DB", "RCB": "DB", "NB": "DB"}

def load_starter_lists():
    """Return long table of (season, team, week, gsis_id) for players listed as starters."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "dc_*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        if "depth_team" in d.columns:
            d = d[(d.formation.isin(["Offense", "Defense"])) & (pd.to_numeric(d.depth_team, errors="coerce") == 1)]
            d = d[d.game_type == "REG"] if "game_type" in d.columns else d
            out.append(d.rename(columns={"club_code": "team"})[["season", "team", "week", "gsis_id"]])
        else:  # 2025+ daily snapshot format; take the latest snapshot before each game
            d = d[(d.pos_rank == 1) & d.pos_abb.map(lambda p: p in GROUP)]
            d["dt"] = pd.to_datetime(d.dt).dt.tz_localize(None)
            season = int(os.path.basename(f)[3:7])
            sched = g[g.season == season]
            for side in ("home", "away"):
                for r in sched[[f"{side}_team", "week", "gameday"]].itertuples(index=False):
                    team, week, day = r
                    snaps = d[(d.team == team) & (d.dt < day)]
                    if snaps.empty:
                        continue
                    snap = snaps[snaps.dt == snaps.dt.max()]
                    out.append(pd.DataFrame({"season": season, "team": team, "week": week, "gsis_id": snap.gsis_id.unique()}))
    s = pd.concat(out).dropna().drop_duplicates()
    s["week"] = s.week.astype(int)
    return s

starters = load_starter_lists()
# A "regular starter" for week w is someone listed as a starter in at least half of the team's
# previous 4 listed weeks (so a player who was hurt and pulled off the chart this week still counts).
starters = starters.sort_values(["season", "team", "week"])
reg = []
for (season, team), grp in starters.groupby(["season", "team"]):
    weeks = sorted(grp.week.unique())
    by_week = {w: set(grp[grp.week == w].gsis_id) for w in weeks}
    for w in range(1, 23):
        prior = [x for x in weeks if x < w][-4:]
        if not prior:
            prior = [x for x in weeks if x <= w][:1]
        if not prior:
            continue
        counts = pd.Series([p for x in prior for p in by_week[x]]).value_counts()
        keep = counts[counts >= max(1, len(prior) / 2)].index
        reg.append(pd.DataFrame({"season": season, "team": team, "week": w, "gsis_id": keep}))
reg = pd.concat(reg)

inj = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "injuries_*.csv"))])
inj = inj[inj.report_status.isin(["Out", "Doubtful"])]
inj["grp"] = inj.position.map(GROUP)
inj = inj.dropna(subset=["grp"])
hurt = inj.merge(reg, on=["season", "team", "week", "gsis_id"])
counts = hurt[hurt.grp != "QB"].pivot_table(index=["season", "team", "week"], columns="grp", values="gsis_id", aggfunc="nunique", fill_value=0)
counts.columns = [f"out_{c}" for c in counts.columns]
counts = counts.reset_index()

g["week_i"] = g.week.astype(int)
for side in ("home", "away"):
    m = g[["season", "week_i", f"{side}_team"]].merge(counts, left_on=["season", "week_i", f"{side}_team"],
                                                       right_on=["season", "week", "team"], how="left")
    for c in ["out_OL", "out_WR", "out_RB", "out_DL", "out_LB", "out_DB"]:
        g[f"{side}_{c}"] = m[c].fillna(0).values if c in m else 0
g["inj_data"] = g.season >= 2015

g["rest_diff"] = g.home_rest - g.away_rest
g["neutral"] = (g.location == "Neutral").astype(int)
g.to_csv(OUT, index=False)
print("rows", len(g), "| starters flagged out, mean per team-game:",
      round(g.loc[g.inj_data, [c for c in g.columns if "_out_" in c]].sum(axis=1).mean() / 2, 2))
print("backup QB starts share:", round(g[["home_backup_qb", "away_backup_qb"]].values.mean(), 3))
