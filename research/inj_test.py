"""Test: weight injured players by recent snap share / opportunity share instead of counting starters out."""
import sys, glob, os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
W = sys.argv[1]; D = os.path.join(W, "data")
GROUP = {"QB": "QB", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "LT": "OL", "LG": "OL", "RG": "OL", "RT": "OL",
         "WR": "WR", "TE": "WR", "RB": "RB", "FB": "RB",
         "DE": "DL", "DT": "DL", "NT": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
         "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB"}
POS = ["OL", "WR", "RB", "DL", "LB", "DB"]

# ---------- recent snap share per player (previous 4 team games, 0 if he didn't play) ----------
sn = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(D, "snap_*.csv")))])
sn["share"] = sn[["offense_pct", "defense_pct"]].max(axis=1)
sn["wk"] = sn.season * 100 + sn.week
games = sn[["team", "season", "week", "wk"]].drop_duplicates().sort_values(["team", "wk"])
games["gi"] = games.groupby("team").cumcount()
sn = sn.merge(games, on=["team", "season", "week", "wk"])
recent = {}
for team, grp in sn.groupby("team"):
    pv = grp.pivot_table(index="gi", columns="pfr_player_id", values="share", aggfunc="max")
    pv = pv.reindex(range(int(games[games.team == team].gi.max()) + 2)).fillna(0)
    recent[team] = pv.shift(1).rolling(4, min_periods=1).mean()     # share over the 4 games before game gi
gi_of = {(r.team, r.season, r.week): r.gi for r in games.itertuples()}
nextgi = games.groupby("team").gi.max().to_dict()

# ---------- opportunity share for skill players (targets + carries, previous 4 team games) ----------
COLS = ["game_id", "season", "week", "season_type", "posteam", "receiver_player_id", "rusher_player_id", "pass_attempt", "rush_attempt"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, "pbp_*.parquet")))])
p = p[p.posteam.notna()]
t = p[p.receiver_player_id.notna()].groupby(["season", "week", "posteam", "receiver_player_id"]).size().rename("n").reset_index().rename(columns={"receiver_player_id": "pid"})
c = p[p.rusher_player_id.notna() & (p.rush_attempt == 1)].groupby(["season", "week", "posteam", "rusher_player_id"]).size().rename("n").reset_index().rename(columns={"rusher_player_id": "pid"})
op = pd.concat([t, c]).groupby(["season", "week", "posteam", "pid"]).n.sum().reset_index()
op["share"] = op.n / op.groupby(["season", "week", "posteam"]).n.transform("sum")
og = op[["posteam", "season", "week"]].drop_duplicates().sort_values(["posteam", "season", "week"])
og["gi"] = og.groupby("posteam").cumcount()
op = op.merge(og, on=["posteam", "season", "week"])
orecent = {}
for team, grp in op.groupby("posteam"):
    pv = grp.pivot_table(index="gi", columns="pid", values="share", aggfunc="sum").reindex(range(int(grp.gi.max()) + 2)).fillna(0)
    orecent[team] = pv.shift(1).rolling(4, min_periods=1).mean()
ogi = {(r.posteam, r.season, r.week): r.gi for r in og.itertuples()}

# ---------- injured players ----------
pl = pd.read_csv(os.path.join(D, "players.csv"), low_memory=False)[["gsis_id", "pfr_id"]].dropna()
pfr = dict(zip(pl.gsis_id, pl.pfr_id))
inj = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(os.path.join(D, "injuries_*.csv"))])
inj = inj[inj.report_status.isin(["Out", "Doubtful"]) & (inj.game_type == "REG")]
inj["grp"] = inj.position.map(GROUP); inj = inj[inj.grp.notna() & (inj.grp != "QB")]
inj["pfr"] = inj.gsis_id.map(pfr)
print("injured rows", len(inj), "| mapped to snap ids", round(inj.pfr.notna().mean(), 3))

def snap_share(r):
    gi = gi_of.get((r.team, r.season, r.week))
    if gi is None:                       # game not played yet: use the next index
        gi = nextgi.get(r.team, -1) + 1
    tab = recent.get(r.team)
    if tab is None or pd.isna(r.pfr) or r.pfr not in tab.columns or gi not in tab.index: return 0.0
    return float(tab.at[gi, r.pfr])
def opp_share(r):
    tab = orecent.get(r.team)
    if tab is None or r.gsis_id not in tab.columns: return 0.0
    gi = ogi.get((r.team, r.season, r.week))
    if gi is None: gi = int(tab.index.max())
    return float(tab.at[gi, r.gsis_id]) if gi in tab.index else 0.0
inj["sw"] = [snap_share(r) for r in inj.itertuples()]
inj["ow"] = [opp_share(r) if r.grp in ("WR", "RB") else 0.0 for r in inj.itertuples()]
inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"])
agg = inj.pivot_table(index=["season", "team", "week"], columns="grp", values=["sw", "ow"], aggfunc="sum", fill_value=0)
agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg = agg.reset_index()

# ---------- model test ----------
g = pd.read_csv(os.path.join(W, "games_features_plus.csv"), parse_dates=["gameday"], low_memory=False)
g["week_i"] = g.week.astype(int)
for side in ("home", "away"):
    m = g[["season", "week_i", f"{side}_team"]].merge(agg, left_on=["season", "week_i", f"{side}_team"], right_on=["season", "week", "team"], how="left")
    for c_ in [f"sw_{x}" for x in POS] + ["ow_WR", "ow_RB"]:
        g[f"{side}_{c_}"] = m[c_].fillna(0).values if c_ in m else 0.0
exec(open(os.path.join(W, "final_model.py")).read().split("done = g[")[0].split("g = pd.read_csv(sys.argv[1]")[0])  # imports only
# rebuild final_model's features
src = open(os.path.join(W, "final_model.py")).read()
body = src.split('OUT = sys.argv[2]\n')[1].split("done = g[")[0]
exec(body)
for x in POS: g[f"diff_sw_{x}"] = g[f"home_sw_{x}"] - g[f"away_sw_{x}"]
for x in ("WR", "RB"): g[f"diff_ow_{x}"] = g[f"home_ow_{x}"] - g[f"away_ow_{x}"]
g["sw_offense"] = sum(g[f"{s}_sw_{x}"] for s in ("home", "away") for x in ("OL", "WR", "RB"))
g["sw_defense"] = sum(g[f"{s}_sw_{x}"] for s in ("home", "away") for x in ("DL", "LB", "DB"))
g["ow_skill"] = sum(g[f"{s}_ow_{x}"] for s in ("home", "away") for x in ("WR", "RB"))
CNT = [f"diff_out_{x}" for x in POS]
SW = [f"diff_sw_{x}" for x in POS]
SWO = [f"diff_sw_{x}" for x in ("OL", "DL", "LB", "DB")] + ["diff_ow_WR", "diff_ow_RB"]
noinj_S = [c for c in FINAL_S if c not in CNT]; noinj_T = [c for c in FINAL_T if c not in ("out_offense", "out_defense")]
V = {"current (starter counts)": (FINAL_S, FINAL_T),
     "no injuries at all": (noinj_S, noinj_T),
     "snap-share weighted": (noinj_S + SW, noinj_T + ["sw_offense", "sw_defense"]),
     "snap share + skill opportunity": (noinj_S + SWO, noinj_T + ["sw_offense", "sw_defense", "ow_skill"]),
     "counts + snap share": (FINAL_S + SW, FINAL_T + ["sw_offense", "sw_defense"])}
done = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data & g.mkt_diff.notna()].copy()
def ats(pred, line, act):
    pick = pred > line; cv = act - line; push = cv == 0
    w = np.where(pick, cv > 0, cv < 0)[~push]; return int(w.sum()), int((~w).sum())
rec = lambda wl: f"{wl[0]}-{wl[1]} ({wl[0]/max(1,sum(wl)):.1%})"
rows = []
for name, (sx, tx) in V.items():
    out = []
    for season in range(2018, 2026):
        tr, te = done[done.season < season], done[done.season == season].copy()
        te["m"] = Ridge(1.0).fit(tr[sx], tr.result).predict(te[sx])
        trr = tr.assign(rt=tr.total - tr.total_line, rs=tr.result - tr.spread_line)
        te["sm"] = te.spread_line + Ridge(50).fit(trr[sx], trr.rs).predict(te[sx])
        te["tm"] = te.total_line + Ridge(50).fit(trr[tx + ["total_line"]], trr.rt).predict(te[tx + ["total_line"]])
        out.append(te)
    P = pd.concat(out)
    r = {"version": name, "spread MAE": round((P.m - P.result).abs().mean(), 3)}
    for e in (0, 2, 3, 4):
        s = P[(P.m - P.spread_line).abs() >= e]; r[f"ATS {e}+"] = rec(ats(s.m.values, s.spread_line.values, s.result.values))
    s = P[(P.sm - P.spread_line).abs() >= 1]; r["V+adj spread 1+"] = rec(ats(s.sm.values, s.spread_line.values, s.result.values))
    s = P[(P.tm - P.total_line).abs() >= 1.5]; r["total lean 1.5+"] = rec(ats(s.tm.values, s.total_line.values, s.total.values))
    r["total MAE"] = round((P.tm - P.total).abs().mean(), 3)
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
fs = Ridge(1.0).fit(done[noinj_S + SWO], done.result)
print("\npoints per fully missing player-share (spread model, home minus away):")
for c_ in SWO: print(f"  {c_:12s} {dict(zip(noinj_S + SWO, fs.coef_))[c_]:+.2f}")
fc = Ridge(1.0).fit(done[FINAL_S], done.result)
print("current counts, points per starter out:", {c_: round(v, 2) for c_, v in zip(FINAL_S, fc.coef_) if c_ in CNT})
