"""Candidate improvements, all pregame (no look-ahead):
  1. Opponent-adjusted efficiency  2. Garbage time removed (win probability 10-90%)
  3. Market-implied power ratings from earlier weeks' closing lines  4. Referee crew tendencies
  5. Previous-week hangover (division game, shootout, overtime, offensive and defensive snaps, turnover margin)  6. Division game between two decent teams
Usage: python3 -I improvements.py DATA_DIR FEATURES_ALL_CSV OUT_CSV"""
import sys, glob, os
import numpy as np
import pandas as pd

DATA, FEAT, OUT = sys.argv[1:4]
g = pd.read_csv(FEAT, parse_dates=["gameday"], low_memory=False).sort_values(["gameday", "game_id"]).reset_index(drop=True)
COLS = ["game_id", "posteam", "defteam", "play_type", "epa", "success", "qb_dropback", "wp", "penalty", "qtr", "interception", "fumble_lost"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet")))], ignore_index=True)
pens = p[p.penalty == 1].groupby("game_id").size().rename("pens")
ot = p.groupby("game_id").qtr.max().gt(4).astype(int).rename("ot")
pl = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()].copy()
pl["garbage"] = ~pl.wp.between(0.10, 0.90)                       # garbage time: game basically decided
pl["db"] = (pl.qb_dropback == 1).astype(int)
pl["pepa"] = pl.epa * pl.db
def agg(x, side):
    return x.groupby(["game_id", side]).agg(n=("epa", "size"), epa=("epa", "sum"), sr=("success", "sum"), db=("db", "sum"), pepa=("pepa", "sum")).reset_index().rename(columns={side: "team"})
clean = pl[~pl.garbage]
off, dfn = agg(clean, "posteam"), agg(clean, "defteam")
snaps_faced = pl.groupby(["game_id", "defteam"]).size().rename("def_snaps").reset_index().rename(columns={"defteam": "team"})
off_snaps = pl.groupby(["game_id", "posteam"]).size().rename("off_snaps")
p["to"] = p.interception.fillna(0) + p.fumble_lost.fillna(0)
give = p[p.posteam.notna()].groupby(["game_id", "posteam"]).to.sum()
take = p[p.defteam.notna()].groupby(["game_id", "defteam"]).to.sum()

# ---------- 1 + 2: garbage-time-free efficiency, raw and opponent-adjusted ----------
HALF = 8.0; A = 1 - 0.5 ** (1 / HALF)
M = ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass", "def_pass"]
PRIOR = {"off_epa": 0.0, "def_epa": 0.0, "off_sr": 0.44, "def_sr": 0.44, "off_pass": 0.05, "def_pass": 0.05}
PW = 150
raw, adj, last = {}, {}, {}
offd = {(r.game_id, r.team): r for r in off.itertuples()}
defd = {(r.game_id, r.team): r for r in dfn.itertuples()}
def new_state(): return {k: [PRIOR[k] * PW, PW] for k in M}
def val(s, k): return s[k][0] / s[k][1]
pre = {}
for r in g.itertuples():
    for team in (r.home_team, r.away_team):
        for st in (raw, adj):
            if team not in st: st[team] = new_state()
        if last.get(team, r.season) != r.season:
            for st in (raw, adj):
                for k in M: st[team][k] = [0.6 * st[team][k][0] + PRIOR[k] * PW, 0.6 * st[team][k][1] + PW]
        last[team] = r.season
    for side, team in (("home", r.home_team), ("away", r.away_team)):
        pre[(r.game_id, side)] = {**{f"gf_{k}": val(raw[team], k) for k in M}, **{f"oa_{k}": val(adj[team], k) for k in M}}
    # league means of current adjusted ratings (for the opponent adjustment)
    teams_now = list(adj)
    lg = {k: np.mean([val(adj[t], k) for t in teams_now]) for k in M}
    for team, opp in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
        o, d_ = offd.get((r.game_id, team)), defd.get((r.game_id, team))
        if o is None or d_ is None: continue
        upd = {"off_epa": (o.epa, o.n), "def_epa": (d_.epa, d_.n), "off_sr": (o.sr, o.n), "def_sr": (d_.sr, d_.n), "off_pass": (o.pepa, o.db), "def_pass": (d_.pepa, d_.db)}
        # adjustment: credit the offense for facing a good defense (and the defense for facing a good offense)
        opp_pair = {"off_epa": "def_epa", "def_epa": "off_epa", "off_sr": "def_sr", "def_sr": "off_sr", "off_pass": "def_pass", "def_pass": "off_pass"}
        for k, (num, den) in upd.items():
            raw[team][k][0] = (1 - A) * raw[team][k][0] + num; raw[team][k][1] = (1 - A) * raw[team][k][1] + den
            shift = val(adj[opp], opp_pair[k]) - lg[opp_pair[k]]
            adj[team][k][0] = (1 - A) * adj[team][k][0] + (num - den * shift); adj[team][k][1] = (1 - A) * adj[team][k][1] + den
for side in ("home", "away"):
    f = pd.DataFrame([pre[(gid, side)] for gid in g.game_id])
    for c in f.columns: g[f"{side}_{c}"] = f[c].values

# ---------- 3: market-implied power ratings from earlier weeks' closing lines ----------
g["wkidx"] = (g.season - 2009) * 25 + g.week.astype(int)
teams = sorted(set(g.home_team) | set(g.away_team)); ti = {t: i for i, t in enumerate(teams)}
mk_spread = np.full(len(g), np.nan); mk_total = np.full(len(g), np.nan)
lined = g[g.spread_line.notna() & g.total_line.notna()]
betas = {}
for (season, week), wk in g.groupby(["season", "week"]):
    idx = wk.wkidx.iloc[0]
    past = lined[(lined.wkidx < idx) & (lined.wkidx >= idx - 60)]
    if len(past) < 40: continue
    age = idx - past.wkidx + 6 * (past.season < season)          # the offseason counts as six extra weeks of staleness
    w = 0.5 ** (age / 6.0)
    n = len(teams); X = np.zeros((len(past), n + 1)); Xt = np.zeros((len(past), n + 1))
    hi = past.home_team.map(ti).values; ai = past.away_team.map(ti).values; rr = np.arange(len(past))
    X[rr, hi] = 1; X[rr, ai] = -1; X[:, n] = 1 - past.neutral.values
    Xt[rr, hi] = 1; Xt[rr, ai] = 1; Xt[:, n] = 1
    W = np.sqrt(w.values)[:, None]
    lam = np.eye(n + 1) * 2.0; lam[n, n] = 0
    beta = np.linalg.solve((X * W).T @ (X * W) + lam, (X * W).T @ (past.spread_line.values * W.ravel()))
    beta_t = np.linalg.solve((Xt * W).T @ (Xt * W) + lam, (Xt * W).T @ (past.total_line.values * W.ravel()))
    betas[(season, week)] = beta
    for i in wk.index:
        h, a = ti[g.at[i, "home_team"]], ti[g.at[i, "away_team"]]
        mk_spread[i] = beta[h] - beta[a]                           # rating gap only; home field is modeled separately
        mk_total[i] = beta_t[n] + beta_t[h] + beta_t[a]
g["mkt_diff"] = mk_spread; g["mkt_total"] = mk_total
# ratings for the current week come only from earlier weeks' lines, so this week's line is not counted twice
open_wk = g[g.result.isna() & g.spread_line.notna()]
key = (int(open_wk.season.min()), int(open_wk[open_wk.season == open_wk.season.min()].week.min())) if len(open_wk) else max(betas)
latest_beta = betas.get(key, betas[max(betas)])
pd.DataFrame({"team": teams, "mkt_rating": np.round(latest_beta[:len(teams)], 3)}).to_csv(os.path.join(os.path.dirname(OUT), "mkt_ratings.csv"), index=False)

# ---------- 4: referee crews ----------
g = g.merge(pens, left_on="game_id", right_index=True, how="left").merge(ot, left_on="game_id", right_index=True, how="left")
g["ou_res"] = g.total - g.total_line
ref_ou, ref_pen, ref_n = {}, {}, {}
r_ou, r_pen = [], []
LG_PEN = g.pens.mean()
for r in g.itertuples():
    ref = r.referee if isinstance(r.referee, str) else None
    k = ref_n.get(ref, 0)
    r_ou.append(ref_ou.get(ref, 0.0) / (k + 16) if ref else 0.0)
    r_pen.append((ref_pen.get(ref, 0.0) + 16 * LG_PEN) / (k + 16) if ref else LG_PEN)
    if ref and not pd.isna(r.ou_res) and not pd.isna(r.pens):
        ref_ou[ref] = ref_ou.get(ref, 0.0) + r.ou_res; ref_pen[ref] = ref_pen.get(ref, 0.0) + r.pens; ref_n[ref] = k + 1
g["ref_ou"] = r_ou; g["ref_pen"] = r_pen

# ---------- 5: previous-week hangover ----------
sf = snaps_faced.set_index(["game_id", "team"]).def_snaps
rows = []
for side, opp in (("home", "away"), ("away", "home")):
    x = g[["game_id", "season", "gameday", f"{side}_team", "div_game", "total", "result", "ot"]].copy()
    x.columns = ["game_id", "season", "gameday", "team", "div", "total", "result", "ot"]
    x["close"] = (x.result.abs() <= 3).astype(float).where(x.result.notna())
    rows.append(x)
t = pd.concat(rows).sort_values(["team", "gameday"])
t["def_snaps"] = [sf.get((a, b), np.nan) for a, b in zip(t.game_id, t.team)]
t["off_snaps"] = [off_snaps.get((a, b), np.nan) for a, b in zip(t.game_id, t.team)]
t["to_margin"] = [take.get((a, b), np.nan) - give.get((a, b), np.nan) for a, b in zip(t.game_id, t.team)]
grp = t.groupby(["team", "season"])
for c in ("div", "total", "ot", "close", "def_snaps", "off_snaps", "to_margin"):
    t[f"prev_{c}"] = grp[c].shift(1)
t["prev_shootout"] = (t.prev_total >= 55).astype(float)
t["prev_gap_days"] = (t.gameday - grp.gameday.shift(1)).dt.days
defaults = {"prev_div": 0, "prev_ot": 0, "prev_close": 0, "prev_shootout": 0, "prev_total": t.total.mean(), "prev_def_snaps": t.def_snaps.mean(), "prev_off_snaps": t.off_snaps.mean(), "prev_to_margin": 0}
last_played = t[t.off_snaps.notna()].groupby("team").tail(1)
last_played[["team", "off_snaps", "def_snaps"]].to_csv(os.path.join(os.path.dirname(OUT), "last_snaps.csv"), index=False)
t = t.fillna(defaults)
t = t.set_index(["game_id", "team"])
for side in ("home", "away"):
    k = list(zip(g.game_id, g[f"{side}_team"]))
    for c in ("prev_div", "prev_ot", "prev_close", "prev_shootout", "prev_total", "prev_def_snaps", "prev_off_snaps", "prev_to_margin"):
        g[f"{side}_{c}"] = t[c].reindex(k).values

# ---------- 6: division game between two decent teams ----------
rec = []
for side, sg in (("home", 1), ("away", -1)):
    x = g[["game_id", "season", "gameday", f"{side}_team", "result"]].copy(); x.columns = ["game_id", "season", "gameday", "team", "result"]
    x["w"] = (sg * x.result > 0).astype(float).where(x.result.notna()); x["gp"] = x.result.notna().astype(float); rec.append(x)
rec = pd.concat(rec).sort_values("gameday")
rg = rec.groupby(["team", "season"])
rec["w_pre"] = rg.w.transform(lambda s: s.shift(1).fillna(0).cumsum()); rec["gp_pre"] = rg.gp.transform(lambda s: s.shift(1).fillna(0).cumsum())
rec["wpct_pre"] = np.where(rec.gp_pre > 0, rec.w_pre / rec.gp_pre.replace(0, np.nan), 0.5)
rec = rec.set_index(["game_id", "team"])
for side in ("home", "away"):
    g[f"{side}_wpct_pre"] = rec.wpct_pre.reindex(list(zip(g.game_id, g[f"{side}_team"]))).values
g["div_good"] = ((g.div_game == 1) & (g.home_wpct_pre >= 0.5) & (g.away_wpct_pre >= 0.5) & (g.week.astype(int) >= 4)).astype(int)

g.to_csv(OUT, index=False)
print("rows", len(g), "| market ratings available:", int(g.mkt_diff.notna().sum()), "| division games with two decent teams:", int(g.div_good.sum()))
