"""Gridline players, step 4: weekly player projections and prop probabilities.

Uses the winning setup from player_experiments.py: ridge regression on usage, efficiency, game script,
opponent and weather for the mean; a normal spread that widens with the mean for yardage milestones;
Poisson for touchdowns; then a calibration layer fitted on walk forward predictions from 2019 on.

Usage: python player_props.py DATA_DIR WORK_DIR
Writes player_props_2026.csv, player_season_2026.csv, player_weather_splits.csv, player_backtest.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import player_features as F

D, W = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{W}/player_best_models.json"))
pgames = pd.read_csv(f"{W}/player_games.csv")
tgames = pd.read_csv(f"{W}/team_games.csv")
games = pd.read_csv(f"{D}/games.csv")
sg = pd.read_csv(f"{W}/season_2026_games.csv")

# ---------- the slate: the next two weeks with unplayed games ----------
fut = sg[sg.home_score.isna()]
weeks = sorted(fut.week.unique())[:2]
slate = fut[fut.week.isin(weeks)].merge(games[["game_id", "home_qb_id", "away_qb_id"]], on="game_id", how="left")
# Market lines when posted, Gridline's projection otherwise
slate["use_total"] = slate.total_line.fillna(slate.p_total)
slate["use_spread"] = slate.spread_line.fillna(slate.p_margin)
slate["lines"] = np.where(slate.total_line.notna(), "market", "Gridline")

# ---------- who plays: recent players for each team, minus Out and Doubtful ----------
cur = pgames[(pgames.season == 2026)].copy()
cur["game_date"] = pd.to_datetime(cur.game_date)
last_team = cur.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id").team
team_recent = cur.groupby("team").game_date.apply(lambda s: sorted(s.unique())[-3:])
recent = cur[cur.apply(lambda r: r.game_date in team_recent[r.team], axis=1)]
roster = recent[recent.player_id.map(last_team) == recent.team].drop_duplicates("player_id")[
    ["player_id", "team", "display_name", "position", "headshot"]]
inj = pd.read_csv(f"{D}/injuries_2026.csv")
inj = inj[inj.week == inj.week.max()]
out_ids = set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id)
roster = roster[~roster.player_id.isin(out_ids)]
inj_note = dict(zip(inj.gsis_id, inj.report_status))

# Weather for the slate: forecast when saved, else typical for that stadium and month
hist = games[(games.season.between(2015, 2025))].copy()
hist["month"] = pd.to_datetime(hist.gameday).dt.month
hist["is_dome"] = hist.roof.isin(["dome", "closed"]).astype(int)
typ = hist.groupby(["home_team", "month"]).agg(dome=("is_dome", "mean"), temp=("temp", "mean"), wind=("wind", "mean"))
fc = pd.read_csv(f"{W}/forecasts.csv").set_index("game_id") if os.path.exists(f"{W}/forecasts.csv") else None

players = pd.read_csv(f"{D}/players.csv", usecols=["gsis_id", "display_name", "headshot"], low_memory=False)
qb_names = dict(zip(players.gsis_id, players.display_name))
heads = dict(zip(players.gsis_id, players.headshot))
rows, trows, wx = [], [], {}
for _, g in slate.iterrows():
    gd = pd.to_datetime(g.gameday)
    if fc is not None and g.game_id in fc.index:
        f = fc.loc[g.game_id]
        closed = int(f.roof_closed) == 1 or g.roof == "dome"
        w = {"roof": "closed" if closed else "outdoors", "temp": float(f.temp_f), "wind": float(f.wind_mph), "wx_source": "forecast"}
    elif g.roof == "dome":
        w = {"roof": "dome", "temp": 70, "wind": 0, "wx_source": "dome"}
    else:
        t = typ.loc[(g.home_team, gd.month)] if (g.home_team, gd.month) in typ.index else typ.xs(g.home_team, level=0).mean()
        w = {"roof": "closed" if t.dome >= 0.5 else "outdoors", "temp": float(t.temp) if pd.notna(t.temp) else 60,
             "wind": float(t.wind) if pd.notna(t.wind) else 6, "wx_source": "typical"}
    wx[g.game_id] = w
    for side, opp in (("home", "away"), ("away", "home")):
        team, oteam = g[f"{side}_team"], g[f"{opp}_team"]
        sign = 1 if side == "home" else -1
        trows.append({"game_id": g.game_id, "season": 2026, "week": g.week, "season_type": "REG", "game_date": g.gameday,
                      "team": team, "opp": oteam, "spread_line": g.use_spread, "total_line": g.use_total,
                      "implied_pts": g.use_total / 2 + sign * g.use_spread / 2, "is_home": int(side == "home")})
        qb_id = g[f"{side}_qb_id"]
        for _, p in roster[roster.team == team].iterrows():
            if p.position == "QB" and p.player_id != qb_id:
                continue   # only the listed starter gets passing props
            rows.append({"game_id": g.game_id, "season": 2026, "week": g.week, "season_type": "REG", "game_date": g.gameday,
                         "team": team, "opp": oteam, "player_id": p.player_id, "display_name": p.display_name,
                         "name": p.display_name, "position": p.position, "headshot": p.headshot,
                         "listed_qb": int(p.player_id == qb_id)})
        if pd.notna(qb_id) and qb_id not in set(roster.player_id):
            nm = qb_names.get(qb_id)
            rows.append({"game_id": g.game_id, "season": 2026, "week": g.week, "season_type": "REG", "game_date": g.gameday,
                         "team": team, "opp": oteam, "player_id": qb_id, "display_name": nm, "name": nm, "position": "QB",
                         "headshot": heads.get(qb_id), "listed_qb": 1})

allp = pd.concat([pgames, pd.DataFrame(rows)], ignore_index=True)
allt = pd.concat([tgames, pd.DataFrame(trows)], ignore_index=True)
for gid, w in wx.items():
    games.loc[games.game_id == gid, ["roof", "temp", "wind"]] = [w["roof"], w["temp"], w["wind"]]
pg = F.build(allp, allt, games)
pg["pos_rb"] = (pg.position == "RB").astype(int)
pg["pos_te"] = (pg.position == "TE").astype(int)
# A backup who is the listed starter: keep his per attempt efficiency, give him a starter's share of dropbacks
bk = (~pg.played) & (pg.get("listed_qb", 0).fillna(0) == 1) & (pg.e_pass_att.fillna(0) < 0.8 * pg.t_team_dropbacks)
if bk.any():
    vol = 0.93 * pg.loc[bk, "t_team_dropbacks"]
    ratio = vol / pg.loc[bk, "e_pass_att"].clip(lower=1)
    pg.loc[bk, "e_pass_att"] = vol
    for c in ["e_pass_yds", "std_pass_yds"]:
        pg.loc[bk, c] = vol * pg.loc[bk, "e_ypa"]
    for c in ["e_pass_td", "std_pass_td"]:
        pg.loc[bk, c] = (pg.loc[bk, c] * ratio).clip(upper=2.0)
    pg.loc[bk, "e_pass_share"] = 0.93
hist_rows = pg[pg.played & (pg.season >= 2016) & (pg.n_prior >= 3)]
new = pg[~pg.played].copy()

# ---------- fit, walk forward for calibration, then project ----------
MARKETS = ["pass_yds", "pass_td", "rush_yds", "rec_yds", "receptions", "tds"]
MILE = {"pass_yds": [("p_pass300", 300)], "rush_yds": [("p_rush100", 100)], "rec_yds": [("p_rec100", 100)],
        "tds": [("p_td1", 1), ("p_td2", 2)], "pass_td": [("p_ptd2", 2), ("p_ptd3", 3)]}
COUNT = {"tds", "pass_td", "receptions"}


def ridge(tr, cols, m):
    med = tr[cols].median()
    r = Ridge(alpha=1.0).fit(tr[cols].fillna(med), tr[m])
    return lambda d: np.clip(r.predict(d[cols].fillna(med)), 0, None)


def raw_prob(m, k, mu, sdfit):
    if m in COUNT:
        return 1 - stats.poisson.cdf(k - 1, np.maximum(mu, 1e-6))
    a, b = sdfit
    return 1 - stats.norm.cdf(k - 0.5, mu, np.maximum(a + b * mu, 5))


def sd_fit(y, mu):
    A = np.vstack([np.ones(len(y)), mu]).T
    return np.linalg.lstsq(A, np.abs(y - mu) * np.sqrt(np.pi / 2), rcond=None)[0]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


backtest, oos_all = {}, []
for m in MARKETS:
    cols = cfg[m]["features"]
    d = hist_rows[F.eligible(hist_rows, m)]
    # Walk forward 2019 to 2026 so every calibration point is out of sample
    oos = []
    for s in range(2019, 2027):
        tr, te = d[d.season < s], d[d.season == s]
        if not len(te):
            continue
        f = ridge(tr, cols, m)
        mu_tr, mu_te = f(tr), f(te)
        sdf = sd_fit(tr[m].values, mu_tr)
        o = te[["season", "game_id", "player_id", m]].assign(mu=mu_te, naive=te[f"std_{m}"].values)
        for name, k in MILE.get(m, []):
            o[name] = raw_prob(m, k, mu_te, sdf)
        oos.append(o)
    oos = pd.concat(oos)
    bt = {"n": int(len(oos)), "mae": float((oos[m] - oos.mu).abs().mean()),
          "mae_season_avg": float((oos[m] - oos.naive).abs().mean()), "milestones": {}}
    f = ridge(d, cols, m)
    mu_all = f(d)
    sdf = sd_fit(d[m].values, mu_all)
    bt["sd"] = [float(sdf[0]), float(sdf[1])]   # spread of outcomes = a + b x projection (used by the bet log meter)
    e = F.eligible(new, m)
    new.loc[e, f"proj_{m}"] = f(new[e])
    for name, k in MILE.get(m, []):
        y = (oos[m] >= k).astype(int).values
        # Platt calibration on the walk forward probabilities (2019 to 2025) corrects any systematic lean
        cal_rows = oos.season < 2026
        cal = LogisticRegression(C=100).fit(logit(oos.loc[cal_rows, name].values).reshape(-1, 1), y[cal_rows.values])
        p_raw = oos[name].values
        p_cal = cal.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
        bins = pd.cut(p_cal, [0, .05, .1, .2, .3, .4, .5, .6, .8, 1])
        rel = pd.DataFrame({"bin": bins, "p": p_cal, "y": y}).groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
        bt["milestones"][name] = {"line": k, "hit_rate": float(y.mean()), "logloss_raw": ll(p_raw), "logloss_cal": ll(p_cal),
                                  "logloss_base": ll(np.full(len(y), y.mean())),
                                  "reliability": [{"bin": str(i), **{kk: (int(v) if kk == "n" else round(float(v), 3)) for kk, v in r.items()}}
                                                  for i, r in rel.iterrows()]}
        new.loc[e, name] = cal.predict_proba(logit(raw_prob(m, k, new.loc[e, f"proj_{m}"].values, sdf)).reshape(-1, 1))[:, 1]
    backtest[m] = bt
    oos_all.append(oos[["season", "game_id", "player_id", "mu"]].assign(market=m))
pd.concat(oos_all).to_csv(f"{W}/player_oos.csv", index=False)   # walk forward game projections, used by player_quarters.py
json.dump(backtest, open(f"{W}/player_backtest.json", "w"), indent=1)


def american(p):
    if pd.isna(p) or p <= 0 or p >= 1:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


keep = ["game_id", "week", "game_date", "team", "opp", "player_id", "display_name", "position", "headshot"] + \
       [c for c in new.columns if c.startswith("proj_") or c.startswith("p_")]
props = new[keep].copy()
props = props[props[[c for c in keep if c.startswith("proj_")]].notna().any(axis=1)]
for c in [c for c in props.columns if c.startswith("p_")]:
    props[c.replace("p_", "fair_")] = props[c].map(american)
props["weather"] = props.game_id.map(lambda g: wx[g])
props["wx_temp"] = props.weather.map(lambda w: w["temp"]); props["wx_wind"] = props.weather.map(lambda w: w["wind"])
props["wx_roof"] = props.weather.map(lambda w: w["roof"]); props["wx_source"] = props.weather.map(lambda w: w["wx_source"])
props["lines"] = props.game_id.map(dict(zip(slate.game_id, slate.lines)))
props["injury"] = props.player_id.map(inj_note).fillna("")
props = props.drop(columns="weather").sort_values(["week", "game_id", "team", "p_td1"], ascending=[True, True, True, False])
props.to_csv(f"{W}/player_props_2026.csv", index=False)

# ---------- raw 2026 season stats ----------
s26 = pgames[(pgames.season == 2026) & (pgames.season_type == "REG")]
agg = {c: "sum" for c in ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
                          "targets", "receptions", "rec_yds", "rec_td", "tds"]}
season = s26.groupby(["player_id", "display_name", "position"]).agg(team=("team", "last"), games=("game_id", "nunique"), **{k: (k, v) for k, v in agg.items()}).reset_index()
season = season[(season.pass_att >= 10) | (season.carries >= 5) | (season.targets >= 5)]
season.to_csv(f"{W}/player_season_2026.csv", index=False)

# ---------- weather splits since 2021: per game averages by condition ----------
wsp = pg[pg.played & (pg.season >= 2021)].copy()
spl = wsp.groupby(["player_id", "weather"]).agg(g=("game_id", "nunique"), pass_yds=("pass_yds", "mean"), rush_yds=("rush_yds", "mean"),
                                                 rec_yds=("rec_yds", "mean"), tds=("tds", "mean")).reset_index()
spl = spl[spl.player_id.isin(set(props.player_id) | set(season.player_id))]
spl.to_csv(f"{W}/player_weather_splits.csv", index=False)
print(f"weeks {weeks}: {len(props)} player projections in {props.game_id.nunique()} games")
for m, b in backtest.items():
    print(f"{m:11s} MAE {b['mae']:.2f} (season average {b['mae_season_avg']:.2f})",
          {k: round(v['logloss_base'] - v['logloss_cal'], 4) for k, v in b["milestones"].items()})
