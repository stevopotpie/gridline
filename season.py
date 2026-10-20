"""Project every 2026 regular season game, grade the games already played, and simulate the rest.
The model is trained on 2015-2025 only, so every 2026 projection is out of sample.
Usage: python3 -I season.py MODEL_DIR"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

M = sys.argv[1]
g = pd.read_csv(os.path.join(M, "games_features_plus.csv"), parse_dates=["gameday"], low_memory=False)
page = json.load(open(os.path.join(M, "page_data.json")))
POS = ["OL", "WR", "RB", "DL", "LB", "DB"]
SEASON = 2026
fut = (g.season == SEASON) & g.home_score.isna()

# ---------- fill unplayed 2026 games with each team's latest form ----------
st = pd.read_csv(os.path.join(M, "st_form.csv")).set_index("team")
cand = pd.read_csv(os.path.join(M, "cand_form.csv")).set_index("team")
for side in ("home", "away"):
    need = fut | ((g.season == SEASON) & ~(g.cand_data & g.st_data))   # also played games whose play-by-play isn't posted yet
    tn = g.loc[need, f"{side}_team"]
    for c in ["ko_ret_start", "pr_avg"]:
        g.loc[need, f"{side}_{c}"] = tn.map(st[c]).values
    for c in ["off_sr", "def_sr", "off_pass_epa", "def_pass_epa", "off_expl", "def_expl", "pace", "pen_yds_pg"]:
        g.loc[need, f"{side}_{c}"] = tn.map(cand[c]).values
    t = g.loc[fut, f"{side}_team"]
    # starting QB: each team's most recent starter, flagged as a backup only if someone else has more 2026 starts
    # starting QB: the starter nflverse lists for that game when there is one (it picks up injuries like
    # Huntley for Lamar), otherwise the team's most recent starter; a backup if someone else has more 2026 starts
    qform = pd.read_csv(os.path.join(M, "qb_form.csv")).set_index("qb_id").qb_epa
    def qb(team, listed_id, listed_name):
        info = page["qbs"].get(team)
        most = max(x["starts26"] for x in info["qbs"])
        if isinstance(listed_id, str) and listed_id:
            known = next((q for q in info["qbs"] if q["id"] == listed_id), None)
            starts = known["starts26"] if known else 0
            return float(qform.get(listed_id, -0.10)), int(starts < most), listed_name if isinstance(listed_name, str) else (known or {}).get("name", "")
        q = next(q for q in info["qbs"] if q["id"] == info["recent"])
        return q["epa"], int(q["starts26"] < most), q["name"]
    vals = [qb(a, b, c) for a, b, c in zip(t, g.loc[fut, f"{side}_qb_id"], g.loc[fut, f"{side}_qb_name"])]
    g.loc[fut, f"{side}_qb_epa"] = [v[0] for v in vals]
    g.loc[fut, f"{side}_backup_qb"] = [v[1] for v in vals]
    g.loc[fut, f"{side}_qb_proj"] = [v[2] for v in vals]
    # injuries: week 5 reports are real; later weeks are unknown, so assume a full roster
    later = fut & (g.week > 5)
    for p in POS:
        g.loc[later, f"{side}_out_{p}"] = 0

# typical weather by stadium and month (2015-2025): roof closed share, and average cold / wind penalties
g["cold"] = np.clip(45 - g.temp_f, 0, None)
hist = g[(g.season.between(2015, 2025)) & (g.neutral == 0)].copy()
hist["month"] = hist.gameday.dt.month
hist["wind15"] = np.clip(hist.wind_mph - 15, 0, None)
wx = hist.groupby(["home_team", "month"]).agg(dome=("dome", "mean"), cold=("cold", "mean"), wind15=("wind15", "mean"), wet=("wet", "mean"))
wx_team = hist.groupby("home_team").agg(dome=("dome", "mean"), cold=("cold", "mean"), wind15=("wind15", "mean"), wet=("wet", "mean"))
g["dome"] = g.dome.astype(float); g["wind15"] = g.wind15.astype(float); g["wet"] = g.wet.astype(float)
for i in g.index[fut & ~g.roof.isin(["dome"])]:
    key = (g.at[i, "home_team"], g.at[i, "gameday"].month)
    row = wx.loc[key] if key in wx.index else wx_team.loc[g.at[i, "home_team"]]
    if g.at[i, "neutral"] == 1:
        row = pd.Series({"dome": 0.3, "cold": 0.0, "wind15": 0.3, "wet": 0.05})
    g.at[i, "dome"] = row.dome; g.at[i, "cold"] = row.cold; g.at[i, "wind15"] = row.wind15; g.at[i, "wet"] = row.wet
g.loc[fut & (g.roof == "dome"), ["dome", "cold", "wind15", "wet"]] = [1, 0, 0, 0]
g["wx_estimated"] = fut & ~(g.roof == "dome")
# kickoff forecasts, when a weather run has saved them, replace the typical-weather estimates
fc_path = os.path.join(M, "forecasts.csv")
if os.path.exists(fc_path):
    fc = pd.read_csv(fc_path).set_index("game_id")
    for i in g.index[fut & g.game_id.isin(fc.index)]:
        f = fc.loc[g.at[i, "game_id"]]
        closed = int(f.roof_closed) == 1 or g.at[i, "roof"] == "dome"
        g.at[i, "dome"] = 1.0 if closed else 0.0
        g.at[i, "cold"] = 0.0 if closed else max(45 - float(f.temp_f), 0)
        g.at[i, "wind15"] = 0.0 if closed else max(float(f.wind_mph) - 15, 0)
        precip = f.get("precip", 0)                 # rain or snow expected at kickoff (column added Oct 18 2026; older files lack it)
        g.at[i, "wet"] = 0.0 if closed or pd.isna(precip) else float(int(precip) == 1)
        g.at[i, "wx_estimated"] = False
    print("forecasts applied:", int((fut & g.game_id.isin(fc.index)).sum()))

# ---------- features (same as final_model.py) ----------
g["home_net"] = g.home_ppg - g.home_papg; g["away_net"] = g.away_ppg - g.away_papg
g["home_field"] = 1 - g.neutral
g["home_field_2021"] = g.home_field * (g.season >= 2021)   # lets home field since 2021 differ from the older era
g["d_prev_off_snaps"] = g.home_prev_off_snaps - g.away_prev_off_snaps   # offensive snaps each team played last week
g["d_prev_def_snaps"] = g.home_prev_def_snaps - g.away_prev_def_snaps   # defensive snaps each team was on the field for last week
for p in POS: g[f"diff_out_{p}"] = g[f"home_out_{p}"] - g[f"away_out_{p}"]
g["home_exp_pts"] = (g.home_ppg + g.away_papg) / 2; g["away_exp_pts"] = (g.away_ppg + g.home_papg) / 2
g["backup_qbs"] = g.home_backup_qb + g.away_backup_qb
g["out_offense"] = sum(g[f"{s}_out_{p}"] for s in ("home", "away") for p in ("OL", "WR", "RB"))
g["out_defense"] = sum(g[f"{s}_out_{p}"] for s in ("home", "away") for p in ("DL", "LB", "DB"))
R = page["coef"]["returns_scale"]
for s in ("home", "away"):
    g[f"{s}_returns"] = 0.7 * (g[f"{s}_ko_ret_start"] - R["kr_m"]) / R["kr_s"] + 0.3 * (g[f"{s}_pr_avg"] - R["pr_m"]) / R["pr_s"]
d = lambda c: g[f"home_{c}"] - g[f"away_{c}"]
sm = lambda c: g[f"home_{c}"] + g[f"away_{c}"]
g["diff_returns"] = d("returns"); g["diff_sr_off"] = d("off_sr"); g["diff_sr_def"] = d("def_sr"); g["diff_qb"] = d("qb_epa")
g["diff_pass_off"] = d("off_pass_epa"); g["diff_pass_def"] = d("def_pass_epa")
g["sum_qb"] = sm("qb_epa"); g["sum_pass_off"] = sm("off_pass_epa"); g["sum_pass_def"] = sm("def_pass_epa")
g["sum_ex_off"] = sm("off_expl"); g["sum_ex_def"] = sm("def_expl"); g["sum_pace"] = sm("pace"); g["sum_pen"] = sm("pen_yds_pg")
FS = ["home_field", "home_field_2021", "home_net", "away_net", "rest_diff", "div_game", "home_backup_qb", "away_backup_qb", "diff_returns"] + \
     [f"diff_out_{p}" for p in POS] + ["diff_sr_off", "diff_sr_def", "diff_qb", "diff_pass_off", "diff_pass_def", "mkt_diff", "d_prev_off_snaps", "d_prev_def_snaps"]
FT = ["home_exp_pts", "away_exp_pts", "dome", "cold", "wind15", "backup_qbs", "out_offense", "out_defense", "div_game",
      "sum_qb", "sum_pass_off", "sum_pass_def", "sum_ex_off", "sum_ex_def", "sum_pace", "sum_pen", "wet"]
# played games keep their real kickoff weather (wind15 recomputed the same way)
g.loc[~fut, "wind15"] = np.clip(g.loc[~fut, "wind_mph"] - 15, 0, None)

train = g[(g.season <= 2025) & g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data & g.mkt_diff.notna()]
ms = Ridge(1.0).fit(train[FS], train.result); mt = Ridge(1.0).fit(train[FT], train.total)
tr = train.assign(rt=train.total - train.total_line)
mm = Ridge(50).fit(tr[FT + ["total_line"]], tr.rt)
SD_M = float((ms.predict(train[FS]) - train.result).std())

s26 = g[g.season == SEASON].copy()
assert s26[FS + FT].notna().all().all(), s26[FS + FT].isna().sum()[lambda x: x > 0]
s26["p_margin"] = ms.predict(s26[FS]); s26["p_total"] = mt.predict(s26[FT])
has_line = s26.total_line.notna()
s26.loc[has_line, "m_total"] = s26.loc[has_line, "total_line"] + mm.predict(s26.loc[has_line, FT + ["total_line"]])
s26["p_home_win"] = 1 - pd.Series(np.vectorize(lambda z: 0.5 * (1 + __import__("math").erf(z / np.sqrt(2))))(-s26.p_margin / SD_M), index=s26.index)

# ---------- grade played games ----------
pl = s26[s26.home_score.notna() & s26.spread_line.notna()].copy()
def rec(win, push): w = int((win & ~push).sum()); l = int((~win & ~push).sum()); return w, l
su = rec((pl.p_margin > 0) == (pl.result > 0), pl.result == 0)
ats = rec(np.where(pl.p_margin > pl.spread_line, pl.result > pl.spread_line, pl.result < pl.spread_line), pl.result == pl.spread_line)
ou = rec(np.where(pl.p_total > pl.total_line, pl.total > pl.total_line, pl.total < pl.total_line), pl.total == pl.total_line)
lean = pl[(pl.m_total - pl.total_line).abs() >= 1.5]
ou_lean = rec(np.where(lean.m_total > lean.total_line, lean.total > lean.total_line, lean.total < lean.total_line), lean.total == lean.total_line)
grade = {"games": len(pl), "su": su, "ats": ats, "ou": ou, "ou_lean": ou_lean,
         "mae_model": round(float((pl.p_margin - pl.result).abs().mean()), 2), "mae_vegas": round(float((pl.spread_line - pl.result).abs().mean()), 2),
         "mae_total_model": round(float((pl.p_total - pl.total).abs().mean()), 2), "mae_total_vegas": round(float((pl.total_line - pl.total).abs().mean()), 2),
         "through": f"{pl.gameday.max():%b %d}"}
print(json.dumps(grade))

# ---------- simulate the rest of the season ----------
DIV = {"AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"], "AFC South": ["HOU", "IND", "JAX", "TEN"],
       "AFC West": ["DEN", "KC", "LV", "LAC"], "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
       "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"]}
team_div = {t: dv for dv, ts in DIV.items() for t in ts}
teams = sorted(team_div)
assert set(teams) == set(s26.home_team), set(s26.home_team) ^ set(teams)
idx = {t: i for i, t in enumerate(teams)}
done26 = s26[s26.home_score.notna()]
w0 = np.zeros(len(teams)); t0 = np.zeros(len(teams)); l0 = np.zeros(len(teams))
for r in done26.itertuples():
    h, a = idx[r.home_team], idx[r.away_team]
    if r.result > 0: w0[h] += 1; l0[a] += 1
    elif r.result < 0: w0[a] += 1; l0[h] += 1
    else: t0[h] += 1; t0[a] += 1
rem = s26[s26.home_score.isna()]
hi_, ai_ = rem.home_team.map(idx).values, rem.away_team.map(idx).values
mu = rem.p_margin.values
N = 20000
rng = np.random.default_rng(26)
strength = rng.normal(0, 2.0, (N, len(teams)))          # uncertainty in how good each team really is, fixed for a whole season
margins = mu + strength[:, hi_] - strength[:, ai_] + rng.normal(0, SD_M, (N, len(mu)))
home_win = margins > 0
wins = np.tile(w0 + 0.5 * t0, (N, 1))
np.add.at(wins.T, hi_, home_win.T.astype(float)); np.add.at(wins.T, ai_, (~home_win).T.astype(float))
wins_t = wins + rng.uniform(0, 0.01, wins.shape)       # random tiebreaker (real NFL tiebreakers are not modeled)
div_win = np.zeros_like(wins, dtype=bool); playoff = np.zeros_like(div_win); seed1 = np.zeros_like(div_win)
for conf in ("AFC", "NFC"):
    divs = [dv for dv in DIV if dv.startswith(conf)]
    conf_idx = [idx[t] for dv in divs for t in DIV[dv]]
    winners = []
    for dv in divs:
        di = [idx[t] for t in DIV[dv]]
        best = np.array(di)[wins_t[:, di].argmax(1)]
        div_win[np.arange(N), best] = True; winners.append(best)
    winners = np.stack(winners, 1)
    playoff[np.arange(N)[:, None], winners] = True
    top = winners[np.arange(N), wins_t[np.arange(N)[:, None], winners].argmax(1)]
    seed1[np.arange(N), top] = True
    rest = wins_t[:, conf_idx].copy()
    for j, ti in enumerate(conf_idx):
        rest[div_win[:, ti], j] = -1
    wc = np.array(conf_idx)[np.argsort(-rest, 1)[:, :3]]
    playoff[np.arange(N)[:, None], wc] = True
proj = pd.DataFrame({"team": teams, "div": [team_div[t] for t in teams], "w": w0, "l": l0, "t": t0,
                     "exp_wins": wins.mean(0).round(1), "w10": np.percentile(wins, 10, axis=0), "w90": np.percentile(wins, 90, axis=0),
                     "p_div": div_win.mean(0).round(3), "p_playoffs": playoff.mean(0).round(3), "p_seed1": seed1.mean(0).round(3)})
print(proj.sort_values("exp_wins", ascending=False).head(10).to_string(index=False))
print("expected wins sum:", proj.exp_wins.sum(), "(should be 272)")
print("playoff odds sum:", proj.p_playoffs.sum().round(2), "(should be 14)")

cols = ["game_id", "week", "gameday", "gametime", "away_team", "home_team", "away_score", "home_score", "result", "total",
        "spread_line", "total_line", "p_margin", "p_total", "m_total", "p_home_win", "roof", "dome", "wx_estimated",
        "home_qb_epa", "away_qb_epa", "home_qb_name", "away_qb_name", "home_qb_proj", "away_qb_proj", "div_game", "home_backup_qb", "away_backup_qb"]
s26[cols].to_csv(os.path.join(M, "season_2026_games.csv"), index=False)
proj.to_csv(os.path.join(M, "season_2026_projection.csv"), index=False)
json.dump(grade, open(os.path.join(M, "season_2026_grade.json"), "w"))
