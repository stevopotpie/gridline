"""Rank candidate factors by what they add on unseen seasons (walk forward 2018-2025).
Usage: python3 -I rank_factors.py FEATURES_ALL_CSV OUT_DIR"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

g = pd.read_csv(sys.argv[1], parse_dates=["gameday"], low_memory=False)
OUT = sys.argv[2]
POS = ["OL", "WR", "RB", "DL", "LB", "DB"]
g["home_net"] = g.home_ppg - g.home_papg; g["away_net"] = g.away_ppg - g.away_papg
g["home_field"] = 1 - g.neutral
for p in POS: g[f"diff_out_{p}"] = g[f"home_out_{p}"] - g[f"away_out_{p}"]
g["home_exp_pts"] = (g.home_ppg + g.away_papg) / 2; g["away_exp_pts"] = (g.away_ppg + g.home_papg) / 2
g["backup_qbs"] = g.home_backup_qb + g.away_backup_qb
g["out_offense"] = sum(g[f"{s}_out_{p}"] for s in ("home", "away") for p in ("OL", "WR", "RB"))
g["out_defense"] = sum(g[f"{s}_out_{p}"] for s in ("home", "away") for p in ("DL", "LB", "DB"))
g["cold"] = np.clip(45 - g.temp_f, 0, None)

# Return units: one rating, kick returns weighted 70%, punt returns 30%, on standardized scales
both = lambda c: pd.concat([g[f"home_{c}"], g[f"away_{c}"]])
KR_M, KR_S, PR_M, PR_S = both("ko_ret_start").mean(), both("ko_ret_start").std(), both("pr_avg").mean(), both("pr_avg").std()
for s in ("home", "away"):
    g[f"{s}_returns"] = 0.7 * (g[f"{s}_ko_ret_start"] - KR_M) / KR_S + 0.3 * (g[f"{s}_pr_avg"] - PR_M) / PR_S
g["diff_returns"] = g.home_returns - g.away_returns

def d(c): return g[f"home_{c}"] - g[f"away_{c}"]
def s(c): return g[f"home_{c}"] + g[f"away_{c}"]
# spread features are home minus away; for defense stats lower is better, so "net" = offense minus defense allowed
GROUPS_S = {
    "Pass efficiency (EPA/dropback)": {"x_pass_off": d("off_pass_epa"), "x_pass_def": d("def_pass_epa")},
    "Run efficiency (EPA/rush)": {"x_rush_off": d("off_rush_epa"), "x_rush_def": d("def_rush_epa")},
    "Success rate": {"x_sr_off": d("off_sr"), "x_sr_def": d("def_sr")},
    "Starting QB rating (EPA/dropback)": {"x_qb": d("qb_epa")},
    "Explosive plays": {"x_ex_off": d("off_expl"), "x_ex_def": d("def_expl")},
    "Pass rush (sacks, QB hits)": {"x_sk_allow": d("sack_rate_allowed"), "x_sk_made": d("sack_rate_made"), "x_hit": d("hit_rate_made")},
    "Turnovers": {"x_give": d("giveaways_pg"), "x_take": d("takeaways_pg")},
    "Red zone TD rate": {"x_rz_off": d("rz_td_off"), "x_rz_def": d("rz_td_def")},
    "Penalty yards": {"x_pen": d("pen_yds_pg")},
    "Travel and time zones": {"x_tz": g.tz_shift_away, "x_west_early": g.west_early},
    "Bye weeks and short weeks": {"x_bye_h": g.home_bye, "x_bye_a": g.away_bye, "x_sw_h": g.short_week_home, "x_sw_a": g.short_week_away},
    "Denver altitude": {"x_alt": g.altitude},
}
GROUPS_T = {
    "Pass efficiency (EPA/dropback)": {"t_pass_off": s("off_pass_epa"), "t_pass_def": s("def_pass_epa")},
    "Run efficiency (EPA/rush)": {"t_rush_off": s("off_rush_epa"), "t_rush_def": s("def_rush_epa")},
    "Success rate": {"t_sr_off": s("off_sr"), "t_sr_def": s("def_sr")},
    "Starting QB rating (EPA/dropback)": {"t_qb": s("qb_epa")},
    "Explosive plays": {"t_ex_off": s("off_expl"), "t_ex_def": s("def_expl")},
    "Pass rush (sacks, QB hits)": {"t_sk_allow": s("sack_rate_allowed"), "t_sk_made": s("sack_rate_made"), "t_hit": s("hit_rate_made")},
    "Turnovers": {"t_give": s("giveaways_pg"), "t_take": s("takeaways_pg")},
    "Red zone TD rate": {"t_rz_off": s("rz_td_off"), "t_rz_def": s("rz_td_def")},
    "Penalty yards": {"t_pen": s("pen_yds_pg")},
    "Pace (sec per snap, neutral)": {"t_pace": s("pace")},
    "Pass rate over expected": {"t_proe": s("proe")},
    "Denver altitude": {"t_alt": g.altitude},
}
for grp in list(GROUPS_S.values()) + list(GROUPS_T.values()):
    for k, v in grp.items(): g[k] = v

BASE_S = ["home_field", "home_net", "away_net", "rest_diff", "div_game", "home_backup_qb", "away_backup_qb", "diff_returns"] + [f"diff_out_{p}" for p in POS]
BASE_T = ["home_exp_pts", "away_exp_pts", "dome", "cold", "wind15", "backup_qbs", "out_offense", "out_defense", "div_game"]
done = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data].copy()
model = lambda: make_pipeline(StandardScaler(), Ridge(alpha=5.0))

def walk(xcols, target, line):
    """Out-of-sample predictions of the outcome and of the miss vs the Vegas line."""
    out = []
    for season in range(2018, 2026):
        tr, te = done[done.season < season], done[done.season == season]
        pred = model().fit(tr[xcols], tr[target]).predict(te[xcols])
        res = model().fit(tr[xcols], tr[target] - tr[line]).predict(te[xcols])
        out.append(pd.DataFrame({"pred": pred, "res": res, "y": te[target].values, "line": te[line].values}, index=te.index))
    return pd.concat(out)

def record(df, frac=0.25):
    """Bet the side the line-miss model favors on its most confident quarter of games."""
    cut = df.res.abs().quantile(1 - frac); s = df[df.res.abs() >= cut]
    cover = s.y - s.line; push = cover == 0
    win = np.where(s.res > 0, cover > 0, cover < 0)[~push]
    return int(win.sum()), int((~win).sum())

rows = []
for kind, BASE, GROUPS, target, line in (("Spread", BASE_S, GROUPS_S, "result", "spread_line"), ("Total", BASE_T, GROUPS_T, "total", "total_line")):
    base = walk(BASE, target, line)
    b_mae = (base.pred - base.y).abs().mean()
    b_corr = np.corrcoef(base.res, base.y - base.line)[0, 1]
    bw, bl = record(base)
    rows.append({"bet": kind, "factor": "Base model (PPG, injuries, QB flag, weather, returns)", "MAE": round(b_mae, 3), "MAE gain": 0.0,
                 "line-miss corr": round(b_corr, 3), "top 25% record": f"{bw}-{bl} ({bw/(bw+bl):.1%})", "pts per SD": ""})
    full_x = BASE + [c for grp in GROUPS.values() for c in grp]
    sc = StandardScaler().fit(done[full_x])
    full = Ridge(alpha=5.0).fit(sc.transform(done[full_x]), done[target])
    impact = dict(zip(full_x, full.coef_))
    for name, grp in GROUPS.items():
        r = walk(BASE + list(grp), target, line)
        mae = (r.pred - r.y).abs().mean()
        corr = np.corrcoef(r.res, r.y - r.line)[0, 1]
        w, l = record(r)
        pts = " / ".join(f"{impact[c]:+.2f}" for c in grp)
        rows.append({"bet": kind, "factor": name, "MAE": round(mae, 3), "MAE gain": round(b_mae - mae, 3),
                     "line-miss corr": round(corr, 3), "top 25% record": f"{w}-{l} ({w/(w+l):.1%})", "pts per SD": pts})
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
for kind in ("Spread", "Total"):
    print(R[R.bet == kind].sort_values("MAE gain", ascending=False).to_string(index=False)); print()
R.to_csv(os.path.join(OUT, "factor_ranking.csv"), index=False)
print("games used:", len(done))
