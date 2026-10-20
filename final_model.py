"""Final model: base + the factors that earned their place. Walk-forward backtest, then fit for the calculator.
Usage: python3 -I final_model.py FEATURES_ALL_CSV OUT_DIR"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

g = pd.read_csv(sys.argv[1], parse_dates=["gameday"], low_memory=False)
OUT = sys.argv[2]
POS = ["OL", "WR", "RB", "DL", "LB", "DB"]
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
g["cold"] = np.clip(45 - g.temp_f, 0, None)
both = lambda c: pd.concat([g[f"home_{c}"], g[f"away_{c}"]])
RET = {"kr_m": both("ko_ret_start").mean(), "kr_s": both("ko_ret_start").std(), "pr_m": both("pr_avg").mean(), "pr_s": both("pr_avg").std()}
for s in ("home", "away"):
    g[f"{s}_returns"] = 0.7 * (g[f"{s}_ko_ret_start"] - RET["kr_m"]) / RET["kr_s"] + 0.3 * (g[f"{s}_pr_avg"] - RET["pr_m"]) / RET["pr_s"]
d = lambda c: g[f"home_{c}"] - g[f"away_{c}"]
sm = lambda c: g[f"home_{c}"] + g[f"away_{c}"]
g["diff_returns"] = d("returns")
g["diff_sr_off"] = d("off_sr"); g["diff_sr_def"] = d("def_sr")
g["diff_qb"] = d("qb_epa")
g["diff_pass_off"] = d("off_pass_epa"); g["diff_pass_def"] = d("def_pass_epa")
g["sum_qb"] = sm("qb_epa"); g["sum_pass_off"] = sm("off_pass_epa"); g["sum_pass_def"] = sm("def_pass_epa")
g["sum_ex_off"] = sm("off_expl"); g["sum_ex_def"] = sm("def_expl")
g["sum_pace"] = sm("pace"); g["sum_pen"] = sm("pen_yds_pg")

BASE_S = ["home_field", "home_field_2021", "home_net", "away_net", "rest_diff", "div_game", "home_backup_qb", "away_backup_qb", "diff_returns"] + [f"diff_out_{p}" for p in POS]
BASE_T = ["home_exp_pts", "away_exp_pts", "dome", "cold", "wind15", "backup_qbs", "out_offense", "out_defense", "div_game"]
FINAL_S = BASE_S + ["diff_sr_off", "diff_sr_def", "diff_qb", "diff_pass_off", "diff_pass_def", "mkt_diff", "d_prev_off_snaps", "d_prev_def_snaps"]
FINAL_T = BASE_T + ["sum_qb", "sum_pass_off", "sum_pass_def", "sum_ex_off", "sum_ex_def", "sum_pace", "sum_pen", "wet"]   # wet = rain or snow at kickoff (Oct 18 2026)
done = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data & g.mkt_diff.notna()].copy()

def ats(pred, line, actual):
    pick = pred > line; cover = actual - line; push = cover == 0
    win = np.where(pick, cover > 0, cover < 0)[~push]
    return int(win.sum()), int((~win).sum())
rec = lambda w, l: f"{w}-{l} ({w/(w+l):.1%})" if w + l else "n/a"

preds = []
for season in range(2018, 2026):
    tr, te = done[done.season < season], done[done.season == season].copy()
    for tag, sx, tx in (("base", BASE_S, BASE_T), ("final", FINAL_S, FINAL_T)):
        te[f"m_{tag}"] = Ridge(1.0).fit(tr[sx], tr.result).predict(te[sx])
        te[f"t_{tag}"] = Ridge(1.0).fit(tr[tx], tr.total).predict(te[tx])
        trr = tr.assign(rt=tr.total - tr.total_line, rs=tr.result - tr.spread_line)
        te[f"tm_{tag}"] = te.total_line + Ridge(50).fit(trr[tx + ["total_line"]], trr.rt).predict(te[tx + ["total_line"]])
        te[f"sm_{tag}"] = te.spread_line + Ridge(50).fit(trr[sx], trr.rs).predict(te[sx])
    preds.append(te)
P = pd.concat(preds)
rows = []
for label, col, line, act, edges in [("Spread, stats only", "m", "spread_line", "result", (0, 2, 3, 4, 5)),
                                     ("Spread, Vegas + adj", "sm", "spread_line", "result", (0, 0.5, 1, 1.5, 2)),
                                     ("Total, stats only", "t", "total_line", "total", (0, 2, 3, 4, 5)),
                                     ("Total, Vegas + adj", "tm", "total_line", "total", (0, 1, 1.5, 2, 3))]:
    for tag in ("base", "final"):
        c = f"{col}_{tag}"
        r = {"model": label, "version": "before" if tag == "base" else "after", "MAE": round((P[c] - P[act]).abs().mean(), 2),
             "Vegas MAE": round((P[line] - P[act]).abs().mean(), 2)}
        for i, e in enumerate(edges):
            sub = P[(P[c] - P[line]).abs() >= e]
            r[f"e{i}"] = f"{e}+: " + rec(*ats(sub[c].values, sub[line].values, sub[act].values))
        rows.append(r)
bt = pd.DataFrame(rows)
pd.set_option("display.width", 260)
print(bt.to_string(index=False))

# per-season stability of the best candidate
print("\nVegas + adj total, edge 1.5+, by season:")
for s_, grp in P.groupby("season"):
    sub = grp[(grp.tm_final - grp.total_line).abs() >= 1.5]
    print(s_, rec(*ats(sub.tm_final.values, sub.total_line.values, sub.total.values)))

fs = Ridge(1.0).fit(done[FINAL_S], done.result); ft = Ridge(1.0).fit(done[FINAL_T], done.total)
dm = done.assign(rt=done.total - done.total_line)
fm = Ridge(50).fit(dm[FINAL_T + ["total_line"]], dm.rt)
sd = {c: float(done[c].std()) for c in FINAL_S + FINAL_T}
coef = {"spread": dict(zip(FINAL_S, fs.coef_)), "spread_intercept": float(fs.intercept_),
        "total": dict(zip(FINAL_T, ft.coef_)), "total_intercept": float(ft.intercept_),
        "total_mkt": dict(zip(FINAL_T + ["total_line"], fm.coef_)), "total_mkt_intercept": float(fm.intercept_),
        "resid_sd_margin": float((fs.predict(done[FINAL_S]) - done.result).std()),
        "resid_sd_total": float((ft.predict(done[FINAL_T]) - done.total).std()), "returns_scale": RET,
        "sd": sd}
json.dump(coef, open(os.path.join(OUT, "coefficients_final.json"), "w"), indent=1)
bt.to_csv(os.path.join(OUT, "backtest_summary_final.csv"), index=False)
P.to_csv(os.path.join(OUT, "backtest_games_final.csv"), index=False)
print("\nPoints per 1 SD gap, final spread model:")
for c in FINAL_S: print(f"  {c:16s} {coef['spread'][c]*sd[c]:+.2f}")
print("Points per 1 SD, final total model:")
for c in FINAL_T: print(f"  {c:16s} {coef['total'][c]*sd[c]:+.2f}")
