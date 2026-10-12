"""Walk-forward test (2018-2025) of each candidate improvement against the current final model.
Usage: python3 -I experiments.py FEATURES_PLUS_CSV FINAL_MODEL_PY"""
import sys, re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

g = pd.read_csv(sys.argv[1], parse_dates=["gameday"], low_memory=False)
src = open(sys.argv[2]).read()
block = src[src.index('POS = ["OL"'):src.index("done = g[")]       # reuse the exact feature engineering of final_model.py
exec(block)
d = lambda c: g[f"home_{c}"] - g[f"away_{c}"]
sm = lambda c: g[f"home_{c}"] + g[f"away_{c}"]
for pre in ("gf", "oa"):
    for k in ("off_epa", "def_epa", "off_sr", "def_sr", "off_pass", "def_pass"):
        g[f"{pre}_d_{k}"] = d(f"{pre}_{k}"); g[f"{pre}_s_{k}"] = sm(f"{pre}_{k}")
for k in ("prev_div", "prev_shootout", "prev_ot", "prev_close", "prev_def_snaps", "prev_off_snaps", "prev_to_margin", "prev_total"):
    g[f"d_{k}"] = d(k); g[f"s_{k}"] = sm(k)
g["div_x_net"] = g.div_game * (g.home_net - g.away_net)
g["div_x_mkt"] = g.div_game * g.mkt_diff.fillna(0)

OLD_S = ["diff_sr_off", "diff_sr_def", "diff_pass_off", "diff_pass_def"]
OLD_T = ["sum_pass_off", "sum_pass_def"]
def swap_s(pre): return [c for c in FINAL_S if c not in OLD_S] + [f"{pre}_d_{k}" for k in ("off_epa", "def_epa", "off_sr", "def_sr", "off_pass", "def_pass")]
def swap_t(pre): return [c for c in FINAL_T if c not in OLD_T] + [f"{pre}_s_{k}" for k in ("off_epa", "def_epa", "off_pass", "def_pass")]
TESTS = {
    "Current model": (FINAL_S, FINAL_T),
    "Garbage time removed": (swap_s("gf"), swap_t("gf")),
    "Garbage time removed + opponent-adjusted": (swap_s("oa"), swap_t("oa")),
    "Market power ratings (earlier weeks' lines)": (FINAL_S + ["mkt_diff"], FINAL_T + ["mkt_total"]),
    "Referee crew": (FINAL_S, FINAL_T + ["ref_ou", "ref_pen"]),
    "Prev week: division game, shootout, OT, close": (FINAL_S + ["d_prev_div", "d_prev_shootout", "d_prev_ot", "d_prev_close"],
                                                     FINAL_T + ["s_prev_div", "s_prev_shootout", "s_prev_ot", "s_prev_close", "s_prev_total"]),
    "Prev week: offensive and defensive snaps": (FINAL_S + ["d_prev_off_snaps", "d_prev_def_snaps"], FINAL_T + ["s_prev_off_snaps", "s_prev_def_snaps"]),
    "Prev week: turnover margin": (FINAL_S + ["d_prev_to_margin"], FINAL_T + ["s_prev_to_margin"]),
    "Division game, both teams decent": (FINAL_S + ["div_good", "div_x_net"], FINAL_T + ["div_good"]),
}
need = set(c for s_, t_ in TESTS.values() for c in s_ + t_)
done = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data & g.mkt_diff.notna()].copy()
done = done.dropna(subset=list(need))

def ats(pred, line, actual):
    pick = pred > line; cover = actual - line; push = cover == 0
    win = np.where(pick, cover > 0, cover < 0)[~push]
    return int(win.sum()), int((~win).sum())
def rec(w, l): return f"{w}-{l} {w/(w+l):.1%}" if w + l else "n/a"

def run(sx, tx):
    out = []
    for season in range(2018, 2026):
        tr, te = done[done.season < season], done[done.season == season].copy()
        te["m"] = Ridge(1.0).fit(tr[sx], tr.result).predict(te[sx])
        te["t"] = Ridge(1.0).fit(tr[tx], tr.total).predict(te[tx])
        trr = tr.assign(rs=tr.result - tr.spread_line, rt=tr.total - tr.total_line)
        te["sm"] = te.spread_line + Ridge(50).fit(trr[sx], trr.rs).predict(te[sx])
        te["tm"] = te.total_line + Ridge(50).fit(trr[tx + ["total_line"]], trr.rt).predict(te[tx + ["total_line"]])
        out.append(te)
    return pd.concat(out)

rows = []
for name, (sx, tx) in TESTS.items():
    P = run(sx, tx)
    e3 = P[(P.m - P.spread_line).abs() >= 3]; e1 = P[(P.sm - P.spread_line).abs() >= 1]; t15 = P[(P.tm - P.total_line).abs() >= 1.5]
    rows.append({"test": name, "margin miss": round((P.m - P.result).abs().mean(), 3), "total miss": round((P.t - P.total).abs().mean(), 3),
                 "ATS, 3+ edge": rec(*ats(e3.m.values, e3.spread_line.values, e3.result.values)),
                 "ATS, Vegas+adj 1+": rec(*ats(e1.sm.values, e1.spread_line.values, e1.result.values)),
                 "Totals, Vegas+adj 1.5+": rec(*ats(t15.tm.values, t15.total_line.values, t15.total.values))})
R = pd.DataFrame(rows)
base = R.iloc[0]
R["margin gain"] = (base["margin miss"] - R["margin miss"]).round(3); R["total gain"] = (base["total miss"] - R["total miss"]).round(3)
pd.set_option("display.width", 260)
print("games:", len(done), "| Vegas margin miss", round((done[done.season >= 2018].spread_line - done[done.season >= 2018].result).abs().mean(), 3),
      "| Vegas total miss", round((done[done.season >= 2018].total_line - done[done.season >= 2018].total).abs().mean(), 3))
print(R[["test", "margin miss", "margin gain", "ATS, 3+ edge", "ATS, Vegas+adj 1+", "total miss", "total gain", "Totals, Vegas+adj 1.5+"]].to_string(index=False))
R.to_csv("experiment_results.csv", index=False)
