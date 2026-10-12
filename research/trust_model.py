"""Trust / avoid model: predicts how far a game's final margin will land from the Vegas spread (unpredictability),
tests which pregame factors actually forecast it, then scores every 2026 team.
Usage: python3 -I trust_model.py MODEL_DIR"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

M = sys.argv[1]
g = pd.read_csv(os.path.join(M, "games_features_plus.csv"), parse_dates=["gameday"], low_memory=False).sort_values(["gameday", "game_id"])
g["miss"] = (g.result - g.spread_line).abs()

# rolling pregame swings: each team's standard deviation of points scored, allowed and of its miss vs the line, last 8 games
rows = []
for side, sg in (("home", 1), ("away", -1)):
    x = g[["game_id", "gameday", f"{side}_team", f"{side}_score", f"{'away' if side == 'home' else 'home'}_score", "result", "spread_line"]].copy()
    x.columns = ["game_id", "gameday", "team", "pf", "pa", "result", "line"]
    x["ats"] = sg * (x.result - x.line)
    rows.append(x)
t = pd.concat(rows).sort_values(["team", "gameday"])
grp = t.groupby("team")
for c in ("pf", "pa", "ats"):
    t[f"sd8_{c}"] = grp[c].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
t["absats8"] = grp.ats.transform(lambda s: s.abs().shift(1).rolling(8, min_periods=4).mean())
t = t.set_index(["game_id", "team"])
for side in ("home", "away"):
    k = list(zip(g.game_id, g[f"{side}_team"]))
    for c in ("sd8_pf", "sd8_pa", "sd8_ats", "absats8"):
        g[f"{side}_{c}"] = t[c].reindex(k).values

sm = lambda c: g[f"home_{c}"] + g[f"away_{c}"]
g["abs_line"] = g.spread_line.abs()
g["backup_qbs"] = g.home_backup_qb + g.away_backup_qb
g["qb_floor"] = np.minimum(g.home_qb_epa, g.away_qb_epa)
g["giveaways"] = sm("giveaways_pg"); g["takeaways"] = sm("takeaways_pg")
g["explosive"] = sm("off_expl") + sm("def_expl")
g["sacks"] = sm("sack_rate_allowed")
g["injuries"] = sum(g[f"{s}_out_{p}"] for s in ("home", "away") for p in ("OL", "WR", "RB", "DL", "LB", "DB"))
g["swings"] = sm("sd8_pf") + sm("sd8_pa")
g["line_swings"] = sm("sd8_ats"); g["line_miss8"] = sm("absats8")
g["windy"] = (g.wind_mph.fillna(0) >= 15).astype(int) * (1 - g.dome)

BASE = ["total_line", "abs_line"]
GROUPS = {
    "Quarterback risk (backup starting, weakest QB rating)": ["backup_qbs", "qb_floor"],
    "Turnover-prone teams (giveaways and takeaways per game)": ["giveaways", "takeaways"],
    "Explosive-play teams (both sides)": ["explosive"],
    "Sack rate allowed": ["sacks"],
    "Starters out": ["injuries"],
    "Wind 15+ mph": ["windy"],
    "Recent swings in points (last 8 games)": ["swings"],
    "Recent misses vs the line (last 8 games)": ["line_swings", "line_miss8"],
}
allc = BASE + [c for v in GROUPS.values() for c in v]
d = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & (g.season >= 2016)].dropna(subset=allc).copy()

def cv(cols):
    pr = pd.Series(index=d.index, dtype=float)
    for s in range(2018, 2026):
        tr, te = d[d.season < s], d[d.season == s]
        pr[te.index] = Ridge(alpha=5).fit(tr[cols], tr.miss).predict(te[cols])
    return pr
test = d[d.season.between(2018, 2025)]
def score(pr):
    p = pr[test.index]
    q = pd.qcut(p, 5, labels=False)
    return np.corrcoef(p, test.miss)[0, 1], test.miss[q == 0].mean(), test.miss[q == 4].mean()
b = score(cv(BASE))
print(f"Games 2018-2025: {len(test)} | average miss vs the line {test.miss.mean():.2f} pts")
print(f"Base (total and spread size): corr {b[0]:.3f} | most predictable fifth {b[1]:.1f} pts | least predictable fifth {b[2]:.1f} pts")
res = []
for name, cols in GROUPS.items():
    c, lo, hi = score(cv(BASE + cols))
    res.append((name, c - b[0], lo, hi))
for name, gain, lo, hi in sorted(res, key=lambda r: -r[1]):
    print(f"  + {name:58s} corr gain {gain:+.3f} | fifths {lo:.1f} / {hi:.1f}")
keep = [c for name, cols in GROUPS.items() for c in cols if dict((r[0], r[1]) for r in res)[name] > 0.003]
fc = score(cv(BASE + keep))
print(f"Kept {keep}: corr {fc[0]:.3f} | most predictable fifth {fc[1]:.1f} pts | least predictable fifth {fc[2]:.1f} pts")

# does a team's predicted unpredictability early in the season line up with its real misses later?
mdl = Ridge(alpha=5).fit(d[d.season <= 2025][BASE + keep], d[d.season <= 2025].miss)
coef = dict(zip(BASE + keep, mdl.coef_.round(4)))
print("coefficients:", coef)

# ---------- 2026 team scores ----------
# team trust score = predicted miss for that team's games, holding the opponent and lines at league average,
# using the team's own current values for the team-level factors
cur = g[g.season == 2026]
last = {}
for side in ("home", "away"):
    for r in cur.sort_values("gameday").itertuples():
        team = getattr(r, f"{side}_team")
        last.setdefault(team, {})
        for c in ("qb_epa", "giveaways_pg", "takeaways_pg", "off_expl", "def_expl", "sack_rate_allowed", "sd8_pf", "sd8_pa", "sd8_ats", "absats8", "backup_qb"):
            v = getattr(r, f"{side}_{c}", np.nan)
            if not pd.isna(v): last[team][c] = v
page = json.load(open(os.path.join(M, "page_data.json")))
lgm = {c: np.nanmean([v.get(c, np.nan) for v in last.values()]) for c in ("qb_epa", "giveaways_pg", "takeaways_pg", "off_expl", "def_expl", "sack_rate_allowed", "sd8_pf", "sd8_pa", "sd8_ats", "absats8")}
out = []
for team, v in last.items():
    info = page["qbs"].get(team); q = next((x for x in info["qbs"] if x["id"] == info["recent"]), None) if info else None
    qb = q["epa"] if q else v.get("qb_epa", lgm["qb_epa"])
    f = lambda c: v.get(c, lgm[c])
    row = {"total_line": d.total_line.mean(), "abs_line": d.abs_line.mean(), "backup_qbs": 0, "qb_floor": min(qb, lgm["qb_epa"]),
           "giveaways": f("giveaways_pg") + lgm["giveaways_pg"], "takeaways": f("takeaways_pg") + lgm["takeaways_pg"],
           "explosive": f("off_expl") + f("def_expl") + lgm["off_expl"] + lgm["def_expl"], "sacks": f("sack_rate_allowed") + lgm["sack_rate_allowed"],
           "injuries": 2.0, "windy": 0, "swings": f("sd8_pf") + f("sd8_pa") + lgm["sd8_pf"] + lgm["sd8_pa"],
           "line_swings": f("sd8_ats") + lgm["sd8_ats"], "line_miss8": f("absats8") + lgm["absats8"]}
    pred = mdl.intercept_ + sum(coef[c] * row[c] for c in BASE + keep)
    out.append({"team": team, "pred_miss": round(float(pred), 2), "qb": q["name"] if q else "", "qb_epa": qb})
o = pd.DataFrame(out).sort_values("pred_miss")
mu, sdv = o.pred_miss.mean(), o.pred_miss.std()
o["z"] = ((o.pred_miss - mu) / sdv).round(2)
o["tier"] = np.where(o.z <= -1.0, "Trust", np.where(o.z <= -0.4, "Lean trust", np.where(o.z >= 1.0, "Avoid", np.where(o.z >= 0.4, "Lean avoid", "Neutral"))))
print(o.to_string(index=False))
o.to_csv(os.path.join(M, "trust_2026.csv"), index=False)
json.dump({"coef": coef, "intercept": float(mdl.intercept_), "kept": keep, "corr_base": round(float(b[0]), 3), "corr_final": round(float(fc[0]), 3),
           "fifths": [round(float(fc[1]), 1), round(float(fc[2]), 1)], "avg_miss": round(float(test.miss.mean()), 2)}, open(os.path.join(M, "trust_model.json"), "w"))
