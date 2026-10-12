"""Fit the live comeback model for the calculator (no live win probability needed as input) and rank 2026 teams.
Usage: python3 -I comeback_model.py MODEL_DIR"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

M = sys.argv[1]
s = pd.read_csv(os.path.join(M, "comeback_situations.csv"))
s["secs2"] = (s.secs / 1800) ** 2; s["under10"] = (s.secs < 600).astype(int); s["half"] = (s.secs >= 1750).astype(int)
X = ["deficit", "secs", "secs2", "under10", "half", "has_ball", "trail_spread", "trail_home", "trail_qb_epa", "trail_off_pass_epa", "trail_off_expl",
     "lead_def_pass_epa", "lead_off_pass_epa", "trail_contender", "trail_out_of_it"]
s["secs_left"] = s.secs
mu, sd = s[X].mean(), s[X].std()
Z = (s[X] - mu) / sd

# leave-one-season-out check against the live win probability model (vegas_wp) as a benchmark
pw, pj = pd.Series(index=s.index, dtype=float), pd.Series(index=s.index, dtype=float)
for yr in s.season.unique():
    tr, te = s.season != yr, s.season == yr
    pw[te] = LogisticRegression(C=0.5, max_iter=3000).fit(Z[tr], s.won[tr]).predict_proba(Z[te])[:, 1]
    pj[te] = LogisticRegression(C=0.5, max_iter=3000).fit(Z[tr], s.within_one[tr]).predict_proba(Z[te])[:, 1]
print("log loss, comeback win: model", round(log_loss(s.won, pw), 4), "| live win prob model", round(log_loss(s.won, s.vwp), 4),
      "| always the average", round(log_loss(s.won, np.full(len(s), s.won.mean())), 4))
print("log loss, within one score: model", round(log_loss(s.within_one, pj), 4),
      "| always the average", round(log_loss(s.within_one, np.full(len(s), s.within_one.mean())), 4))
# calibration of the within-one model by predicted bucket
cal = pd.DataFrame({"p": pj, "y": s.within_one}).assign(b=lambda d: pd.cut(d.p, [0, .05, .1, .2, .3, .45, 1]))
print(cal.groupby("b", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3))

mw = LogisticRegression(C=0.5, max_iter=3000).fit(Z, s.won)
mj = LogisticRegression(C=0.5, max_iter=3000).fit(Z, s.within_one)
mp = Ridge(alpha=5).fit(Z, s.trail_pts_after)
out = {"x": X, "mu": mu.round(5).to_dict(), "sd": sd.round(5).to_dict(),
       "won": {"b": float(mw.intercept_[0]), "w": dict(zip(X, mw.coef_[0].round(4)))},
       "within": {"b": float(mj.intercept_[0]), "w": dict(zip(X, mj.coef_[0].round(4)))},
       "pts": {"b": float(mp.intercept_), "w": dict(zip(X, mp.coef_.round(4)))},
       "n": int(len(s)), "n_won": int(s.won.sum()), "n_within": int(s.within_one.sum())}
for k in ("won", "within", "pts"):
    print(k, {c: round(v, 2) for c, v in out[k]["w"].items()})

# 2026 comeback profiles: down 17 at the start of the third quarter, neutral site and line, average opponent, 50/50 possession
cand = pd.read_csv(os.path.join(M, "cand_form.csv")).set_index("team")
page = json.load(open(os.path.join(M, "page_data.json")))
proj = pd.read_csv(os.path.join(M, "season_2026_projection.csv")).set_index("team")
week = int(page.get("week") or 1)
rows = []
lg = cand.loc[list(proj.index)]
for t in proj.index:
    info = page["qbs"][t]; q = next(x for x in info["qbs"] if x["id"] == info["recent"])
    gp = proj.at[t, "w"] + proj.at[t, "l"] + proj.at[t, "t"]; wp = (proj.at[t, "w"] + 0.5 * proj.at[t, "t"]) / gp if gp else 0.5
    v = {"deficit": 17, "secs": 1800, "secs2": 1, "under10": 0, "half": 1, "has_ball": 0.5, "trail_spread": 0, "trail_home": 0.5, "trail_qb_epa": q["epa"],
         "trail_off_pass_epa": cand.at[t, "off_pass_epa"], "trail_off_expl": cand.at[t, "off_expl"],
         "lead_def_pass_epa": lg.def_pass_epa.mean(), "lead_off_pass_epa": lg.off_pass_epa.mean(),
         "trail_contender": int(wp >= 0.5 and week >= 8), "trail_out_of_it": int(wp <= 0.3 and week >= 10)}
    z = np.array([(v[c] - mu[c]) / sd[c] for c in X])
    f = lambda m: 1 / (1 + np.exp(-(m["b"] + sum(m["w"][c] * zz for c, zz in zip(X, z)))))
    rows.append({"team": t, "qb": q["name"], "qb_epa": q["epa"], "within": round(float(f(out["within"])), 3), "won": round(float(f(out["won"])), 3),
                 "pts": round(float(out["pts"]["b"] + sum(out["pts"]["w"][c] * zz for c, zz in zip(X, z))), 1),
                 "def_within": None})
# how often each team, as the LEADER, lets an average trailing team back within one score
for r in rows:
    t = r["team"]
    v = {"deficit": 17, "secs": 1800, "secs2": 1, "under10": 0, "half": 1, "has_ball": 0.5, "trail_spread": 0, "trail_home": 0.5, "trail_qb_epa": float(np.median([x["qb_epa"] for x in rows])),
         "trail_off_pass_epa": lg.off_pass_epa.mean(), "trail_off_expl": lg.off_expl.mean(),
         "lead_def_pass_epa": cand.at[t, "def_pass_epa"], "lead_off_pass_epa": cand.at[t, "off_pass_epa"], "trail_contender": 0, "trail_out_of_it": 0}
    z = np.array([(v[c] - mu[c]) / sd[c] for c in X])
    r["def_within"] = round(float(1 / (1 + np.exp(-(out["within"]["b"] + sum(out["within"]["w"][c] * zz for c, zz in zip(X, z)))))), 3)
rank = pd.DataFrame(rows).sort_values("within", ascending=False)
print(rank.to_string(index=False))
out["rank"] = rank.to_dict("records"); out["week"] = week
out["base"] = pd.read_csv(os.path.join(M, "comeback_base_rates.csv")).to_dict("records")
json.dump(out, open(os.path.join(M, "comeback_model.json"), "w"))
