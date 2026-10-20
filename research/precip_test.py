"""Test: does rain or snow at kickoff (pbp weather text) help the totals model?"""
import sys, glob, os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
W = sys.argv[1]; D = os.path.join(W, "data")
w = pd.concat([pd.read_parquet(f, columns=["game_id", "weather"]).drop_duplicates("game_id") for f in sorted(glob.glob(os.path.join(D, "pbp_20*.parquet")))])
s = w.weather.fillna("").str.lower()
wet = s.str.contains("rain|shower|drizzle|storm|snow|flurr|sleet") & ~s.str.contains("chance|forecast|no rain")
w["rain"] = (wet & ~s.str.contains("snow|flurr|sleet")).astype(int)
w["snow"] = (wet & s.str.contains("snow|flurr|sleet")).astype(int)
w["heavy"] = (wet & s.str.contains("heavy|steady|storm|downpour")).astype(int)
src = open(os.path.join(W, "final_model.py")).read()
exec(src.split("g = pd.read_csv(sys.argv[1]")[0])
g = pd.read_csv(os.path.join(W, "games_features_plus.csv"), parse_dates=["gameday"], low_memory=False)
exec(src.split('OUT = sys.argv[2]\n')[1].split("done = g[")[0])
g = g.merge(w[["game_id", "rain", "snow", "heavy"]], on="game_id", how="left").fillna({"rain": 0, "snow": 0, "heavy": 0})
g.loc[g.dome == 1, ["rain", "snow", "heavy"]] = 0
g["wet"] = g[["rain", "snow"]].max(axis=1)
g["wet_wind"] = g.wet * np.clip(g.wind_mph - 10, 0, None)
done = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & g.st_data & g.cand_data & g.mkt_diff.notna()].copy()
d = done[(done.season >= 2015) & (done.dome == 0) & (done.total != done.total_line)].assign(diff=lambda x: x.total - x.total_line)
print("outdoor games:", len(d))
for name, m in [("dry", d.wet == 0), ("rain", d.rain == 1), ("snow", d.snow == 1), ("heavy rain/snow", d.heavy == 1),
                ("wet + wind 10+", (d.wet == 1) & (d.wind_mph >= 10)), ("wet + wind 15+", (d.wet == 1) & (d.wind_mph >= 15)), ("dry + wind 15+", (d.wet == 0) & (d.wind_mph >= 15))]:
    x = d[m]; print(f"  {name:16s} n={len(x):4d} avg total {x.total.mean():5.1f} line {x.total_line.mean():5.1f} vs line {x['diff'].mean():+5.1f} under {(x['diff'] < 0).mean():.1%}")
def ats(pred, line, act):
    pick = pred > line; cv = act - line; push = cv == 0
    ww = np.where(pick, cv > 0, cv < 0)[~push]; return int(ww.sum()), int((~ww).sum())
for name, tx in [("current", FINAL_T), ("+ rain, snow", FINAL_T + ["rain", "snow"]), ("+ wet", FINAL_T + ["wet"]), ("+ wet, wet x wind", FINAL_T + ["wet", "wet_wind"])]:
    out = []
    for season in range(2018, 2026):
        tr, te = done[done.season < season], done[done.season == season].copy()
        trr = tr.assign(rt=tr.total - tr.total_line)
        mm = Ridge(50).fit(trr[tx + ["total_line"]], trr.rt)
        te["tm"] = te.total_line + mm.predict(te[tx + ["total_line"]]); out.append(te)
    P = pd.concat(out); res = {}
    for e in (1, 1.5, 2):
        s_ = P[(P.tm - P.total_line).abs() >= e]; a = ats(s_.tm.values, s_.total_line.values, s_.total.values); res[e] = f"{a[0]}-{a[1]} ({a[0]/sum(a):.1%})"
    wetp = P[P.wet == 1]; a = ats(wetp.tm.values, wetp.total_line.values, wetp.total.values)
    print(f"{name:18s} MAE {(P.tm - P.total).abs().mean():.3f} | 1+ {res[1]} | 1.5+ {res[1.5]} | 2+ {res[2]} | wet games {a[0]}-{a[1]}")
dm = done.assign(rt=done.total - done.total_line)
tx = FINAL_T + ["wet", "wet_wind"]; mm = Ridge(50).fit(dm[tx + ["total_line"]], dm.rt)
print("Vegas+adj points: wet", round(dict(zip(tx, mm.coef_))["wet"], 2), "| per mph over 10 when wet", round(dict(zip(tx, mm.coef_))["wet_wind"], 2))
