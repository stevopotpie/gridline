"""Blowout repeat offenders: teams that keep losing by 17+. Tests what the flag means for the next game.
Usage: python3 -I blowouts.py MODEL_DIR"""
import sys, os, json
import numpy as np, pandas as pd
M = sys.argv[1]
g = pd.read_csv(os.path.join(M, "games_features_all.csv"), parse_dates=["gameday"], low_memory=False)
g = g[g.game_type == "REG"]
rows = []
for side, opp in (("home", "away"), ("away", "home")):
    x = g[["game_id", "season", "week", "gameday", f"{side}_team", f"{opp}_team", "result", "total", "spread_line", "total_line"]].copy()
    x.columns = ["game_id", "season", "week", "gameday", "team", "opp", "result", "total", "spread_line", "total_line"]
    sgn = 1 if side == "home" else -1
    x["margin"] = sgn * x.result                  # team's margin
    x["line"] = sgn * x.spread_line               # + = team favored by that much
    rows.append(x)
t = pd.concat(rows).sort_values(["team", "gameday"]).reset_index(drop=True)
t["blowout_loss"] = (t.margin <= -17).astype(float).where(t.margin.notna())
t["blowout_win"] = (t.margin >= 17).astype(float).where(t.margin.notna())
grp = t.groupby(["team", "season"])
t["bl_season"] = grp.blowout_loss.transform(lambda s: s.shift(1).fillna(0).cumsum())          # entering this game
t["bl_last5"] = grp.blowout_loss.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
t["gp"] = grp.cumcount()
t["flag"] = ((t.bl_last5 >= 2) | (t.bl_season >= 3)).astype(int)

h = t[(t.season.between(2015, 2025)) & t.margin.notna() & t.line.notna()].copy()
h["cover"] = np.sign(h.margin + h.line * 0)  # placeholder
h["ats"] = np.sign(h.margin + (-h.line) * -1)
# team covers if margin + (spread from team's view as a number to add) > 0; team favored by L means it must win by more than L
h["ats_margin"] = h.margin - h.line
h["exp_bl"] = None
print("Team-games 2015-2025:", len(h), "| flagged:", int(h.flag.sum()), f"({h.flag.mean():.1%})")
def summary(x, label):
    a = x[x.ats_margin != 0]
    print(f"  {label:38s} n={len(x):4d} | blowout loss {x.blowout_loss.mean():.1%} | ATS cover {(a.ats_margin>0).mean():.1%} "
          f"| avg line {x.line.mean():+.1f} | avg margin {x.margin.mean():+.1f} | beat the line by {x.ats_margin.mean():+.1f}")
summary(h[h.flag == 1], "Flagged (2+ in last 5 or 3+ this season)")
summary(h[h.flag == 0], "Not flagged")
# compare with teams given similar lines: are flagged teams worse than the market expects?
h["line_b"] = pd.cut(h.line, [-30, -10, -6.5, -3.5, 0, 3.5, 30])
cmp = h.groupby(["line_b", "flag"], observed=True).agg(n=("margin", "size"), blowout=("blowout_loss", "mean"),
        cover=("ats_margin", lambda s: (s[s != 0] > 0).mean()), beat=("ats_margin", "mean")).round(3)
print(cmp.to_string())
for k in (2, 3, 4):
    summary(h[h.bl_last5 >= k], f"{k}+ blowout losses in last 5")
    summary(h[(h.bl_season >= k + 1) & (h.week >= 8)], f"{k+1}+ blowout losses this season, wk 8+")
# totals in flagged teams' games
a = h[(h.flag == 1) & (h.total != h.total_line)]
print(f"  Totals in flagged teams' games: over {(a.total > a.total_line).mean():.1%} of {len(a)}")
# comeback behaviour: flagged teams trailing 17+ in the second half
cs = os.path.join(M, "comeback_situations.csv")
s = pd.read_csv(cs) if os.path.exists(cs) else pd.DataFrame(columns=["game_id", "trail", "within_one", "won", "trail_pts_after"])
s = s.merge(t[["game_id", "team", "flag", "bl_season"]].rename(columns={"team": "trail"}), on=["game_id", "trail"], how="left")
for f in (1, 0):
    x = s[s.flag == f]
    print(f"  Down 17+ in 2nd half, {'flagged' if f else 'not flagged'}: n={len(x)} within one {x.within_one.mean():.1%} | won {x.won.mean():.1%} | pts after {x.trail_pts_after.mean():.1f}")

cur = t[t.season == 2026].groupby("team").agg(bl_season=("blowout_loss", "sum"), bw_season=("blowout_win", "sum"),
        played=("margin", "count")).reset_index()
last5 = t[(t.season == 2026) & t.margin.notna()].groupby("team").tail(5).groupby("team").blowout_loss.sum().rename("bl_last5")
cur = cur.merge(last5, on="team", how="left").fillna({"bl_last5": 0})
cur["flag"] = ((cur.bl_last5 >= 2) | (cur.bl_season >= 3)).astype(int)
# also how many of the last 3 seasons (2023-2025) each team finished with 4+ blowout losses: the "habitual" part
hist = t[t.season.between(2023, 2025)].groupby(["team", "season"]).blowout_loss.sum().reset_index()
cur = cur.merge(hist.groupby("team").blowout_loss.sum().rename("bl_last3yrs"), on="team", how="left").fillna({"bl_last3yrs": 0})
cur = cur.sort_values(["bl_season", "bl_last5"], ascending=False)
print(cur.to_string(index=False))
cur.to_csv(os.path.join(M, "blowouts_2026.csv"), index=False)
