"""Do players have real "fast starter" or "slow starter" patterns?

Builds per player, per quarter stats from play by play, then asks whether the share of a player's yards that come
in the first quarter or first half repeats: odd numbered vs even numbered games in the same season, and one season
to the next. If it does not repeat, quarter props should lean on league norms for the position, not the player.

Usage: python3 -I research/quarter_patterns.py WORK_DIR
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import player_quarters as Q

W = sys.argv[1]
qs = Q.quarter_stats(os.path.join(W, "data"))
pg = pd.read_csv(os.path.join(W, "player_games.csv"), usecols=["game_id", "player_id", "position", "season", "season_type"])
qs = qs.merge(pg, on=["game_id", "player_id"], how="inner")
qs = qs[(qs.season_type == "REG") & qs.season.between(2015, 2025)]

MK = {"pass_yds": ("QB", 150), "rush_yds": ("RB", 30), "rec_yds": (("WR", "TE", "RB"), 25)}
print("League share of regulation yards by quarter, 2015 to 2025")
for m, (pos, floor) in MK.items():
    d = qs[qs.position.isin([pos] if isinstance(pos, str) else pos)]
    tot = d[[f"{m}_q{q}" for q in range(1, 5)]].sum()
    print(f"  {m:9s}", "  ".join(f"Q{q} {tot.iloc[q-1] / tot.sum():.1%}" for q in range(1, 5)))
print()

def stability(m, pos, floor, early):
    d = qs[qs.position.isin([pos] if isinstance(pos, str) else pos)].copy()
    d["reg"] = d[[f"{m}_q{q}" for q in range(1, 5)]].sum(axis=1)
    d = d[d.reg >= floor]
    d["early"] = d[[f"{m}_q{q}" for q in early]].sum(axis=1)
    d["n"] = d.groupby(["player_id", "season"]).cumcount()
    d["half"] = np.where(d.n % 2 == 0, "odd", "even")
    agg = d.groupby(["player_id", "season", "half"]).agg(g=("reg", "size"), e=("early", "sum"), r=("reg", "sum")).reset_index()
    agg["share"] = agg.e / agg.r
    w = agg[agg.g >= 4].pivot_table(index=["player_id", "season"], columns="half", values="share").dropna()
    split = w["odd"].corr(w["even"])
    s = d.groupby(["player_id", "season"]).agg(g=("reg", "size"), e=("early", "sum"), r=("reg", "sum")).reset_index()
    s = s[s.g >= 8]; s["share"] = s.e / s.r
    nxt = s.assign(season=s.season - 1)[["player_id", "season", "share"]].rename(columns={"share": "next"})
    yy = s.merge(nxt, on=["player_id", "season"])
    return len(w), split, len(yy), yy.share.corr(yy.next), s.share.std()

print("Does a player's early share repeat? (correlation; 0 = pure noise, 1 = perfectly repeatable)")
for m, (pos, floor) in MK.items():
    for label, early in (("1st quarter", [1]), ("1st half", [1, 2])):
        n1, c1, n2, c2, sd = stability(m, pos, floor, early)
        print(f"  {m:9s} {label:11s} odd vs even games {c1:+.2f} ({n1} player seasons) | season to season {c2:+.2f} ({n2}) | spread of shares {sd:.3f}")
print()

# Game script: does the point spread move the early share? (favorites run late, underdogs pass late)
tg = pd.read_csv(os.path.join(W, "team_games.csv"), usecols=["game_id", "team", "spread_line", "is_home"])
pg2 = pd.read_csv(os.path.join(W, "player_games.csv"), usecols=["game_id", "player_id", "team"])
d = qs.merge(pg2, on=["game_id", "player_id"]).merge(tg, on=["game_id", "team"])
d["team_spread"] = np.where(d.is_home == 1, d.spread_line, -d.spread_line)
d["fav"] = pd.cut(d.team_spread, [-40, -6.5, -2.5, 2.5, 6.5, 40], labels=["dog 7+", "dog 3-6", "close", "fav 3-6", "fav 7+"])
print("First half share of regulation yards by pregame spread")
for m, (pos, floor) in MK.items():
    x = d[d.position.isin([pos] if isinstance(pos, str) else pos)]
    g = x.groupby("fav", observed=True).apply(lambda t: t[[f"{m}_q1", f"{m}_q2"]].sum().sum() / t[[f"{m}_q{q}" for q in range(1, 5)]].sum().sum())
    print(f"  {m:9s}", "  ".join(f"{k} {v:.1%}" for k, v in g.items()))

# First touchdown of the game: who scores it
ft = Q.first_tds(os.path.join(W, "data"))
ft = ft[ft.season.between(2015, 2025)].merge(pg.drop_duplicates(["player_id"])[["player_id", "position"]], on="player_id", how="left")
print(f"\nFirst touchdown of the game, {len(ft)} games: by", ft.position.fillna("defense/special teams").value_counts(normalize=True).round(3).to_dict())
print("  quarter of the first TD:", ft.qtr.value_counts(normalize=True).sort_index().round(3).to_dict())
