"""Do penalties eat into a quarterback's passing yards?

Two ways they could: offensive penalties (holding, false starts) wipe out completions and stall drives, and defensive
pass interference turns would-be catches into penalty yards instead of passing yards.

1. Same game: in games with lots of flags, did QBs throw for fewer yards than the player model expected?
   (Shows whether the mechanism exists, but these flags are not known before kickoff.)
2. Before the game: does a team's penalty habit (offense) or an opponent's (defense), known ahead of time,
   predict passing yards beyond what the model already expects? Tested walk forward on 2019 to 2025,
   the same way the player model was chosen.

Usage: python3 -I research/penalties_passing.py WORK_DIR   (after run_all.sh, which builds player_games.csv)
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import player_features as F

W = sys.argv[1]
D = os.path.join(W, "data")
HL = 6
TESTS = range(2019, 2026)
BASE = ["e_pass_yds", "std_pass_yds", "e_pass_att", "e_ypa", "implied_pts", "team_spread", "t_team_dropbacks",
        "d_pass_yds", "d_dropbacks", "is_home", "dome", "temp", "wind", "wind_over10", "cold"]

# ---------- penalties per team per game, split by who committed them ----------
cols = ["game_id", "game_date", "posteam", "defteam", "penalty", "penalty_team", "penalty_yards", "penalty_type"]
rows = []
for y in range(2015, 2027):
    f = os.path.join(D, f"pbp_{y}.parquet")
    if not os.path.exists(f): continue
    p = pd.read_parquet(f, columns=cols)
    p = p[(p.penalty == 1) & p.penalty_team.notna() & p.posteam.notna()]
    p["side"] = np.where(p.penalty_team == p.posteam, "off", np.where(p.penalty_team == p.defteam, "def", None))
    p["dpi"] = p.penalty_type.eq("Defensive Pass Interference")
    p["hold"] = p.penalty_type.eq("Offensive Holding")
    rows.append(p[p.side.notna()])
p = pd.concat(rows, ignore_index=True)
off = p[p.side == "off"].groupby(["game_id", "penalty_team"]).agg(off_pen_yds=("penalty_yards", "sum"), off_holds=("hold", "sum"))
p["dpi_yds"] = p.penalty_yards.where(p.dpi, 0)
dfn = p[p.side == "def"].groupby(["game_id", "penalty_team"]).agg(def_pen_yds=("penalty_yards", "sum"), dpi_yds=("dpi_yds", "sum"),
                                                                    dpi_flags=("dpi", "sum"))
pen = off.join(dfn, how="outer").fillna(0).reset_index().rename(columns={"penalty_team": "team"})

# every team game, so teams with zero flags count as zero (not missing)
tg = pd.read_csv(os.path.join(W, "team_games.csv"))[["game_id", "team", "opp", "game_date"]]
tg = tg.merge(pen, on=["game_id", "team"], how="left")
played = tg.game_id.isin(p.game_id.unique())
for c in ["off_pen_yds", "off_holds", "def_pen_yds", "dpi_yds"]:
    tg.loc[played, c] = tg.loc[played, c].fillna(0)
tg["game_date"] = pd.to_datetime(tg.game_date)
tg = tg.sort_values(["team", "game_date"])
habit = tg.groupby("team")[["off_pen_yds", "off_holds", "def_pen_yds", "dpi_yds"]].transform(
    lambda s: s.shift(1).ewm(halflife=HL, min_periods=1).mean())
tg[["h_off_pen", "h_off_holds", "h_def_pen", "h_dpi"]] = habit.values

# ---------- quarterbacks ----------
pg = F.build(pd.read_csv(os.path.join(W, "player_games.csv")), pd.read_csv(os.path.join(W, "team_games.csv")),
             pd.read_csv(os.path.join(D, "games.csv")))
pg = pg[pg.played & (pg.season >= 2016) & (pg.n_prior >= 3)].copy()
pg = pg[F.eligible(pg, "pass_yds")].copy()
own = tg[["game_id", "team", "off_pen_yds", "off_holds", "h_off_pen", "h_off_holds"]]
opp = tg[["game_id", "team", "def_pen_yds", "dpi_yds", "h_def_pen", "h_dpi"]].rename(columns={"team": "opp"})
pg = pg.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left")
pg = pg[pg.h_off_pen.notna() & pg.h_def_pen.notna()]


def walk_forward(cols):
    out = []
    for s in TESTS:
        tr, te = pg[pg.season < s], pg[pg.season == s]
        X, Xt = tr[cols].fillna(tr[cols].median()), te[cols].fillna(tr[cols].median())
        out.append(pd.Series(np.clip(Ridge(alpha=1.0).fit(X, tr.pass_yds).predict(Xt), 0, None), index=te.index))
    return pd.concat(out)


test = pg[pg.season.isin(TESTS)].copy()
test["pred"] = walk_forward(BASE)
test["resid"] = test.pass_yds - test.pred           # positive = threw for more than the model expected
print(f"QB games 2019 to 2025: {len(test):,} | model average miss {np.mean(np.abs(test.resid)):.1f} yds\n")

def by_fifth(col, label):
    q = pd.qcut(test[col].rank(method="first"), 5, labels=["fewest", "2", "3", "4", "most"])
    t = test.groupby(q, observed=True).agg(games=("resid", "size"), avg=(col, "mean"), vs_model=("resid", "mean"),
                                           under_model=("resid", lambda r: (r < 0).mean()))
    print(label); print(t.round(2).to_string()); print()

print("1. SAME GAME (not known before kickoff)")
by_fifth("off_pen_yds", "QB's own offense, penalty yards that game:")
by_fifth("dpi_yds", "Opposing defense, pass interference yards that game:")

print("2. BEFORE THE GAME (team habits from earlier games)")
by_fifth("h_off_pen", "QB's own offense, penalty yards per game so far:")
by_fifth("h_def_pen", "Opposing defense, penalty yards per game so far:")
by_fifth("h_dpi", "Opposing defense, pass interference yards per game so far:")

for name, extra in [("+ own offense penalty habit", ["h_off_pen"]), ("+ own offensive holding habit", ["h_off_holds"]),
                    ("+ opponent defense penalty habit", ["h_def_pen"]), ("+ opponent pass interference habit", ["h_dpi"]),
                    ("+ all four", ["h_off_pen", "h_off_holds", "h_def_pen", "h_dpi"])]:
    p2 = walk_forward(BASE + extra)
    print(f"{name:36s} average miss {np.mean(np.abs(test.pass_yds - p2)):.2f} yds (model now {np.mean(np.abs(test.resid)):.2f})")

# how stable is the habit? first half of a season vs second half
tg["season"] = tg.game_id.str[:4].astype(int)
tg["n"] = tg.groupby(["team", "season"]).cumcount()
h = tg[tg.season.between(2015, 2025)].assign(half=lambda d: np.where(d.n < 8, "first", "second"))
s = h.pivot_table(index=["team", "season"], columns="half", values=["off_pen_yds", "def_pen_yds", "dpi_yds"], aggfunc="mean")
print("\nCarry over, first 8 games vs the rest of the season (correlation):")
for c in ["off_pen_yds", "def_pen_yds", "dpi_yds"]:
    print(f"  {c}: {s[c].corr().loc['first', 'second']:.2f}")
