"""Do some teams settle for the loss when down 17+ in the second half? Measures urgency in those stretches,
tests whether it is a repeatable coach/team habit, and whether a coach's past urgency predicts what happens next.
Usage: python3 -I urgency.py DATA_DIR MODEL_DIR"""
import sys, glob, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

DATA, M = sys.argv[1:3]
COLS = ["game_id", "play_id", "season", "posteam", "home_team", "away_team", "qtr", "game_seconds_remaining", "down", "ydstogo",
        "yardline_100", "play_type", "qb_dropback", "no_huddle", "pass_oe", "passer_player_id", "fixed_drive",
        "total_home_score", "total_away_score"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet")))], ignore_index=True)
p = p.sort_values(["game_id", "play_id"]).reset_index(drop=True)
for c in ("total_home_score", "total_away_score"):
    p[c] = p.groupby("game_id")[c].ffill().fillna(0)
    p[c + "_pre"] = p.groupby("game_id")[c].shift(1).fillna(0)
p["off_gap"] = np.where(p.posteam == p.home_team, p.total_home_score_pre - p.total_away_score_pre, p.total_away_score_pre - p.total_home_score_pre)

# pace: seconds to the next snap in the same drive
snaps = p[p.play_type.isin(["pass", "run"])].copy()
snaps["gap"] = snaps.game_seconds_remaining - snaps.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
p = p.merge(snaps[["game_id", "play_id", "gap"]], on=["game_id", "play_id"], how="left")

# starting QB: the passer with the most dropbacks in the first half
first = p[(p.qtr <= 2) & (p.qb_dropback == 1) & p.passer_player_id.notna()]
starter = first.groupby(["game_id", "posteam"]).passer_player_id.agg(lambda s: s.value_counts().index[0]).rename("starter")

# window: the trailing team's offense, second half, down 17+, more than 2 minutes left
w = p[(p.qtr >= 3) & (p.off_gap <= -17) & (p.game_seconds_remaining > 120)].copy()
w = w.join(starter, on=["game_id", "posteam"])
w["is_snap"] = w.play_type.isin(["pass", "run"]).astype(int)
w["db"] = ((w.qb_dropback == 1) & (w.is_snap == 1)).astype(int)
w["nh"] = ((w.no_huddle == 1) & (w.is_snap == 1)).astype(int)
w["pace_ok"] = w.gap.between(5, 60).astype(int)
fourth = (w.down == 4) & (w.ydstogo <= 5) & (w.yardline_100 <= 60) & w.play_type.isin(["pass", "run", "punt", "field_goal"])
w["fd"] = fourth.astype(int); w["fd_go"] = (fourth & w.play_type.isin(["pass", "run"])).astype(int)
w["backup_threw"] = ((w.db == 1) & w.passer_player_id.notna() & w.starter.notna() & (w.passer_player_id != w.starter) & (w.game_seconds_remaining > 300)).astype(int)
gm = w.groupby(["game_id", "posteam"]).agg(season=("season", "first"), snaps=("is_snap", "sum"), db=("db", "sum"), nh=("nh", "sum"),
        proe=("pass_oe", "mean"), pace=("gap", lambda s: s[s.between(5, 60)].mean()), fd=("fd", "sum"), fd_go=("fd_go", "sum"),
        qb_pulled=("backup_threw", "max")).reset_index().rename(columns={"posteam": "team"})
gm = gm[gm.snaps >= 6].copy()
print("game stretches with 6+ snaps while down 17+:", len(gm))

# coach for each team-game
g = pd.read_csv(os.path.join(DATA, "games.csv"))
coach = pd.concat([g[["game_id", "home_team", "home_coach"]].set_axis(["game_id", "team", "coach"], axis=1),
                   g[["game_id", "away_team", "away_coach"]].set_axis(["game_id", "team", "coach"], axis=1)])
gm = gm.merge(coach, on=["game_id", "team"], how="left")
gm["nh_rate"] = gm.nh / gm.snaps; gm["db_rate"] = gm.db / gm.snaps
gm["pace"] = gm.pace.fillna(gm.pace.median())
# urgency index: pass rate over expected, no-huddle share, and faster snaps (each standardized)
z = lambda s: (s - s.mean()) / s.std()
gm["urgency"] = (z(gm.proe.fillna(gm.proe.mean())) + z(gm.nh_rate) - z(gm.pace)) / 3

print("\nWhat 'down 17+' football looks like (per game stretch):")
print(f"  pass rate {gm.db_rate.mean():.0%}, no-huddle {gm.nh_rate.mean():.0%}, {gm.pace.mean():.1f} sec between snaps, "
      f"pass rate vs expected {gm.proe.mean():+.1f} pts, starter pulled with 5+ min left in {gm.qb_pulled.mean():.1%} of games")
f4 = gm[gm.fd > 0]
print(f"  4th and 5 or less from their own 40 or closer to the end zone: went for it {f4.fd_go.sum()/f4.fd.sum():.0%} of {int(f4.fd.sum())} chances")

# ---------- stability: is it a habit? ----------
def stability(key, min_games=3):
    s = gm.groupby([key, "season"]).agg(u=("urgency", "mean"), n=("urgency", "size"), pull=("qb_pulled", "mean")).reset_index()
    s = s[s.n >= min_games]
    nxt = s.assign(season=s.season - 1)
    m = s.merge(nxt, on=[key, "season"], suffixes=("", "_next"))
    return len(m), m.u.corr(m.u_next), m.pull.corr(m.pull_next)
for key in ("coach", "team"):
    n, cu, cp = stability(key)
    print(f"\nYear to year by {key}: {n} pairs | urgency correlation {cu:.2f} | QB pull rate correlation {cp:.2f}")

# ---------- does a coach's PAST urgency predict comebacks? ----------
s = pd.read_csv(os.path.join(M, "comeback_situations.csv"))
s = s.merge(coach.rename(columns={"team": "trail", "coach": "trail_coach"}), on=["game_id", "trail"], how="left")
gdates = g.set_index("game_id").gameday
gm["date"] = gm.game_id.map(gdates); s["date"] = s.game_id.map(gdates)
gm = gm.sort_values("date")
prior_u, prior_n, prior_pull = [], [], []
for r in s.itertuples():
    h = gm[(gm.coach == r.trail_coach) & (gm.date < r.date)]
    k = len(h)
    # shrink toward the league average (0) by 8 games' worth of evidence
    prior_u.append(h.urgency.sum() / (k + 8)); prior_n.append(k)
    prior_pull.append((h.qb_pulled.sum() + 8 * gm.qb_pulled.mean()) / (k + 8))
s["coach_urgency"] = prior_u; s["coach_n"] = prior_n; s["coach_pull"] = prior_pull
s["secs2"] = (s.secs / 1800) ** 2; s["under10"] = (s.secs < 600).astype(int); s["half"] = (s.secs >= 1750).astype(int)
BASE = ["deficit", "secs", "secs2", "under10", "half", "has_ball", "trail_spread", "trail_home", "trail_qb_epa", "trail_off_pass_epa",
        "trail_off_expl", "lead_def_pass_epa", "lead_off_pass_epa", "trail_contender", "trail_out_of_it"]
def cv(cols, target, logit=True):
    mu, sd = s[cols].mean(), s[cols].std().replace(0, 1); Z = (s[cols] - mu) / sd
    pr = pd.Series(index=s.index, dtype=float)
    for yr in s.season.unique():
        tr, te = s.season != yr, s.season == yr
        if logit: pr[te] = LogisticRegression(C=0.5, max_iter=3000).fit(Z[tr], s[target][tr]).predict_proba(Z[te])[:, 1]
        else: pr[te] = Ridge(alpha=5).fit(Z[tr], s[target][tr]).predict(Z[te])
    return pr
print("\nAdding the coach's past urgency to the comeback model (season-by-season holdout; positive = helps):")
for target, logit in (("within_one", True), ("won", True), ("trail_pts_after", False)):
    for extra in (["coach_urgency"], ["coach_pull"], ["coach_urgency", "coach_pull"]):
        b, a = cv(BASE, target, logit), cv(BASE + extra, target, logit)
        if logit: gain = log_loss(s[target], b) - log_loss(s[target], a)
        else: gain = (b - s[target]).abs().mean() - (a - s[target]).abs().mean()
        print(f"  {target:16s} + {'+'.join(extra):26s} gain {gain:+.4f}")

# simple split: coaches with a real track record (10+ prior stretches), top vs bottom third of past urgency
has = s[s.coach_n >= 10]
lo, hi = has.coach_urgency.quantile([1/3, 2/3])
for name, m in (("Most urgent third of coaches", has.coach_urgency >= hi), ("Least urgent third", has.coach_urgency <= lo)):
    x = has[m]
    print(f"  {name:30s} n={len(x):3d} within one {x.within_one.mean():.1%} | won {x.won.mean():.1%} | pts after {x.trail_pts_after.mean():.1f} | pregame line {x.trail_spread.mean():+.1f}")

# ---------- 2026 head coaches ----------
cur = g[g.season == 2026]
c26 = pd.concat([cur[["home_team", "home_coach"]].set_axis(["team", "coach"], axis=1), cur[["away_team", "away_coach"]].set_axis(["team", "coach"], axis=1)]) \
        .drop_duplicates("team", keep="last")
prof = gm.groupby("coach").agg(n=("urgency", "size"), urgency=("urgency", "sum"), pull=("qb_pulled", "sum"), proe=("proe", "mean"),
        nh=("nh_rate", "mean"), pace=("pace", "mean"), fd=("fd", "sum"), fd_go=("fd_go", "sum")).reset_index()
prof["urgency_shrunk"] = prof.urgency / (prof.n + 8)
prof["pull_rate"] = prof.pull / prof.n
prof["go_rate"] = prof.fd_go / prof.fd.replace(0, np.nan)
c26 = c26.merge(prof, on="coach", how="left").fillna({"n": 0})
c26 = c26.sort_values("urgency_shrunk", ascending=False)
pd.set_option("display.width", 200)
print("\n2026 head coaches, urgency when down 17+ (shrunk toward average; n = past game stretches since 2015):")
print(c26[["team", "coach", "n", "urgency_shrunk", "proe", "nh", "pace", "go_rate", "pull_rate"]].round(3).to_string(index=False))
c26.to_csv(os.path.join(M, "coach_urgency_2026.csv"), index=False)
gm.to_csv(os.path.join(M, "urgency_games.csv"), index=False)
s.to_csv(os.path.join(M, "comeback_situations_urgency.csv"), index=False)
