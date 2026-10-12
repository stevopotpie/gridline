"""Second half 17+ point deficits since 2015: who comes back, who gets within one score, and why.
Usage: python3 -I comebacks.py DATA_DIR MODEL_DIR"""
import sys, glob, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

DATA, M = sys.argv[1:3]
COLS = ["game_id", "play_id", "season", "week", "home_team", "away_team", "posteam", "qtr", "game_seconds_remaining",
        "total_home_score", "total_away_score", "vegas_home_wp", "home_wp", "season_type"]
p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(DATA, "pbp_*.parquet")))], ignore_index=True)
p = p.sort_values(["game_id", "play_id"]).reset_index(drop=True)
# score before each play = score after the previous play
for c in ("total_home_score", "total_away_score"):
    p[c] = p.groupby("game_id")[c].ffill().fillna(0)
    p[c + "_pre"] = p.groupby("game_id")[c].shift(1).fillna(0)
p["gap_pre"] = p.total_home_score_pre - p.total_away_score_pre      # home minus away before the play
p["gap_post"] = p.total_home_score - p.total_away_score

snap = p[(p.qtr >= 3) & (p.gap_pre.abs() >= 17) & p.vegas_home_wp.notna()].groupby("game_id").head(1).copy()
snap["trail"] = np.where(snap.gap_pre < 0, snap.home_team, snap.away_team)
snap["lead"] = np.where(snap.gap_pre < 0, snap.away_team, snap.home_team)
snap["trail_home"] = (snap.gap_pre < 0).astype(int)
snap["deficit"] = snap.gap_pre.abs()
snap["secs"] = snap.game_seconds_remaining
snap["has_ball"] = (snap.posteam == snap.trail).astype(int)
snap["vwp"] = np.where(snap.trail_home == 1, snap.vegas_home_wp, 1 - snap.vegas_home_wp).clip(0.001, 0.999)

# what happened after the snapshot
rest = p.merge(snap[["game_id", "play_id", "trail_home"]].rename(columns={"play_id": "snap_play"}), on="game_id")
rest = rest[rest.play_id >= rest.snap_play]
rest["trail_gap"] = np.where(rest.trail_home == 1, rest.gap_post, -rest.gap_post)   # trailing team minus leading team
agg = rest.groupby("game_id").agg(min_deficit=("trail_gap", lambda s: -s.max()), final_gap=("trail_gap", "last"),
                                  hs_end=("total_home_score", "last"), as_end=("total_away_score", "last"))
snap = snap.merge(agg, on="game_id")
snap["won"] = (snap.final_gap > 0).astype(int)
snap["within_one"] = (snap.min_deficit <= 8).astype(int)            # got back to a one score game at some point
snap["trail_pts_after"] = np.where(snap.trail_home == 1, snap.hs_end - snap.total_home_score_pre, snap.as_end - snap.total_away_score_pre)
snap["lead_pts_after"] = np.where(snap.trail_home == 1, snap.as_end - snap.total_away_score_pre, snap.hs_end - snap.total_home_score_pre)

# pregame context
g = pd.read_csv(os.path.join(M, "games_features_all.csv"), parse_dates=["gameday"], low_memory=False)
# records entering each game
recs = []
for side in ("home", "away"):
    x = g[["game_id", "season", "gameday", f"{side}_team", "result"]].copy(); x.columns = ["game_id", "season", "gameday", "team", "result"]
    x["w"] = np.where(x.result.isna(), np.nan, (x.result > 0) if side == "home" else (x.result < 0)).astype(float)
    x["l"] = np.where(x.result.isna(), np.nan, (x.result < 0) if side == "home" else (x.result > 0)).astype(float)
    recs.append(x)
recs = pd.concat(recs).sort_values("gameday")
recs["w_before"] = recs.groupby(["season", "team"]).w.cumsum() - recs.w.fillna(0)
recs["l_before"] = recs.groupby(["season", "team"]).l.cumsum() - recs.l.fillna(0)
recs = recs.set_index(["game_id", "team"])[["w_before", "l_before"]]

cols = ["game_id", "home_team", "away_team", "spread_line", "total_line", "div_game", "gametime", "dome", "wind_mph", "temp_f"]
stats = ["qb_epa", "off_pass_epa", "def_pass_epa", "off_expl", "def_expl", "pace", "proe", "off_sr", "def_sr", "ppg", "papg"]
snap = snap.merge(g[cols + [f"{s}_{k}" for s in ("home", "away") for k in stats]], on=["game_id", "home_team", "away_team"])
for k in stats:
    snap[f"trail_{k}"] = np.where(snap.trail_home == 1, snap[f"home_{k}"], snap[f"away_{k}"])
    snap[f"lead_{k}"] = np.where(snap.trail_home == 1, snap[f"away_{k}"], snap[f"home_{k}"])
snap["trail_spread"] = np.where(snap.trail_home == 1, snap.spread_line, -snap.spread_line)   # + means the trailing team was favored
snap["trail_favored"] = (snap.trail_spread > 0).astype(int)
idx = list(zip(snap.game_id, snap.trail)); idx2 = list(zip(snap.game_id, snap.lead))
snap["trail_w"] = recs.reindex(idx).w_before.values; snap["trail_l"] = recs.reindex(idx).l_before.values
snap["lead_w"] = recs.reindex(idx2).w_before.values; snap["lead_l"] = recs.reindex(idx2).l_before.values
gp = (snap.trail_w + snap.trail_l).replace(0, np.nan)
snap["trail_winpct"] = (snap.trail_w / gp).fillna(0.5)
snap["late"] = (snap.week >= 12).astype(int)
snap["trail_contender"] = ((snap.trail_winpct >= 0.5) & (snap.week >= 8)).astype(int)
snap["trail_out_of_it"] = ((snap.trail_winpct <= 0.3) & (snap.week >= 10)).astype(int)
snap["lead_contender"] = (((snap.lead_w / (snap.lead_w + snap.lead_l).replace(0, np.nan)).fillna(.5) >= 0.5) & (snap.week >= 8)).astype(int)
snap["playoffs"] = (snap.season_type == "POST").astype(int)
snap["primetime"] = snap.gametime.astype(str).str[:2].isin(["19", "20", "21"]).astype(int)
snap["windy"] = ((snap.wind_mph >= 15) & (snap.dome == 0)).astype(int)
snap["logit_vwp"] = np.log(snap.vwp / (1 - snap.vwp))
snap = snap[snap.trail_qb_epa.notna() & snap.lead_def_pass_epa.notna()].copy()
print("situations:", len(snap), "| comebacks won:", int(snap.won.sum()), f"({snap.won.mean():.1%})",
      "| got within one score:", int(snap.within_one.sum()), f"({snap.within_one.mean():.1%})")

# ---------- 1. base rates ----------
snap["def_b"] = pd.cut(snap.deficit, [16, 20, 24, 99], labels=["17-20", "21-24", "25+"])
snap["time_b"] = pd.cut(snap.secs, [-1, 480, 900, 1350, 1800], labels=["4Q, under 8 min", "4Q, 8-15 min", "3Q, under 7:30", "3Q, first half of quarter"])
base = snap.groupby(["def_b", "time_b"], observed=True).agg(n=("won", "size"), won=("won", "mean"), within_one=("within_one", "mean"),
        trail_pts=("trail_pts_after", "mean"), vegas_wp=("vwp", "mean")).round(3).reset_index()
print(base.to_string(index=False))

# ---------- 2. which factors beat the win probability model ----------
BASE = ["logit_vwp", "deficit", "secs", "has_ball"]
GROUPS = {
    "Trailing QB rating": ["trail_qb_epa"],
    "Trailing pass offense (EPA, explosives)": ["trail_off_pass_epa", "trail_off_expl"],
    "Trailing pace and pass rate": ["trail_pace", "trail_proe"],
    "Leading team pass defense": ["lead_def_pass_epa", "lead_def_expl"],
    "Leading team offense (keeps scoring)": ["lead_off_pass_epa", "lead_off_sr"],
    "Trailing team was the favorite": ["trail_spread"],
    "Trailing team at home": ["trail_home"],
    "Motivation: trailing contender / out of it": ["trail_contender", "trail_out_of_it"],
    "Motivation: leading team contender": ["lead_contender"],
    "Stakes: division game, late season, primetime": ["div_game", "late", "primetime"],
    "Wind 15+ mph": ["windy"],
}
seasons = sorted(snap.season.unique())
def cv(cols, target, kind="logit"):
    preds = pd.Series(index=snap.index, dtype=float)
    for s in seasons:
        tr, te = snap[snap.season != s], snap[snap.season == s]
        if kind == "logit":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(tr[cols], tr[target])
            preds[te.index] = m.predict_proba(te[cols])[:, 1]
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=5)).fit(tr[cols], tr[target]); preds[te.index] = m.predict(te[cols])
    return preds
rows = []
for target, kind in (("won", "logit"), ("within_one", "logit"), ("trail_pts_after", "ridge")):
    b = cv(BASE, target, kind)
    bscore = log_loss(snap[target], b) if kind == "logit" else (b - snap[target]).abs().mean()
    full = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000) if kind == "logit" else Ridge(alpha=5))
    allc = BASE + [c for v in GROUPS.values() for c in v]
    full.fit(snap[allc], snap[target]); coefs = dict(zip(allc, full[-1].coef_.ravel()))
    for name, gcols in GROUPS.items():
        pr = cv(BASE + gcols, target, kind)
        sc = log_loss(snap[target], pr) if kind == "logit" else (pr - snap[target]).abs().mean()
        eff = " / ".join(f"{coefs[c]:+.2f}" for c in gcols)
        rows.append({"target": target, "factor": name, "gain": round(bscore - sc, 4), "effect per SD": eff})
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for t in ("won", "within_one", "trail_pts_after"):
    print(f"\n== {t} (gain = better than base; logit effects are log-odds per SD, points effects are pts per SD) ==")
    print(R[R.target == t].sort_values("gain", ascending=False).to_string(index=False))

# simple splits that are easy to read
print("\nSimple splits (within one score rate / comeback win rate / trailing pts after):")
def split(name, mask):
    a, b = snap[mask], snap[~mask]
    print(f"  {name:42s} yes n={len(a):4d}: {a.within_one.mean():.1%} / {a.won.mean():.1%} / {a.trail_pts_after.mean():.1f}   "
          f"no n={len(b):4d}: {b.within_one.mean():.1%} / {b.won.mean():.1%} / {b.trail_pts_after.mean():.1f}")
q = snap.trail_qb_epa.quantile([.25, .75])
split("Trailing QB in top quarter", snap.trail_qb_epa >= q.iloc[1])
split("Trailing QB in bottom quarter", snap.trail_qb_epa <= q.iloc[0])
split("Trailing team was favored pregame", snap.trail_spread > 0)
split("Trailing team is a contender (wk 8+, .500+)", snap.trail_contender == 1)
split("Trailing team out of it (wk 10+, .300 or worse)", snap.trail_out_of_it == 1)
split("Trailing team at home", snap.trail_home == 1)
split("Division game", snap.div_game == 1)
split("Leading pass D in top quarter", snap.lead_def_pass_epa <= snap.lead_def_pass_epa.quantile(.25))
split("Playoff game", snap.playoffs == 1)

snap.to_csv(os.path.join(M, "comeback_situations.csv"), index=False)
base.to_csv(os.path.join(M, "comeback_base_rates.csv"), index=False)
R.to_csv(os.path.join(M, "comeback_factors.csv"), index=False)
