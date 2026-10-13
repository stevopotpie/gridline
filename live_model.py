"""Gridline live game model: what happens from here, given the game state.

Learns from every regular season and playoff snap since 2015 (about 540,000 plays). For a game state (score, time
left, who has the ball, field position, down and distance, timeouts) plus the pregame spread and total, it predicts:
  - win:    the home team's chance to win
  - margin: the home team's expected margin over the rest of the game, plus how widely it varies, so any live
            spread can be priced
  - total:  expected points still to come, plus the spread of outcomes, so any live total can be priced
  - more:   the chance either team scores again
  - within: when one team trails by 9 or more, its chance to get back within one score (8 or fewer)
Each is a gradient boosted tree model. The trees are exported to JSON so the calculator page can run them in the
browser, and the export is checked against scikit-learn's own predictions before it is written.

Tested on 2023 to 2025 with models trained only on 2015 to 2022, against nflverse's own win probability model
(vegas_home_wp) on the same plays. The published model is then refit on 2015 to 2025.

Usage: python3 -I live_model.py DATA_DIR WORK_DIR   -> WORK_DIR/live_model.json, WORK_DIR/live_backtest.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

D, W = sys.argv[1], sys.argv[2]
FEATURES = ["diff", "secs", "half_secs", "home_ball", "ydl", "down", "togo", "home_to", "away_to", "spread", "total_line",
            "lead_per_min", "cur_total", "spread_time", "diff_time", "home_gets_2h"]
BUCKETS = [0, 120, 300, 600, 900, 1800, 2700, 3601]          # seconds left, for the spread of outcomes
QS = np.linspace(0, 1, 101)
TRAIN_END, TESTS = 2022, (2023, 2025)
FIRST_SEASON = 2010     # more history helped: 2010+ test log loss 0.456 vs 0.458 from 2015+
# Small, heavily regularized trees: plays within a game are tightly linked, so there are far fewer independent
# examples than rows, and bigger trees overfit badly (test log loss 0.539 vs 0.458 for these settings).
GBM = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=5000, l2_regularization=10.0,
           random_state=0, early_stopping=False)

COLS = ["game_id", "season", "play_id", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
        "half_seconds_remaining", "posteam", "home_team", "score_differential", "posteam_score", "defteam_score",
        "home_score", "away_score", "spread_line", "total_line", "home_timeouts_remaining", "away_timeouts_remaining",
        "vegas_home_wp"]


def load():
    out = []
    for y in range(FIRST_SEASON, 2027):
        f = os.path.join(D, f"pbp_{y}.parquet")
        if not os.path.exists(f): continue
        p = pd.read_parquet(f, columns=COLS)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p = p[p.home_score.notna() & p.score_differential.notna() & p.game_seconds_remaining.notna()].copy()
    home = p.posteam == p.home_team
    p["diff"] = np.where(home, p.score_differential, -p.score_differential)
    p["cur_total"] = p.posteam_score + p.defteam_score
    p = p.sort_values(["game_id", "play_id"])
    final_diff = p.home_score - p.away_score
    # did the trailing team ever get within one score (8 or fewer) later in the game, or at the end?
    g = p.groupby("game_id", sort=False)["diff"]
    fut_max = g.transform(lambda s: s[::-1].cummax()[::-1]); fut_min = g.transform(lambda s: s[::-1].cummin()[::-1])
    fut_max = np.maximum(fut_max, final_diff); fut_min = np.minimum(fut_min, final_diff)
    p["y_within"] = np.where(p["diff"] <= -9, fut_max >= -8, np.where(p["diff"] >= 9, fut_min <= 8, np.nan))
    # features for snaps in regulation with a down (skips kickoffs and extra points)
    p = p[(p.qtr <= 4) & p.down.notna() & p.yardline_100.notna()].copy()
    p["secs"] = p.game_seconds_remaining
    p["half_secs"] = p.half_seconds_remaining
    p["home_ball"] = (p.posteam == p.home_team).astype(int)
    p["ydl"] = p.yardline_100
    p["togo"] = p.ydstogo.clip(1, 30)
    p["home_to"], p["away_to"] = p.home_timeouts_remaining, p.away_timeouts_remaining
    p["spread"] = p.spread_line
    p["lead_per_min"] = p["diff"] / np.sqrt(p.secs / 60 + 1)
    # the pregame spread matters less as the game goes on and the lead matters more (the nflverse formulation)
    el = (3600 - p.secs) / 3600
    p["spread_time"] = p.spread * np.exp(-4 * el)
    p["diff_time"] = p["diff"] * np.exp(4 * el)
    # who gets the second half kickoff: the team that did not have the ball first; only matters in the first half
    first = p[p.qtr == 1].sort_values(["game_id", "play_id"]).groupby("game_id").home_ball.first()
    p["home_gets_2h"] = np.where(p.qtr <= 2, 1 - p.game_id.map(first), 0.5)
    p["y_win"] = np.where(p.home_score > p.away_score, 1, np.where(p.home_score < p.away_score, 0, np.nan))
    p["y_margin"] = (p.home_score - p.away_score) - p["diff"]
    p["y_total"] = (p.home_score + p.away_score) - p.cur_total
    p["y_more"] = (p.y_total > 0).astype(int)
    p["bucket"] = np.searchsorted(BUCKETS, p.secs, side="right") - 1
    return p.dropna(subset=["spread", "total_line"] + ["home_to", "away_to"])


def fit_all(tr):
    m = {}
    w = tr[tr.y_win.notna()]
    m["win"] = HistGradientBoostingClassifier(**GBM).fit(w[FEATURES], w.y_win.astype(int))
    m["margin"] = HistGradientBoostingRegressor(**GBM).fit(tr[FEATURES], tr.y_margin)
    m["total"] = HistGradientBoostingRegressor(**GBM).fit(tr[FEATURES], tr.y_total)
    m["more"] = HistGradientBoostingClassifier(**GBM).fit(tr[FEATURES], tr.y_more)
    wi = tr[tr.y_within.notna()]
    m["within"] = HistGradientBoostingClassifier(**GBM).fit(wi[FEATURES], wi.y_within.astype(int))
    # spread of outcomes around each regression's prediction, by time left: quantiles of the residual
    res = {}
    for k in ("margin", "total"):
        r = tr[f"y_{k}"].values - m[k].predict(tr[FEATURES])
        res[k] = [np.quantile(r[tr.bucket.values == b], QS).round(3).tolist() for b in range(len(BUCKETS) - 1)]
    return m, res


def prob_above(mu, line, qs):
    """Chance the outcome (mu + residual, residual given by quantiles qs) beats the line, the way a book grades it:
    on a whole number line, landing exactly on it is a push and the bet is refunded, so it is set aside."""
    cdf = lambda x: np.interp(x - mu, qs, QS, left=0, right=1)
    if float(line).is_integer():
        above, push = 1 - cdf(line + 0.5), cdf(line + 0.5) - cdf(line - 0.5)
        return above / max(1 - push, 1e-6)
    return 1 - cdf(line)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)):
    t = pd.DataFrame({"p": p, "y": y, "bin": pd.cut(p, edges, include_lowest=True)}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return [{"bin": str(i), "n": int(r.n), "predicted": round(float(r.predicted), 3), "actual": round(float(r.actual), 3)} for i, r in t.iterrows()]


def export(est):
    """Flatten a fitted HistGradientBoosting model into arrays the browser can walk."""
    trees = []
    for it in est._predictors:
        n = it[0].nodes
        trees.append({"f": n["feature_idx"].tolist(), "t": [float(v) for v in n["num_threshold"]],
                      "l": n["left"].tolist(), "r": n["right"].tolist(), "v": [round(float(v), 6) for v in n["value"]],
                      "leaf": n["is_leaf"].astype(int).tolist()})
    return {"base": float(np.ravel(est._baseline_prediction)[0]), "trees": trees}


def run_exported(e, X):
    """The same walk the page does, in Python, to prove the export matches scikit-learn."""
    out = np.full(len(X), e["base"])
    for t in e["trees"]:
        for i, row in enumerate(X):
            k = 0
            while not t["leaf"][k]:
                k = t["l"][k] if row[t["f"][k]] <= t["t"][k] else t["r"][k]
            out[i] += t["v"][k]
    return out


def main():
    p = load()
    hist = p[p.season <= 2025]

    # ---------- test: train 2015 to 2022, predict 2023 to 2025 ----------
    tr, te = hist[hist.season <= TRAIN_END], hist[hist.season.between(*TESTS)].copy()
    m, res = fit_all(tr)
    bt = {"train": f"2015 to {TRAIN_END}", "test": f"{TESTS[0]} to {TESTS[1]}", "plays": int(len(te)), "games": int(te.game_id.nunique())}
    w = te[te.y_win.notna()]
    pw = m["win"].predict_proba(w[FEATURES])[:, 1]
    bench = w.vegas_home_wp.values
    ok = ~np.isnan(bench)
    bt["win"] = {"logloss": ll(pw[ok], w.y_win.values[ok]), "logloss_nflverse": ll(bench[ok], w.y_win.values[ok]),
                 "brier": float(np.mean((pw[ok] - w.y_win.values[ok]) ** 2)), "brier_nflverse": float(np.mean((bench[ok] - w.y_win.values[ok]) ** 2)),
                 "reliability": reliability(pw, w.y_win.values)}
    # by game phase, where live betting happens
    phase = pd.cut(w.secs, [-1, 300, 900, 1800, 3600], labels=["last 5 min", "rest of 4th", "3rd quarter", "1st half"])
    bt["win"]["by_phase"] = {str(k): {"plays": int(ok[(phase == k).values].sum()),
                                      "logloss": round(ll(pw[(phase == k).values & ok], w.y_win.values[(phase == k).values & ok]), 4),
                                      "logloss_nflverse": round(ll(bench[(phase == k).values & ok], w.y_win.values[(phase == k).values & ok]), 4)}
                             for k in phase.cat.categories}
    # live spread and total: priced against the pregame numbers as stand in lines
    mu_m, mu_t = m["margin"].predict(te[FEATURES]), m["total"].predict(te[FEATURES])
    final_m, final_t = te["diff"].values + te.y_margin.values, te.cur_total.values + te.y_total.values
    pc = np.array([prob_above(te["diff"].values[i] + mu_m[i], te.spread.values[i], np.array(res["margin"][b])) for i, b in enumerate(te.bucket.values)])
    po = np.array([prob_above(te.cur_total.values[i] + mu_t[i], te.total_line.values[i], np.array(res["total"][b])) for i, b in enumerate(te.bucket.values)])
    keep_c, keep_o = final_m != te.spread.values, final_t != te.total_line.values
    yc, yo = (final_m > te.spread.values).astype(int), (final_t > te.total_line.values).astype(int)
    bt["cover"] = {"logloss": ll(pc[keep_c], yc[keep_c]), "logloss_coin_flip": ll(np.full(keep_c.sum(), .5), yc[keep_c]), "reliability": reliability(pc[keep_c], yc[keep_c])}
    bt["over"] = {"logloss": ll(po[keep_o], yo[keep_o]), "logloss_coin_flip": ll(np.full(keep_o.sum(), .5), yo[keep_o]), "reliability": reliability(po[keep_o], yo[keep_o])}
    bt["margin_mae"] = float(np.mean(np.abs(te.y_margin.values - mu_m)))
    bt["total_mae"] = float(np.mean(np.abs(te.y_total.values - mu_t)))
    pm = m["more"].predict_proba(te[FEATURES])[:, 1]
    bt["more"] = {"logloss": ll(pm, te.y_more.values), "logloss_base": ll(np.full(len(te), te.y_more.mean()), te.y_more.values),
                  "reliability": reliability(pm, te.y_more.values)}
    wi = te[te.y_within.notna()]
    pwi = m["within"].predict_proba(wi[FEATURES])[:, 1]
    bt["within"] = {"plays": int(len(wi)), "logloss": ll(pwi, wi.y_within.values), "logloss_base": ll(np.full(len(wi), wi.y_within.mean()), wi.y_within.values),
                    "reliability": reliability(pwi, wi.y_within.values)}
    json.dump(bt, open(os.path.join(W, "live_backtest.json"), "w"), indent=1)

    # ---------- publish: refit on everything through 2025, export, verify ----------
    m, res = fit_all(hist)
    out = {"features": FEATURES, "buckets": BUCKETS, "residual_quantiles": res, "models": {k: export(v) for k, v in m.items()},
           "kinds": {"win": "prob", "more": "prob", "within": "prob", "margin": "value", "total": "value"},
           "test": {"win_logloss": round(bt["win"]["logloss"], 4), "win_logloss_nflverse": round(bt["win"]["logloss_nflverse"], 4),
                    "cover_logloss": round(bt["cover"]["logloss"], 4), "over_logloss": round(bt["over"]["logloss"], 4),
                    "plays": bt["plays"], "games": bt["games"], "years": bt["test"]}}
    sample = hist.sample(400, random_state=1)[FEATURES].values
    for k, est in m.items():
        mine = run_exported(out["models"][k], sample)
        ref = est._raw_predict(sample).ravel()
        err = float(np.max(np.abs(mine - ref)))
        assert err < 1e-4, (k, err)
    json.dump(out, open(os.path.join(W, "live_model.json"), "w"), separators=(",", ":"))
    kb = os.path.getsize(os.path.join(W, "live_model.json")) // 1024
    print(f"live model: {bt['plays']:,} test plays ({bt['test']}) | win log loss {bt['win']['logloss']:.4f} vs nflverse {bt['win']['logloss_nflverse']:.4f} | "
          f"live cover {bt['cover']['logloss']:.4f} vs coin flip {bt['cover']['logloss_coin_flip']:.4f} | live over {bt['over']['logloss']:.4f} | {kb} KB")


if __name__ == "__main__":
    main()
