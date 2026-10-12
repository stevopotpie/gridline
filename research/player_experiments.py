"""Gridline players, step 3: find the best algorithm for each prop market.

Walk forward by season: each test season 2019 to 2025 is predicted by models trained only on earlier seasons.
Compares five ways to project the mean, then three ways to turn the mean into milestone probabilities.
Usage: python research/player_experiments.py WORK_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import player_features as F

W = sys.argv[1]   # folder with player_games.csv, team_games.csv and data/games.csv (run player_stats.py first)
pg = F.build(pd.read_csv(f"{W}/player_games.csv"), pd.read_csv(f"{W}/team_games.csv"), pd.read_csv(f"{W}/data/games.csv"))
pg = pg[pg.played]
pg = pg[(pg.season >= 2016) & (pg.n_prior >= 3)].copy()
pg["pos_rb"] = (pg.position == "RB").astype(int)
pg["pos_te"] = (pg.position == "TE").astype(int)

WEATHER = ["dome", "temp", "wind", "wind_over10", "cold"]
BASE = {
    "pass_yds":   ["e_pass_yds", "std_pass_yds", "e_pass_att", "e_ypa", "implied_pts", "team_spread", "t_team_dropbacks", "d_pass_yds", "d_dropbacks", "is_home"],
    "pass_td":    ["e_pass_td", "std_pass_td", "e_pass_att", "e_ypa", "implied_pts", "team_spread", "d_pts", "d_pass_yds", "is_home"],
    "rush_yds":   ["e_rush_yds", "std_rush_yds", "e_carries", "e_car_share", "e_ypc", "implied_pts", "team_spread", "t_team_carries", "d_rush_yds", "d_carries", "pos_rb", "is_home"],
    "rec_yds":    ["e_rec_yds", "std_rec_yds", "e_targets", "e_tgt_share", "e_ypt", "e_air_yds", "implied_pts", "team_spread", "t_team_targets", "d_rec_pos", "d_pass_yds", "pos_rb", "pos_te", "is_home"],
    "receptions": ["e_receptions", "std_receptions", "e_targets", "e_tgt_share", "e_catch", "implied_pts", "team_spread", "t_team_targets", "d_rec_pos", "pos_rb", "pos_te", "is_home"],
    "tds":        ["e_tds", "std_tds", "e_rz_carries", "e_rz_targets", "e_carries", "e_targets", "e_car_share", "e_tgt_share", "implied_pts", "team_spread", "d_pts", "pos_rb", "pos_te", "is_home"],
}
MILESTONES = {"pass_yds": [("pass_300", 300)], "rush_yds": [("rush_100", 100)], "rec_yds": [("rec_100", 100)],
              "tds": [("td_1", 1), ("td_2", 2)], "pass_td": [("ptd_2", 2), ("ptd_3", 3)], "receptions": []}
COUNT = {"tds", "pass_td", "receptions"}
TESTS = range(2019, 2026)


def structural(d, m):
    """Usage x efficiency x team volume x opponent, the way a pricing desk would build it by hand."""
    lg = {k: d[k].mean() for k in ["d_pass_yds", "d_rush_yds", "d_rec_pos", "implied_pts", "d_pts"]}
    imp = (d.implied_pts / lg["implied_pts"]).fillna(1)
    if m == "pass_yds":
        return d.e_pass_share.clip(0, 1) * d.t_team_dropbacks * d.e_ypa * (d.d_pass_yds / lg["d_pass_yds"]) ** 0.5 * imp ** 0.3
    if m == "rush_yds":
        return d.e_car_share * d.t_team_carries * d.e_ypc * (d.d_rush_yds / lg["d_rush_yds"]) ** 0.5 * imp ** 0.3
    if m == "rec_yds":
        return d.e_tgt_share * d.t_team_targets * d.e_ypt * (d.d_rec_pos / d.d_rec_pos.mean()) ** 0.5
    if m == "receptions":
        return d.e_tgt_share * d.t_team_targets * d.e_catch
    if m in ("tds", "pass_td"):
        return d[f"e_{m}"] * imp * (d.d_pts / lg["d_pts"]) ** 0.5


def fit_predict(kind, tr, te, m, cols):
    X, Xt = tr[cols].fillna(tr[cols].median()), te[cols].fillna(tr[cols].median())
    if kind == "ridge":
        r = Ridge(alpha=1.0).fit(X, tr[m])
        return np.clip(r.predict(Xt), 0, None)
    loss = "poisson" if m in COUNT else "squared_error"
    r = HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=80, l2_regularization=1.0, random_state=0).fit(X, tr[m])
    return np.clip(r.predict(Xt), 0, None)


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


results, prob_results, best_cfg = [], [], {}
for m, base in BASE.items():
    d = pg[F.eligible(pg, m)].copy()
    preds = {k: [] for k in ["season_avg", "ewma", "structural", "ridge", "ridge_weather", "gbm", "gbm_weather"]}
    ys, frames = [], []
    for s in TESTS:
        tr, te = d[d.season < s], d[d.season == s]
        preds["season_avg"].append(te[f"std_{m}"].values)
        preds["ewma"].append(te[f"e_{m}"].values)
        preds["structural"].append(structural(te, m).fillna(te[f"e_{m}"]).values)
        preds["ridge"].append(fit_predict("ridge", tr, te, m, base))
        preds["ridge_weather"].append(fit_predict("ridge", tr, te, m, base + WEATHER))
        preds["gbm"].append(fit_predict("gbm", tr, te, m, base))
        preds["gbm_weather"].append(fit_predict("gbm", tr, te, m, base + WEATHER))
        ys.append(te[m].values)
        frames.append(te)
    y = np.concatenate(ys)
    te_all = pd.concat(frames)
    for k, v in preds.items():
        p = np.concatenate(v)
        results.append({"market": m, "model": k, "n": len(y), "mae": float(np.mean(np.abs(y - p))),
                        "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "bias": float(np.mean(p - y))})
    # Yards props are lines near the median, so judge them on absolute error; counts on squared error
    res = pd.DataFrame([r for r in results if r["market"] == m]).sort_values("rmse" if m in COUNT else "mae")
    best = res.iloc[0].model
    mu = np.concatenate(preds[best])
    te_all = te_all.assign(mu=mu)
    best_cfg[m] = {"model": best, "features": base + (WEATHER if best.endswith("_weather") else [])}

    # Milestones: three ways to get a probability from the projection
    for name, k in MILESTONES[m]:
        hit = (y >= k).astype(int)
        out = {}
        # (a) parametric: Poisson for counts, normal with spread proportional to the mean for yards
        if m in COUNT:
            out["poisson"] = 1 - stats.poisson.cdf(k - 1, np.maximum(mu, 1e-6))
        # (b) empirical ratio and (c) direct classifier, both trained walk forward
        emp, clf = [], []
        for s in TESTS:
            tr = te_all[te_all.season < s]
            te = te_all[te_all.season == s]
            if len(tr) < 500:
                tr = te_all[te_all.season < s + 1].iloc[:0]
            if m not in COUNT:
                # Normal around the mean with sd = a + b * mean, fitted on training residuals
                if len(tr):
                    A = np.vstack([np.ones(len(tr)), tr.mu]).T
                    a, b = np.linalg.lstsq(A, np.abs(tr[m] - tr.mu) * np.sqrt(np.pi / 2), rcond=None)[0]
                else:
                    a, b = 15, 0.45
                sd = np.maximum(a + b * te.mu, 5)
                emp.append(1 - stats.norm.cdf(k - 0.5, te.mu, sd))
            if len(tr):
                c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=80, random_state=0)
                cols = best_cfg[m]["features"] + ["mu"]
                c.fit(tr[cols].fillna(0), (tr[m] >= k).astype(int))
                clf.append(c.predict_proba(te[cols].fillna(0))[:, 1])
            else:
                clf.append(np.full(len(te), np.nan))
        if m not in COUNT:
            out["normal_scaled"] = np.concatenate(emp)
        out["classifier"] = np.concatenate(clf)
        ok = ~np.isnan(out["classifier"])
        for meth, p in out.items():
            prob_results.append({"market": name, "method": meth, "n": int(ok.sum()), "hit_rate": float(hit[ok].mean()),
                                 "avg_prob": float(np.mean(p[ok])), "logloss": logloss(hit[ok], p[ok]),
                                 "brier": float(np.mean((p[ok] - hit[ok]) ** 2)),
                                 "base_logloss": logloss(hit[ok], np.full(ok.sum(), hit[ok].mean()))})

r = pd.DataFrame(results)
pr = pd.DataFrame(prob_results)
r.to_csv(f"{W}/player_model_comparison.csv", index=False)
pr.to_csv(f"{W}/player_prob_comparison.csv", index=False)
json.dump(best_cfg, open(f"{W}/player_best_models.json", "w"), indent=1)
pd.set_option("display.width", 160)
print(r.pivot(index="market", columns="model", values="mae").round(2))
print(r.pivot(index="market", columns="model", values="rmse").round(3))
print(pr[["market","method","n","hit_rate","avg_prob","logloss","base_logloss"]].round(4).to_string())
print(best_cfg)
