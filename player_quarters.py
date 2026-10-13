"""Gridline players, step 5: quarter by quarter stats and early-game props.

Builds every player's yards, catches and touchdowns by quarter from play by play, then projects:
  - first quarter and first half passing, rushing and receiving yards: the game projection from player_props.py
    times the share of yards that come early for that position at that point spread. Favorites throw more early and
    run more late; underdogs the reverse. A player's own early share does not repeat (research/quarter_patterns.py)
    and adding it made the walk forward projections worse, so it is left out;
  - the chance each player scores the game's first touchdown: his expected touchdowns as a share of everyone's in
    the game, times the chance the first touchdown is scored by a player Gridline projects.
Every number is checked walk forward on 2019 to 2025 against what actually happened.

Usage: python3 -I player_quarters.py DATA_DIR WORK_DIR
Reads player_games.csv, team_games.csv, player_props_2026.csv and player_oos.csv (walk forward game projections
written by player_props.py). Adds early-game columns to player_props_2026.csv and writes player_quarter_splits.csv
and player_quarter_backtest.json.
"""
import json, os, sys
import numpy as np
import pandas as pd

STATS = ["pass_yds", "rush_yds", "rec_yds", "receptions", "tds"]
MARKETS = {"pass_yds": ["QB"], "rush_yds": ["RB", "QB", "WR"], "rec_yds": ["WR", "TE", "RB"]}
PRIOR_GAMES = 40        # used only in the backtest: a player's own early share, shrunk this hard, still made projections worse
TESTS = range(2019, 2026)


def _pbp(D, cols, years=range(2015, 2027)):
    out = []
    for y in years:
        f = os.path.join(D, f"pbp_{y}.parquet")
        if os.path.exists(f):
            out.append(pd.read_parquet(f, columns=cols))
    return pd.concat(out, ignore_index=True)


def quarter_stats(D):
    """One row per player per game: each stat by quarter (q1 to q4, q5 = overtime)."""
    p = _pbp(D, ["game_id", "play_id", "qtr", "posteam", "pass_attempt", "sack", "rush_attempt", "two_point_attempt",
                 "passer_player_id", "rusher_player_id", "receiver_player_id", "passing_yards", "rushing_yards",
                 "receiving_yards", "complete_pass", "touchdown", "td_player_id"])
    p = p[(p.two_point_attempt != 1) & p.posteam.notna()]
    p["q"] = p.qtr.clip(upper=5).astype(int)
    pa = p[(p.pass_attempt == 1) & (p.sack != 1)]
    parts = [
        pa[pa.passer_player_id.notna()].groupby(["game_id", "passer_player_id", "q"]).passing_yards.sum().rename("pass_yds")
          .reset_index().rename(columns={"passer_player_id": "player_id"}),
        p[(p.rush_attempt == 1) & p.rusher_player_id.notna()].groupby(["game_id", "rusher_player_id", "q"]).rushing_yards.sum()
          .rename("rush_yds").reset_index().rename(columns={"rusher_player_id": "player_id"}),
        pa[pa.receiver_player_id.notna()].groupby(["game_id", "receiver_player_id", "q"]).agg(rec_yds=("receiving_yards", "sum"),
          receptions=("complete_pass", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"}),
        p[(p.touchdown == 1) & p.td_player_id.notna()].groupby(["game_id", "td_player_id", "q"]).size().rename("tds")
          .reset_index().rename(columns={"td_player_id": "player_id"}),
    ]
    long = parts[0]
    for x in parts[1:]:
        long = long.merge(x, on=["game_id", "player_id", "q"], how="outer")
    long = long.fillna(0)
    wide = long.pivot_table(index=["game_id", "player_id"], columns="q", values=STATS, aggfunc="sum", fill_value=0)
    wide.columns = [f"{s}_q{q}" for s, q in wide.columns]
    for s in STATS:
        for q in range(1, 6):
            if f"{s}_q{q}" not in wide:
                wide[f"{s}_q{q}"] = 0.0
    return wide.reset_index()


def first_tds(D):
    """The first touchdown of every game: who scored it, for which team, in which quarter."""
    p = _pbp(D, ["game_id", "play_id", "qtr", "touchdown", "td_player_id", "td_team", "posteam"])
    t = p[p.touchdown == 1].sort_values(["game_id", "play_id"]).groupby("game_id").head(1)
    t = t.rename(columns={"td_player_id": "player_id"})
    t["season"] = t.game_id.str[:4].astype(int)
    return t[["game_id", "season", "qtr", "player_id", "td_team", "posteam"]]


# ---------------------------------------------------------------------------------------------------------------
def early_shares(qs, tg, pos_of, spread_of):
    """League share of regulation yards in Q1 and in the first half, by market, position and spread bucket."""
    out = {}
    for m, poss in MARKETS.items():
        for pos in poss:
            d = qs[qs.position == pos]
            for b, lo, hi in SPREAD_BUCKETS:
                x = d[(d.team_spread > lo) & (d.team_spread <= hi)]
                reg = x[[f"{m}_q{q}" for q in range(1, 5)]].sum().sum()
                if reg <= 0: continue
                out[(m, pos, b)] = (x[f"{m}_q1"].sum() / reg, (x[f"{m}_q1"].sum() + x[f"{m}_q2"].sum()) / reg)
    return out


SPREAD_BUCKETS = [("dog 7+", -99, -6.5), ("dog 3-6", -6.5, -2.5), ("close", -2.5, 2.5), ("fav 3-6", 2.5, 6.5), ("fav 7+", 6.5, 99)]
bucket = lambda s: next((b for b, lo, hi in SPREAD_BUCKETS if lo < s <= hi), "close") if pd.notna(s) else "close"


def player_shares(hist, m, norm_q1, norm_h1):
    """A player's early shares from his past games, shrunk toward the norm (PRIOR_GAMES pseudo games)."""
    reg = hist[[f"{m}_q{q}" for q in range(1, 5)]].sum(axis=1)
    g = hist.assign(reg=reg, q1=hist[f"{m}_q1"], h1=hist[f"{m}_q1"] + hist[f"{m}_q2"]).groupby("player_id").agg(
        n=("reg", "size"), reg=("reg", "sum"), q1=("q1", "sum"), h1=("h1", "sum"))
    avg = (g.reg / g.n).clip(lower=1)
    k = PRIOR_GAMES
    g["q1_dev"] = (g.q1 - norm_q1 * g.reg) / (g.n + k) / avg      # shrunk deviation from the norm, as a share
    g["h1_dev"] = (g.h1 - norm_h1 * g.reg) / (g.n + k) / avg
    return g[["q1_dev", "h1_dev"]]


def main(D, W):
    pg = pd.read_csv(os.path.join(W, "player_games.csv"), low_memory=False)
    tg = pd.read_csv(os.path.join(W, "team_games.csv"))
    tg["team_spread"] = np.where(tg.is_home == 1, tg.spread_line, -tg.spread_line)
    qs = quarter_stats(D)
    meta = pg[["game_id", "player_id", "team", "position", "season", "season_type", "game_date", "display_name"]]
    qs = qs.merge(meta, on=["game_id", "player_id"], how="inner").merge(tg[["game_id", "team", "team_spread"]], on=["game_id", "team"], how="left")
    qs = qs[qs.season_type == "REG"].copy()
    qs["bucket"] = qs.team_spread.map(bucket)
    hist = qs[qs.season.between(2015, 2025)]
    shares = early_shares(hist, tg, None, None)

    def norm(m, pos, b):
        return shares.get((m, pos, b)) or shares.get((m, pos, "close")) or (0.25, 0.5)

    # ---------- walk forward check of 1Q and 1H yards ----------
    bt = {"prior_games": PRIOR_GAMES}
    oos_path = os.path.join(W, "player_oos.csv")
    oos = pd.read_csv(oos_path) if os.path.exists(oos_path) else pd.DataFrame()
    sd_fit = {}
    for m, poss in MARKETS.items():
        o = oos[oos.market == m] if len(oos) else oos
        if not len(o): continue
        d = o.drop(columns="season").merge(qs, on=["game_id", "player_id"], how="inner")
        d = d[d.position.isin(poss) & d.season.between(2019, 2025)]
        res = {"n": int(len(d))}
        for per, col, qcols in (("q1", 0, [1]), ("h1", 1, [1, 2])):
            y = d[[f"{m}_q{q}" for q in qcols]].sum(axis=1)
            flat = d.mu * np.where(d.position == "QB", 0.25 if per == "q1" else 0.5, 0.25 if per == "q1" else 0.5)
            normed, personal = [], []
            for s in TESTS:
                te = d[d.season == s]
                tr_hist = qs[qs.season < s]
                sh = early_shares(tr_hist, tg, None, None)
                base = np.array([(sh.get((m, r.position, r.bucket)) or sh.get((m, r.position, "close")) or (0.25, 0.5))[col] for r in te.itertuples()])
                normed.append(pd.Series(te.mu.values * base, index=te.index))
                # player deviation relative to his own position norm, from games before this season
                ph = tr_hist[tr_hist.player_id.isin(te.player_id) & tr_hist.position.isin(poss)]
                devs = []
                for pos in poss:
                    nq1, nh1 = sh.get((m, pos, "close")) or (0.25, 0.5)
                    devs.append(player_shares(ph[ph.position == pos], m, nq1, nh1))
                dev = pd.concat(devs)
                dev = dev[~dev.index.duplicated()]
                add = te.player_id.map(dev[f"{per}_dev"]).fillna(0).values
                personal.append(pd.Series(te.mu.values * np.clip(base + add, 0.02, 0.98), index=te.index))
            normed, personal = pd.concat(normed).reindex(d.index), pd.concat(personal).reindex(d.index)
            res[per] = {"actual_avg": round(float(y.mean()), 1),
                        "mae_flat": round(float((y - flat).abs().mean()), 2),
                        "mae_position_spread": round(float((y - normed).abs().mean()), 2),
                        "mae_with_player_history": round(float((y - personal).abs().mean()), 2)}
            A = np.vstack([np.ones(len(y)), normed.values]).T      # spread of outcomes = a + b x projection (bet log meter)
            sd_fit[f"{m}_{per}"] = [float(v) for v in np.linalg.lstsq(A, np.abs(y.values - normed.values) * np.sqrt(np.pi / 2), rcond=None)[0]]
        bt[m] = res

    # ---------- first touchdown scorer ----------
    ft = first_tds(D)
    if len(oos):
        t = oos[oos.market == "tds"].merge(meta[["game_id", "player_id", "team"]], on=["game_id", "player_id"])
        t = t[t.season.between(2019, 2025)]
        t["lam_sum"] = t.groupby("game_id").mu.transform("sum")
        t = t.merge(ft[["game_id", "player_id"]].rename(columns={"player_id": "first_id"}), on="game_id", how="inner")
        t["y"] = (t.player_id == t.first_id).astype(int)
        covered = t.groupby("game_id").y.max()           # games where a projected player scored first
        probs = []
        for s in TESTS:
            prior_games = t[t.season < s]
            c = float(prior_games.groupby("game_id").y.max().mean()) if len(prior_games) else 0.85
            te = t[t.season == s]
            probs.append(pd.Series(c * te.mu / te.lam_sum, index=te.index))
        t["p"] = pd.concat(probs).reindex(t.index)
        t = t[t.p.notna()]
        ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
        cnt = t.groupby("game_id").size()
        naive = t.game_id.map(covered.mean() / cnt)
        rel = t.assign(bin=pd.cut(t.p, [0, .03, .06, .09, .12, .16, .25, 1])).groupby("bin", observed=True).agg(
            n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)
        bt["first_td"] = {"games": int(t.game_id.nunique()), "share_by_projected_player": round(float(covered.mean()), 3),
                          "logloss": round(ll(t.p.values, t.y.values), 4), "logloss_everyone_equal": round(ll(naive.values, t.y.values), 4),
                          "reliability": [{"bin": str(i), "n": int(r.n), "predicted": float(r.predicted), "actual": float(r.actual)} for i, r in rel.iterrows()]}
        c_now = float(covered.mean())
    else:
        c_now = 0.85
    bt["sd"] = sd_fit

    # ---------- this slate ----------
    pp = os.path.join(W, "player_props_2026.csv")
    props = pd.read_csv(pp)
    # spread for upcoming games: the posted line, else Gridline's projected margin (positive = home favored)
    sg = pd.read_csv(os.path.join(W, "season_2026_games.csv")).set_index("game_id")
    home_margin = sg.spread_line.fillna(sg.p_margin)
    props["team_spread"] = [home_margin.get(g, np.nan) * (1 if sg.home_team.get(g) == t else -1) for g, t in zip(props.game_id, props.team)]
    props["bucket"] = props.team_spread.map(bucket)
    for m, poss in MARKETS.items():
        col = f"proj_{m}"
        if col not in props: continue
        ok = props[col].notna() & props.position.isin(poss)
        base = props.loc[ok].apply(lambda r: norm(m, r.position, r.bucket), axis=1)
        q1, h1 = base.map(lambda x: x[0]).values, base.map(lambda x: x[1]).values
        props.loc[ok, f"proj_{m}_q1"] = (props.loc[ok, col] * q1).round(1)
        props.loc[ok, f"proj_{m}_h1"] = (props.loc[ok, col] * h1).round(1)
    if "proj_tds" in props:
        lam = props.proj_tds.fillna(0)
        props["p_first_td"] = (c_now * lam / props.assign(l=lam).groupby("game_id").l.transform("sum").replace(0, np.nan)).round(4)
        props["fair_first_td"] = props.p_first_td.map(lambda p: None if pd.isna(p) or p <= 0 or p >= 1 else
                                                      int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p)))
    props.drop(columns=["team_spread", "bucket"]).to_csv(pp, index=False)

    # ---------- per game averages by quarter since 2021, for the season board ----------
    ids = set(props.player_id) | set(pg[pg.season == 2026].player_id)
    s = qs[(qs.season >= 2021) & qs.player_id.isin(ids)]
    cols = [f"{m}_q{q}" for m in STATS for q in range(1, 5)]
    split = s.groupby("player_id").agg(g=("game_id", "nunique"), **{c: (c, "mean") for c in cols}).round(2).reset_index()
    split.to_csv(os.path.join(W, "player_quarter_splits.csv"), index=False)
    json.dump(bt, open(os.path.join(W, "player_quarter_backtest.json"), "w"), indent=1)

    msg = []
    for m in MARKETS:
        if m in bt:
            r = bt[m]["h1"]
            msg.append(f"{m} 1H miss {r['mae_position_spread']} (flat half {r['mae_flat']})")
    if "first_td" in bt:
        f = bt["first_td"]
        msg.append(f"first TD log loss {f['logloss']} vs {f['logloss_everyone_equal']} if everyone equal")
    print("quarters:", " | ".join(msg), f"| {int(props.p_first_td.notna().sum()) if 'p_first_td' in props else 0} first TD chances")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
