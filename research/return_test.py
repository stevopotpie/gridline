"""Test: do players under- or over-perform Gridline's projection in their first games back from an injury absence,
or when they play while listed Questionable?

Uses the walk forward projections in player_oos.csv (each season predicted by a model trained only on earlier seasons),
so the residuals are honest out of sample misses. A "return" is a player's game where he had played earlier that season,
missed the team's previous game, and was listed Out or Doubtful on the injury report for that missed game.

Usage: python3 -I research/return_test.py DATA_DIR WORK_DIR
Writes WORK_DIR/return_effects.json: per market, the average actual / projection ratio for each group with its
standard error, and the shrunk multiplier player_props.py applies.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

D, W = sys.argv[1], sys.argv[2]
SHRINK_N = 150            # a group with n rows keeps n / (n + 150) of its measured effect
MARKETS = ["pass_yds", "pass_td", "rush_yds", "rec_yds", "receptions", "tds"]


def return_flags(pg, tg, inj):
    """For every player game: games missed right before it (injury listed), and whether it is the 1st or 2nd game back,
    and whether he was listed Questionable that week."""
    tg = tg[tg.season_type == "REG"].sort_values(["team", "season", "week"])
    pg = pg[pg.season_type == "REG"]
    out = inj[inj.report_status.isin(["Out", "Doubtful"]) & (inj.game_type == "REG")]
    out_key = set(zip(out.season, out.week, out.team, out.gsis_id))
    q = inj[(inj.report_status == "Questionable") & (inj.game_type == "REG")]
    q_key = set(zip(q.season, q.week, q.team, q.gsis_id))
    played = set(zip(pg.season, pg.week, pg.player_id))
    weeks = {(t, s): grp.week.tolist() for (t, s), grp in tg.groupby(["team", "season"])}
    rows = []
    for r in pg[["season", "week", "team", "player_id", "game_id"]].itertuples(index=False):
        wk = weeks.get((r.team, r.season), [])
        if r.week not in wk:
            continue
        i = wk.index(r.week)
        prior = wk[:i]
        if not any((r.season, w, r.player_id) in played for w in prior):
            rows.append((r.game_id, r.player_id, 0, 0, (r.season, r.week, r.team, r.player_id) in q_key)); continue
        # count consecutive missed team games right before this one, all listed Out or Doubtful
        missed, j = 0, i - 1
        while j >= 0 and (r.season, wk[j], r.player_id) not in played and (r.season, wk[j], r.team, r.player_id) in out_key:
            missed += 1; j -= 1
        injured_gap = missed > 0 and (j < 0 or (r.season, wk[j], r.player_id) in played)
        back = 1 if injured_gap else 0
        if not back and i >= 2 and (r.season, wk[i - 1], r.player_id) in played:
            # second game back: previous game was itself a first game back
            k, m2 = i - 2, 0
            while k >= 0 and (r.season, wk[k], r.player_id) not in played and (r.season, wk[k], r.team, r.player_id) in out_key:
                m2 += 1; k -= 1
            if m2 > 0 and (k < 0 or (r.season, wk[k], r.player_id) in played):
                back, missed = 2, m2
        rows.append((r.game_id, r.player_id, back, missed, (r.season, r.week, r.team, r.player_id) in q_key))
    return pd.DataFrame(rows, columns=["game_id", "player_id", "back", "missed", "questionable"])


def group_of(r):
    if r.back == 1:
        return "back1_missed1" if r.missed == 1 else "back1_missed2plus"
    if r.back == 2:
        return "back2"
    return "questionable" if r.questionable else "normal"


if __name__ == "__main__":
    pg = pd.read_csv(f"{W}/player_games.csv")
    tg = pd.read_csv(f"{W}/team_games.csv")
    inj = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{D}/injuries_*.csv"))])
    oos = pd.read_csv(f"{W}/player_oos.csv")
    flags = return_flags(pg, tg, inj)
    flags["group"] = flags.apply(group_of, axis=1)
    res = {}
    for m in MARKETS:
        o = oos[oos.market == m].merge(pg[["game_id", "player_id", m]], on=["game_id", "player_id"]).merge(flags, on=["game_id", "player_id"])
        o = o[o.mu > (0.2 if m in ("pass_td", "tds") else 2 if m == "receptions" else 15)]
        # ratio of sums is steadier than the mean of ratios for count stats
        base = o[o.group == "normal"]
        base_ratio = base[m].sum() / base.mu.sum()       # the model's own overall lean, so groups are measured against it
        res[m] = {}
        for gname, x in o.groupby("group"):
            ratio = x[m].sum() / x.mu.sum() / base_ratio
            # bootstrap standard error
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(x), (300, len(x)))
            boots = x[m].values[idx].sum(1) / x.mu.values[idx].sum(1) / base_ratio
            n = len(x)
            res[m][gname] = {"n": int(n), "ratio": round(float(ratio), 4), "se": round(float(boots.std()), 4),
                             "applied": 1.0 if gname == "normal" else round(float(1 + (ratio - 1) * n / (n + SHRINK_N)), 4)}
    json.dump(res, open(f"{W}/return_effects.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    for m, g in res.items():
        print(m, {k: f"{v['ratio']:.3f}±{v['se']:.3f} n={v['n']} -> {v['applied']}" for k, v in g.items() if k != "normal"})
