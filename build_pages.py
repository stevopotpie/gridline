"""Assemble the data for both Gridline pages and write the finished HTML files.
Usage:
  python3 -I build_pages.py calc   MODEL_DIR   -> page_data.json + gridline_model.html (the calculator)
  python3 -I build_pages.py season MODEL_DIR   -> season_data.json + gridline_2026.html (needs season.py output)
Optional MODEL_DIR/forecasts.csv (game_id,temp_f,wind_mph,roof_closed,as_of) sets kickoff weather for upcoming games."""
import sys, os, json
import numpy as np
import pandas as pd

MODE, M = sys.argv[1], sys.argv[2]
SEASON = 2026
J = lambda f: os.path.join(M, f)

def template(name):
    """Templates sit next to the scripts in a scheduled run, or in templates/ in the repo."""
    return J(name) if os.path.exists(J(name)) else J(os.path.join("templates", name))

def forecasts():
    f = J("forecasts.csv")
    return pd.read_csv(f).set_index("game_id") if os.path.exists(f) else pd.DataFrame()

def settings():
    f = J("settings.json")
    return json.load(open(f)) if os.path.exists(f) else {}

def vs_vegas():
    """Per team, 2026 so far: points over/under the spread (team margin minus the line) and the game total minus the posted total."""
    f = J("season_2026_games.csv")
    if not os.path.exists(f): return {}
    x = pd.read_csv(f); x = x[x.home_score.notna() & x.spread_line.notna()]
    out = {}
    for side, sg in (("home", 1), ("away", -1)):
        for r in x.itertuples():
            out.setdefault(getattr(r, f"{side}_team"), []).append((sg * (r.result - r.spread_line), r.total - r.total_line))
    res = {}
    for t, v in out.items():
        ms = np.array([a for a, _ in v]); ts = np.array([b for _, b in v if not pd.isna(b)])
        res[t] = {"g": len(ms), "covers": int((ms > 0).sum()), "fails": int((ms < 0).sum()), "pushes": int((ms == 0).sum()),
                  "net": round(float(ms.mean()), 1), "sum": round(float(ms.sum()), 1), "miss": round(float(np.abs(ms).mean()), 1),
                  "overs": int((ts > 0).sum()), "unders": int((ts < 0).sum()), "ou_push": int((ts == 0).sum()),
                  "ou_net": round(float(ts.mean()), 1) if len(ts) else 0.0, "ou_sum": round(float(ts.sum()), 1) if len(ts) else 0.0}
    return res

def volatility():
    f = J("volatility_2026.csv")
    if not os.path.exists(f): return {}
    v = pd.read_csv(f)
    return {r.team: {"index": round(float(r.index), 2), "label": r.label, "pf": round(float(r.pf), 1), "pa": round(float(r.pa), 1),
                     "off_epa": round(float(r.off_epa), 3), "def_epa": round(float(r.def_epa), 3), "yds": round(float(r.yds)), "yds_allowed": round(float(r.yds_allowed))}
            for r in v.itertuples()}

def model_vs_vegas():
    """Who was closer, the model or Vegas, by week; plus this week's biggest disagreements."""
    f = J("season_2026_games.csv")
    if not os.path.exists(f): return {}
    x = pd.read_csv(f)
    d = x[x.home_score.notna() & x.spread_line.notna()].copy()
    d["me"] = (d.p_margin - d.result).abs(); d["ve"] = (d.spread_line - d.result).abs()
    d["mt"] = (d.p_total - d.total).abs(); d["vt"] = (d.total_line - d.total).abs()
    weeks = [{"wk": int(w), "n": int(len(k)), "m": round(float(k.me.mean()), 1), "v": round(float(k.ve.mean()), 1),
              "mc": int((k.me < k.ve).sum()), "vc": int((k.ve < k.me).sum()), "mt": round(float(k.mt.mean()), 1), "vt": round(float(k.vt.mean()), 1),
              "mtc": int((k.mt < k.vt).sum()), "vtc": int((k.vt < k.mt).sum())} for w, k in d.groupby("week")]
    tot = {"n": int(len(d)), "m": round(float(d.me.mean()), 2), "v": round(float(d.ve.mean()), 2), "mc": int((d.me < d.ve).sum()), "vc": int((d.ve < d.me).sum()),
           "mt": round(float(d.mt.mean()), 2), "vt": round(float(d.vt.mean()), 2), "mtc": int((d.mt < d.vt).sum()), "vtc": int((d.vt < d.mt).sum())} if len(d) else {}
    u = x[x.home_score.isna() & x.spread_line.notna()].copy()
    u["diff"] = u.p_margin - u.spread_line
    u = u.reindex(u["diff"].abs().sort_values(ascending=False).index).head(6)
    upcoming = [{"a": r.away_team, "h": r.home_team, "wk": int(r.week), "sl": float(r.spread_line), "pm": round(float(r.p_margin), 1),
                 "diff": round(float(r.diff), 1), "tl": float(r.total_line), "pt": round(float(r.p_total), 1),
                 "mt": None if pd.isna(r.m_total) else round(float(r.m_total), 1)} for r in u.itertuples()]
    return {"weeks": weeks, "total": tot, "upcoming": upcoming}

def players():
    """Player props for the next two weeks, 2026 raw stats, weather splits and the player backtest."""
    out = {}
    if os.path.exists(J("player_props_2026.csv")):
        p = pd.read_csv(J("player_props_2026.csv"))
        r = lambda v, n=1: None if pd.isna(v) else round(float(v), n)
        out["props"] = [dict(g=x.game_id, n=x.display_name, t=x.team, pos=x.position, inj=x.injury if isinstance(x.injury, str) else "",
                             py=r(x.proj_pass_yds, 0), ptd=r(x.proj_pass_td, 2), p300=r(x.p_pass300, 3), ptd2=r(x.p_ptd2, 3), ptd3=r(x.p_ptd3, 3),
                             ry=r(x.proj_rush_yds, 0), r100=r(x.p_rush100, 3), cy=r(x.proj_rec_yds, 0), c100=r(x.p_rec100, 3),
                             rec=r(x.proj_receptions, 1), td=r(x.proj_tds, 2), td1=r(x.p_td1, 3), td2=r(x.p_td2, 3),
                             wx=x.wx_source, wind=r(x.wx_wind, 0), temp=r(x.wx_temp, 0), roof=x.wx_roof, lines=x.lines) for x in p.itertuples()]
    if os.path.exists(J("player_season_2026.csv")):
        s = pd.read_csv(J("player_season_2026.csv"))
        cols = ["games", "completions", "pass_att", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td", "targets", "receptions", "rec_yds", "rec_td", "tds"]
        out["pstats"] = [dict(id=x.player_id, n=x.display_name, t=x.team, pos=x.position, **{c: int(getattr(x, c)) for c in cols}) for x in s.itertuples()]
    if os.path.exists(J("player_weather_splits.csv")):
        w = pd.read_csv(J("player_weather_splits.csv"))
        sp = {}
        for x in w.itertuples():
            sp.setdefault(x.player_id, []).append([x.weather, int(x.g), round(x.pass_yds, 1), round(x.rush_yds, 1), round(x.rec_yds, 1), round(x.tds, 2)])
        out["splits"] = sp
    if os.path.exists(J("player_backtest.json")):
        out["pbt"] = json.load(open(J("player_backtest.json")))
    if os.path.exists(J("q4_data.json")):
        q = json.load(open(J("q4_data.json")))
        out["q4"] = {"league": q["league"], "team": q["team"]}
    return out


def blowouts():
    f = J("blowouts_2026.csv")
    if not os.path.exists(f): return {}
    b = pd.read_csv(f)
    return {r.team: {"season": int(r.bl_season), "last5": int(r.bl_last5), "flag": int(r.flag), "wins": int(r.bw_season), "prior3": int(r.bl_last3yrs)} for r in b.itertuples()}

if MODE == "calc":
    g = pd.read_csv(J("games_features_all.csv"), parse_dates=["gameday"], low_memory=False)
    cur = set(g[g.season == SEASON].home_team)
    teams = pd.read_csv(J("team_form.csv")); teams = teams[teams.team.isin(cur)].round(2)
    st = pd.read_csv(J("st_form.csv")); st = st[st.team.isin(cur)].round(3)
    cand = pd.read_csv(J("cand_form.csv")); cand = cand[cand.team.isin(cur)].round(4)
    fc = forecasts()

    # current week: the earliest 2026 week that still has unplayed games with a posted line
    open_ = g[(g.season == SEASON) & g.home_score.isna() & g.spread_line.notna()]
    wk = int(open_.week.min()) if len(open_) else None
    slate = []
    for r in (open_[open_.week == wk].sort_values(["gameday", "gametime"]).itertuples() if wk else []):
        e = dict(away=r.away_team, home=r.home_team, when=f"{pd.Timestamp(r.gameday):%a} {r.gametime}", homeLine=-r.spread_line,
                 total=r.total_line, dome=int(r.roof in ("dome", "closed")), roof=None if pd.isna(r.roof) else r.roof,
                 div=int(r.div_game), rest=int(r.rest_diff),
                 hqbId=r.home_qb_id if isinstance(r.home_qb_id, str) else None, aqbId=r.away_qb_id if isinstance(r.away_qb_id, str) else None,
                 hqbName=r.home_qb_name if isinstance(r.home_qb_name, str) else None, aqbName=r.away_qb_name if isinstance(r.away_qb_name, str) else None,
                 **{f"{s}_{p}": int(getattr(r, f"{s}_out_{p}")) for s in ("home", "away") for p in ["OL", "WR", "RB", "DL", "LB", "DB"]})
        if r.game_id in fc.index:
            f = fc.loc[r.game_id]
            e.update(temp=float(f.temp_f), wind=float(f.wind_mph), fcAsOf=str(f.as_of))
            if int(f.roof_closed) == 1: e["dome"] = 1
        slate.append(e)

    # historical under rate vs closing total by kickoff wind (outdoor games through last season)
    w = g[g.total.notna() & g.total_line.notna() & (g.dome == 0) & (g.wx_missing == 0) & (g.season < SEASON)].copy()
    w["diff"] = w.total - w.total_line; w = w[w["diff"] != 0]
    wind = [dict(lo=a, hi=b, n=int(len(s)), under=round(float((s["diff"] < 0).mean()), 3))
            for a, b in [(0, 9.99), (10, 14.99), (15, 19.99), (20, 24.99), (25, 99)]
            for s in [w[(w.wind_mph >= a) & (w.wind_mph <= b)]]]

    # quarterback options per team: everyone who started for them since last season, most recent first
    q = pd.read_csv(J("qb_form.csv")).set_index("qb_id").qb_epa
    rows = []
    for side in ("home", "away"):
        x = g[(g.season >= SEASON - 1) & g.home_score.notna()][["season", "gameday", f"{side}_team", f"{side}_qb_id", f"{side}_qb_name"]]
        x.columns = ["season", "gameday", "team", "id", "name"]; rows.append(x)
    s = pd.concat(rows).dropna(subset=["id"]).sort_values("gameday")
    qbs = {}
    for t, grp in s.groupby("team"):
        cur26 = grp[grp.season == SEASON]
        recent = cur26.id.iloc[-1] if len(cur26) else grp.id.iloc[-1]
        lst = [{"id": i, "name": grp[grp.id == i].name.iloc[-1], "epa": round(float(q.get(i, -0.10)), 3),
                "starts26": int((cur26.id == i).sum())} for i in grp.id.unique()[::-1]]
        qbs[t] = {"recent": recent, "qbs": lst}
    # add listed starters for this week's games who have not started for the team since last season
    for e in slate:
        for side in ("home", "away"):
            qid, qname, team = e.get(f"{side[0]}qbId"), e.get(f"{side[0]}qbName"), e[side]
            if qid and team in qbs and not any(x["id"] == qid for x in qbs[team]["qbs"]):
                qbs[team]["qbs"].append({"id": qid, "name": qname or "Listed starter", "epa": round(float(q.get(qid, -0.10)), 3), "starts26": 0})
    bk = pd.concat([g.loc[(g.season >= 2018) & (g.home_backup_qb == 1), "home_qb_epa"], g.loc[(g.season >= 2018) & (g.away_backup_qb == 1), "away_qb_epa"]])

    data = {"teams": teams.to_dict("records"), "slate": slate, "week": wk, "coef": json.load(open(J("coefficients_final.json"))),
            "bt": pd.read_csv(J("backtest_summary_final.csv")).to_dict("records"), "wind": wind,
            "st": st.to_dict("records"), "cand": cand.to_dict("records"), "qbs": qbs, "backup_epa": round(float(bk.median()), 3),
            "updated": pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d, %I:%M %p ET")}
    data["blowouts"] = blowouts(); data["avoid"] = settings().get("avoid", []); data["trust"] = settings().get("trust", []); data["vsVegas"] = vs_vegas(); data["vol"] = volatility()
    if os.path.exists(J("mkt_ratings.csv")):
        data["mkt"] = {r.team: float(r.mkt_rating) for r in pd.read_csv(J("mkt_ratings.csv")).itertuples()}
    if os.path.exists(J("last_snaps.csv")):
        data["snaps"] = {r.team: [float(r.off_snaps), float(r.def_snaps)] for r in pd.read_csv(J("last_snaps.csv")).itertuples()}
    if os.path.exists(J("comeback_model.json")):
        data["cb"] = json.load(open(J("comeback_model.json")))
    if os.path.exists(J("season_2026_projection.csv")):
        pr = pd.read_csv(J("season_2026_projection.csv"))
        data["records"] = {r.team: [int(r.w), int(r.l), int(r.t)] for r in pr.itertuples()}
    json.dump(data, open(J("page_data.json"), "w"))
    html = open(template("page_template.html")).read().replace("__DATA__", json.dumps(data))
    open(J("gridline_model.html"), "w").write(html)
    print("calculator built: week", wk, "|", len(slate), "games on the slate |", sum("temp" in e for e in slate), "with forecasts")

elif MODE == "season":
    x = pd.read_csv(J("season_2026_games.csv"), parse_dates=["gameday"]).sort_values(["week", "gameday", "gametime"])
    p = pd.read_csv(J("season_2026_projection.csv")); gr = json.load(open(J("season_2026_grade.json")))
    fc = forecasts()
    f1 = lambda v, n=1: None if pd.isna(v) else round(float(v), n)
    games = []
    for r in x.itertuples():
        e = dict(wk=int(r.week), day=f"{r.gameday:%a %b} {r.gameday.day}", time=str(r.gametime), a=r.away_team, h=r.home_team,
                 as_=None if pd.isna(r.away_score) else int(r.away_score), hs=None if pd.isna(r.home_score) else int(r.home_score),
                 sl=f1(r.spread_line), tl=f1(r.total_line), pm=f1(r.p_margin, 3), pt=f1(r.p_total, 3), mt=f1(r.m_total, 3),
                 hw=f1(r.p_home_win, 3), dome=f1(r.dome, 2), est=bool(r.wx_estimated),
                 hq=r.home_qb_proj if isinstance(r.home_qb_proj, str) else r.home_qb_name,
                 aq=r.away_qb_proj if isinstance(r.away_qb_proj, str) else r.away_qb_name,
                 hqb=int(r.home_backup_qb) if not pd.isna(r.home_backup_qb) else 0, aqb=int(r.away_backup_qb) if not pd.isna(r.away_backup_qb) else 0)
        if r.game_id in fc.index and e["hs"] is None:
            f = fc.loc[r.game_id]
            e.update(fcT=float(f.temp_f), fcW=float(f.wind_mph), fcRoof=int(f.roof_closed), fcAsOf=str(f.as_of))
        e["id"] = r.game_id
        games.append(e)
    data = {"games": games, "proj": p.round(3).to_dict("records"), "grade": gr, "blowouts": blowouts(), "avoid": settings().get("avoid", []), "trust": settings().get("trust", []), "vsVegas": vs_vegas(), "vol": volatility(), "mvv": model_vs_vegas(), **players(),
            "updated": pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d, %I:%M %p ET")}
    json.dump(data, open(J("season_data.json"), "w"))
    html = open(template("season_template.html")).read().replace("__DATA__", json.dumps(data))
    open(J("gridline_2026.html"), "w").write(html)
    print("season page built:", len(games), "games |", gr["games"], "graded |", sum("fcT" in e for e in games), "with forecasts")
