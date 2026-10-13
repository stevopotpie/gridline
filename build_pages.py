"""Assemble the data for both Gridline pages and write the finished HTML files.
Usage:
  python3 -I build_pages.py calc   MODEL_DIR   -> page_data.json + gridline_model.html (the calculator)
  python3 -I build_pages.py season MODEL_DIR   -> season_data.json + gridline_2026.html (needs season.py output)
  python3 -I build_pages.py betlog MODEL_DIR   -> gridline_bet_log.html (the bet log, with a snapshot of the model for its risk meter)
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

def odds():
    """Live book lines from odds.py: middle line, best number per side with its book, and closing lines."""
    f = J("odds_games.json")
    return json.load(open(f)) if os.path.exists(f) else {}

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
                             wx=x.wx_source, wind=r(x.wx_wind, 0), temp=r(x.wx_temp, 0), roof=x.wx_roof, lines=x.lines,
                             **early(x, r)) for x in p.itertuples()]
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
    if os.path.exists(J("player_quarter_splits.csv")):
        q = pd.read_csv(J("player_quarter_splits.csv"))
        cols = [f"{m}_q{i}" for m in ("pass_yds", "rush_yds", "rec_yds", "tds") for i in range(1, 5)]
        out["qsplits"] = {x.player_id: [int(x.g)] + [round(float(getattr(x, c)), 2) for c in cols] for x in q.itertuples()}
    if os.path.exists(J("player_quarter_backtest.json")):
        b = json.load(open(J("player_quarter_backtest.json")))
        out["qbt"] = {k: b[k] for k in ("pass_yds", "rush_yds", "rec_yds", "first_td") if k in b}
    if os.path.exists(J("q4_data.json")):
        q = json.load(open(J("q4_data.json")))
        out["q4"] = {"league": q["league"], "team": q["team"]}
    return out


def early(x, r):
    """First quarter and first half yards and the first touchdown chance (player_quarters.py), when present."""
    g = lambda c, n=0: r(getattr(x, c), n) if hasattr(x, c) else None
    return dict(py1=g("proj_pass_yds_q1"), pyh=g("proj_pass_yds_h1"), ry1=g("proj_rush_yds_q1"), ryh=g("proj_rush_yds_h1"),
                cy1=g("proj_rec_yds_q1"), cyh=g("proj_rec_yds_h1"), ftd=g("p_first_td", 3))


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
    if os.path.exists(J("live_model.json")):
        data["live"] = json.load(open(J("live_model.json")))       # live game model (live_model.py), run in the page
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
    fc = forecasts(); OD = odds()
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
        if r.game_id in OD: e["odds"] = OD[r.game_id]
        games.append(e)
    data = {"games": games, "proj": p.round(3).to_dict("records"), "grade": gr, "blowouts": blowouts(), "avoid": settings().get("avoid", []), "trust": settings().get("trust", []), "vsVegas": vs_vegas(), "vol": volatility(), "mvv": model_vs_vegas(), **players(),
            "updated": pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d, %I:%M %p ET")}
    json.dump(data, open(J("season_data.json"), "w"))
    html = open(template("season_template.html")).read().replace("__DATA__", json.dumps(data))
    open(J("gridline_2026.html"), "w").write(html)
    print("season page built:", len(games), "games |", gr["games"], "graded |", sum("fcT" in e for e in games), "with forecasts |", sum("odds" in e for e in games), "with book odds")

elif MODE == "betlog":
    # Snapshot of Gridline's numbers for the bet log's risk meter: every 2026 game, this slate's player props,
    # and calibration factors fit on the 2018 to 2025 backtest so the meter never oversells a side.
    from scipy import stats, optimize
    coef = json.load(open(J("coefficients_final.json")))
    sdM, sdT = coef["resid_sd_margin"], coef["resid_sd_total"]
    # Probit slopes, each fit on 2018 to 2025 games the model had not seen:
    #   veg: how well the Vegas spread alone predicts the winner; mod: the same for Gridline's margin alone
    #   sp:  how much of Gridline's gap from the Vegas spread shows up in covers (heavily shrunk)
    #   tot: the same for Gridline's gap from the Vegas total
    k = {"veg": 1.13, "mod": 0.95, "sp": 0.22, "tot": 0.6}
    if os.path.exists(J("backtest_games_final.csv")):
        b = pd.read_csv(J("backtest_games_final.csv")); b = b[b.spread_line.notna() & b.total_line.notna()]
        def fit(x, y):
            f = lambda c: -np.sum(y * np.log(stats.norm.cdf(c * x).clip(1e-6)) + (1 - y) * np.log((1 - stats.norm.cdf(c * x)).clip(1e-6)))
            return float(optimize.minimize_scalar(f, bounds=(0, 3), method="bounded").x)
        w = b[b.result != 0]
        k["veg"] = fit((w.spread_line / sdM).values, (w.result > 0).astype(int).values)
        k["mod"] = fit((w.m_final / sdM).values, (w.result > 0).astype(int).values)
        s_ = b[b.result != b.spread_line]
        k["sp"] = fit(((s_.m_final - s_.spread_line) / sdM).values, (s_.result > s_.spread_line).astype(int).values)
        t = b[b.total != b.total_line]
        k["tot"] = fit(((t.tm_final - t.total_line) / sdT).values, (t.total > t.total_line).astype(int).values)
    x = pd.read_csv(J("season_2026_games.csv"))
    f1 = lambda v, n=2: None if pd.isna(v) else round(float(v), n)
    OD = odds()
    def book_info(gid):
        e = OD.get(gid)
        if not e: return {}
        if "close" in e: return {"cl": e["close"]}
        return {"bn": e["best"], "bk": e["books"], "bas": e["asof"]}
    games = [dict(wk=int(r.week), d=str(r.gameday), a=r.away_team, h=r.home_team, pm=f1(r.p_margin), sl=f1(r.spread_line, 2),
                  tl=f1(r.total_line, 2), tot=f1(r.m_total if not pd.isna(r.m_total) else r.p_total), done=not pd.isna(r.home_score),
                  **book_info(r.game_id))
             for r in x.itertuples()]
    props, psd = [], {}
    if os.path.exists(J("player_props_2026.csv")):
        p = pd.read_csv(J("player_props_2026.csv"))
        gk = {r.game_id: f"{r.away_team} @ {r.home_team}" for r in x.itertuples()}
        for r in p.itertuples():
            props.append(dict(g=gk.get(r.game_id), wk=int(r.week), n=r.display_name, t=r.team, pos=r.position,
                              py=f1(r.proj_pass_yds, 1), ptd=f1(r.proj_pass_td), ry=f1(r.proj_rush_yds, 1), cy=f1(r.proj_rec_yds, 1),
                              rec=f1(r.proj_receptions), td=f1(r.proj_tds), td1=f1(r.p_td1, 3), td2=f1(r.p_td2, 3), p300=f1(r.p_pass300, 3),
                              ptd2=f1(r.p_ptd2, 3), ptd3=f1(r.p_ptd3, 3), r100=f1(r.p_rush100, 3), c100=f1(r.p_rec100, 3),
                              **early(r, f1)))
    if os.path.exists(J("player_backtest.json")):
        psd = {m: v.get("sd") for m, v in json.load(open(J("player_backtest.json"))).items()}
    if os.path.exists(J("player_quarter_backtest.json")):
        psd.update(json.load(open(J("player_quarter_backtest.json"))).get("sd", {}))
    model = {"asof": pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d"), "sdM": round(sdM, 2), "sdT": round(sdT, 2),
             "k": {kk: round(v, 3) for kk, v in k.items()}, "games": games, "props": props, "psd": psd}
    html = open(template("bet_log.html")).read().replace("__MODEL__", json.dumps(model))
    open(J("gridline_bet_log.html"), "w").write(html)
    print("bet log built:", len(games), "games |", len(props), "player props |", sum("cl" in g for g in games), "closing lines |", sum("bn" in g for g in games), "games with live odds | calibration", model["k"])
