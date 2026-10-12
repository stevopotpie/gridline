"""Bring Steven's live sportsbook odds into Gridline.

The odds are pulled on Steven's PC (fetch_odds.py, The Odds API) and pushed to github.com/stevopotpie/gridline under odds/.
This script reads every snapshot there, matches each game to nflverse's game_id, and:
  1. patches data/games.csv so unplayed games use the current middle line across books (spread, total, moneylines)
     instead of nflverse's single line, when a recent pull has 3+ books on the game;
  2. writes odds_games.json with, per game: the middle line, the best number on each side and which book has it,
     how far apart the books are, and the closing line (last pull before kickoff) for games already started.
Played games keep nflverse's closing lines everywhere the model is trained or graded.

Usage: python3 -I odds.py DATA_DIR WORK_DIR
"""
import sys, os, json, glob, subprocess, statistics
from datetime import datetime, timezone, timedelta
import pandas as pd

D, W = sys.argv[1], sys.argv[2]
REPO = "https://github.com/stevopotpie/gridline.git"
SEASON = 2026
FRESH_DAYS = 4          # only override nflverse with a pull at most this old
MIN_BOOKS = 3
ET = "America/New_York"

ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "betrivers": "BetRivers", "williamhill_us": "Caesars",
         "espnbet": "ESPN BET", "fanatics": "Fanatics", "bovada": "Bovada", "betonlineag": "BetOnline", "mybookieag": "MyBookie",
         "betus": "BetUS", "lowvig": "LowVig", "ballybet": "Bally Bet", "hardrockbet": "Hard Rock", "betparx": "betPARX"}
book_name = lambda k: BOOKS.get(k, k)
payout = lambda o: o / 100 if o > 0 else 100 / abs(o)          # profit per 1 risked; higher is better for the bettor


def fetch_repo():
    # Running inside a checkout of the repo (GitHub Actions or a local clone): use its odds/ folder directly
    here = os.path.join(W, "odds", "snapshots")
    if os.path.isdir(here) and glob.glob(os.path.join(here, "nfl_odds_*.json")):
        return here
    dst = os.path.join(D, "oddsrepo")
    try:
        if os.path.isdir(os.path.join(dst, ".git")):
            subprocess.run(["git", "-C", dst, "pull", "-q", "--ff-only"], check=True, capture_output=True, timeout=120)
        else:
            subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, dst], check=True, capture_output=True, timeout=300)
    except Exception as e:
        print("odds: could not update the odds repo, using what is on disk:", e)
    return os.path.join(dst, "odds", "snapshots")


def load_snapshots(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "nfl_odds_*.json"))):
        try:
            snap = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        pulled = pd.Timestamp(snap["pulled_utc"])
        for g in snap.get("games", []):
            home, away = g["home_team"], g["away_team"]
            for b in g.get("bookmakers", []):
                r = dict(pulled=pulled, commence=pd.Timestamp(g["commence_time"]), event=g["id"],
                         away=ABBR.get(away, away), home=ABBR.get(home, home), book=b["key"])
                for m in b.get("markets", []):
                    o = {x["name"]: x for x in m.get("outcomes", [])}
                    if m["key"] == "spreads" and home in o and away in o:
                        r.update(hs=o[home].get("point"), hs_p=o[home]["price"], as_p=o[away]["price"])
                    elif m["key"] == "totals" and "Over" in o and "Under" in o:
                        r.update(tot=o["Over"].get("point"), ov_p=o["Over"]["price"], un_p=o["Under"]["price"])
                    elif m["key"] == "h2h" and home in o and away in o:
                        r.update(hml=o[home]["price"], aml=o[away]["price"])
                rows.append(r)
    return pd.DataFrame(rows)


def match_games(o, g):
    """nflverse game_id for each odds row: same teams, kickoff on the same Eastern date (or within a day)."""
    s = g[(g.season == SEASON) & (g.game_type == "REG")][["game_id", "gameday", "away_team", "home_team"]].copy()
    s["day"] = pd.to_datetime(s.gameday).dt.date
    o["day"] = o.commence.dt.tz_convert(ET).dt.date
    keys = o[["event", "away", "home", "day"]].drop_duplicates("event")
    out = {}
    for k in keys.itertuples():
        c = s[(s.away_team == k.away) & (s.home_team == k.home)]
        if not len(c): continue
        gap = c.day.map(lambda d: abs((d - k.day).days))
        if gap.min() <= 1: out[k.event] = c.loc[gap.idxmin(), "game_id"]
    o["game_id"] = o.event.map(out)
    return o[o.game_id.notna()]


def med(v):
    v = [x for x in v if pd.notna(x)]
    return float(statistics.median(v)) if v else None


def market(v):
    """The market's number: the point most books are dealing, ties going to the one nearest the median (never a 3.25)."""
    v = [float(x) for x in v if pd.notna(x)]
    if not v: return None
    m = statistics.median(v)
    counts = {x: v.count(x) for x in set(v)}
    top = max(counts.values())
    return min((x for x, c in counts.items() if c == top), key=lambda x: (abs(x - m), x))


def best(rows, point, price, prefer_high_point):
    """The bettor's best number: most favorable point, then the best price at that point."""
    r = rows[rows[point].notna() & rows[price].notna()] if point else rows[rows[price].notna()]
    if not len(r): return None
    if point:
        top = r[point].max() if prefer_high_point else r[point].min()
        r = r[r[point] == top]
    x = r.loc[r[price].map(payout).idxmax()]
    out = {"price": int(x[price]), "book": book_name(x.book)}
    if point: out["pt"] = float(x[point])
    return out


def summarize(rows):
    """One pull of one game across books."""
    hs, tot = rows.hs.dropna() if "hs" in rows else [], rows.tot.dropna() if "tot" in rows else []
    rows = rows.assign(nhs=-rows.hs if "hs" in rows else None)
    return {
        "books": int(rows.book.nunique()),
        "cons": {"hs": market(hs), "tot": market(tot), "hml": med(rows.get("hml", [])), "aml": med(rows.get("aml", []))},
        "range": {"hs": [float(min(hs)), float(max(hs))] if len(hs) else None, "tot": [float(min(tot)), float(max(tot))] if len(tot) else None},
        "best": {"home": best(rows, "hs", "hs_p", True), "away": best(rows, "nhs", "as_p", True),
                 "over": best(rows, "tot", "ov_p", False), "under": best(rows, "tot", "un_p", True),
                 "hml": best(rows, None, "hml", True), "aml": best(rows, None, "aml", True)},
        "lines": sorted([{"book": book_name(r.book), "hs": None if pd.isna(r.get("hs")) else float(r.hs),
                          "hs_p": None if pd.isna(r.get("hs_p")) else int(r.hs_p), "as_p": None if pd.isna(r.get("as_p")) else int(r.as_p),
                          "tot": None if pd.isna(r.get("tot")) else float(r.tot), "ov_p": None if pd.isna(r.get("ov_p")) else int(r.ov_p),
                          "un_p": None if pd.isna(r.get("un_p")) else int(r.un_p),
                          "hml": None if pd.isna(r.get("hml")) else int(r.hml), "aml": None if pd.isna(r.get("aml")) else int(r.aml)}
                         for _, r in rows.iterrows()], key=lambda x: x["book"]),
    }


def stamp(ts):
    t = ts.tz_convert(ET)
    return f"{t:%a %b} {t.day}, {t:%I:%M %p}".replace(" 0", " ") + " ET"


def main():
    gpath = os.path.join(D, "games.csv")
    g = pd.read_csv(gpath, low_memory=False)
    o = load_snapshots(fetch_repo())
    if not len(o):
        print("odds: no snapshots found, using nflverse lines only"); json.dump({}, open(os.path.join(W, "odds_games.json"), "w")); return
    for c in ["hs", "hs_p", "as_p", "tot", "ov_p", "un_p", "hml", "aml"]:
        if c not in o: o[c] = float("nan")
    o = match_games(o, g)
    pre = o[o.pulled < o.commence]                       # only lines posted before kickoff
    now = pd.Timestamp.now(tz="UTC")
    out, patched = {}, []
    for gid, rows in pre.groupby("game_id"):
        last = rows.pulled.max(); snap = rows[rows.pulled == last]
        kick = rows.commence.iloc[0]
        e = summarize(snap)
        e.update(asof=stamp(last), pulled=last.isoformat(), kickoff=kick.isoformat(), pulls=int(rows.pulled.nunique()))
        if kick <= now:                                   # started: this last pre-kickoff pull is the closing line
            e["close"] = {"hs": e["cons"]["hs"], "tot": e["cons"]["tot"], "hml": e["cons"]["hml"], "aml": e["cons"]["aml"],
                          "mins": int((kick - last).total_seconds() // 60), "asof": e["asof"]}
        else:
            # the first pull of the week, for line movement
            first = rows.pulled.min()
            if first < last:
                f = summarize(rows[rows.pulled == first])["cons"]
                e["open"] = {"hs": f["hs"], "tot": f["tot"], "asof": stamp(first)}
        out[gid] = e

    # patch nflverse's line for unplayed games with a fresh pull that has enough books
    idx = g.set_index("game_id")
    for gid, e in out.items():
        if gid not in idx.index or pd.notna(idx.at[gid, "home_score"]) or "close" in e: continue
        if e["books"] < MIN_BOOKS or now - pd.Timestamp(e["pulled"]) > timedelta(days=FRESH_DAYS): continue
        c = e["cons"]
        e["nfl"] = {"sl": None if pd.isna(idx.at[gid, "spread_line"]) else float(idx.at[gid, "spread_line"]),
                    "tl": None if pd.isna(idx.at[gid, "total_line"]) else float(idx.at[gid, "total_line"])}
        m = g.game_id == gid
        if c["hs"] is not None: g.loc[m, "spread_line"] = -c["hs"]
        if c["tot"] is not None: g.loc[m, "total_line"] = c["tot"]
        if c["hml"] is not None and c["aml"] is not None:
            g.loc[m, "home_moneyline"] = round(c["hml"]); g.loc[m, "away_moneyline"] = round(c["aml"])
        e["used"] = True
        patched.append(gid)
    g.to_csv(gpath, index=False)
    json.dump(out, open(os.path.join(W, "odds_games.json"), "w"))
    latest = o.pulled.max()
    print(f"odds: {o.pulled.nunique()} pulls, latest {stamp(latest)} | {len(out)} games matched | "
          f"{len(patched)} unplayed games now use the books' middle line | {sum('close' in e for e in out.values())} closing lines")
    for gid in patched:
        e = out[gid]; n = e["nfl"]
        if n["sl"] is not None and e["cons"]["hs"] is not None and abs(-e["cons"]["hs"] - n["sl"]) >= 0.5:
            print(f"   {gid}: spread {n['sl']:+.1f} (nflverse) -> {-e['cons']['hs']:+.1f} (books)")


if __name__ == "__main__":
    main()
