"""Gridline odds puller for GitHub Actions.

Same pull and same files as the PC script (fetch_odds.py in Documents\\gridline), but it writes
straight into this repo's odds/ folder. The workflow (.github/workflows/odds.yml) commits and pushes.

  odds/odds_latest.csv                    one row per game per book, newest pull
  odds/odds_consensus.csv                 one row per game: median line across books and the range
  odds/snapshots/nfl_odds_<UTC stamp>.json  full raw pull, kept as line history

The Odds API key comes from the ODDS_API_KEY repository secret. Standard library only.
Usage: python3 -I actions/fetch_odds.py <odds folder>
"""
import csv, json, os, statistics, sys, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# The Odds API full names -> nflverse abbreviations (what Gridline uses)
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


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)


def fetch(key):
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals",
                                "oddsFormat": "american", "dateFormat": "iso"})
    try:
        with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            h = r.headers
            log(f"API ok: {len(data)} games, cost {h.get('x-requests-last')} credits, "
                f"{h.get('x-requests-remaining')} left this month")
            return data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300].replace(key, "***")
        hint = {401: "the key was rejected; check the ODDS_API_KEY secret",
                429: "out of credits or too many requests"}.get(e.code, "")
        log(f"ERROR: API returned {e.code} {hint}: {body}")
    except urllib.error.URLError as e:
        log(f"ERROR: could not reach The Odds API ({e.reason})")
    sys.exit(1)


def rows_from(data, pulled):
    rows = []
    for g in data:
        home, away = g["home_team"], g["away_team"]
        for b in g.get("bookmakers", []):
            r = {"pulled_utc": pulled, "commence_utc": g["commence_time"], "event_id": g["id"],
                 "away": ABBR.get(away, away), "home": ABBR.get(home, home), "book": b["key"],
                 "book_updated_utc": b.get("last_update", ""),
                 "home_spread": "", "home_spread_price": "", "away_spread_price": "",
                 "total": "", "over_price": "", "under_price": "", "home_ml": "", "away_ml": ""}
            for m in b.get("markets", []):
                o = {x["name"]: x for x in m.get("outcomes", [])}
                if m["key"] == "spreads" and home in o and away in o:
                    r.update(home_spread=o[home].get("point"), home_spread_price=o[home]["price"],
                             away_spread_price=o[away]["price"])
                elif m["key"] == "totals" and "Over" in o and "Under" in o:
                    r.update(total=o["Over"].get("point"), over_price=o["Over"]["price"], under_price=o["Under"]["price"])
                elif m["key"] == "h2h" and home in o and away in o:
                    r.update(home_ml=o[home]["price"], away_ml=o[away]["price"])
            rows.append(r)
    return rows


def consensus(rows):
    games = {}
    for r in rows:
        games.setdefault((r["event_id"], r["commence_utc"], r["away"], r["home"]), []).append(r)
    out = []
    for (eid, t, away, home), rs in sorted(games.items(), key=lambda kv: kv[0][1]):
        sp = [r["home_spread"] for r in rs if r["home_spread"] != ""]
        tot = [r["total"] for r in rs if r["total"] != ""]
        out.append({"pulled_utc": rs[0]["pulled_utc"], "commence_utc": t, "event_id": eid, "away": away, "home": home,
                    "books": len(rs),
                    "home_spread": statistics.median(sp) if sp else "", "spread_min": min(sp) if sp else "",
                    "spread_max": max(sp) if sp else "",
                    "total": statistics.median(tot) if tot else "", "total_min": min(tot) if tot else "",
                    "total_max": max(tot) if tot else ""})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_odds.py <odds folder>")
    out = Path(sys.argv[1])
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        log("ERROR: the ODDS_API_KEY secret is not set (repo Settings > Secrets and variables > Actions).")
        sys.exit(1)

    data = fetch(key)
    if not data:
        log("No NFL games listed (off season); nothing to save.")
        return

    now = datetime.now(timezone.utc)
    pulled, stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%d_%H%MZ")
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    (out / "snapshots" / f"nfl_odds_{stamp}.json").write_text(
        json.dumps({"pulled_utc": pulled, "games": data}, indent=1), encoding="utf-8")

    rows = rows_from(data, pulled)
    cons = consensus(rows)
    if rows:
        write_csv(out / "odds_latest.csv", rows)
        write_csv(out / "odds_consensus.csv", cons)
    log(f"Saved {len(rows)} book lines for {len(cons)} games")

    for c in cons:
        if c["spread_min"] != "" and c["spread_max"] - c["spread_min"] >= 1.5:
            log(f"  Shop this one: {c['away']} @ {c['home']} home spread ranges {c['spread_min']} to {c['spread_max']}")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"stamp={stamp}\n")


if __name__ == "__main__":
    main()
