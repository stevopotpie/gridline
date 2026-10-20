"""Static, crawlable pages for search engines, built from the pipeline's output.

Called by site/assemble.py. Writes, under OUT:
  games/<season>-week-<n>-<away>-vs-<home>.html   one page per game: prediction before kickoff, graded recap after
  weeks/<season>-week-<n>.html                    every game that week
  teams/<team>.html                               a team's projections, playoff odds and schedule
  research/<topic>.html                           data-backed answers (wind, rain, injuries, records, halves)
  predictions.html, research.html                 hub pages
Every page is plain HTML (no script needed to read it) with a title, description, canonical URL, Open Graph tags,
breadcrumbs and schema.org JSON-LD. Returns the list of (path, lastmod) for sitemap.xml.
"""
import os, json, html, re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
TEAMS = {
    "ARI": ("Arizona Cardinals", "Cardinals"), "ATL": ("Atlanta Falcons", "Falcons"), "BAL": ("Baltimore Ravens", "Ravens"),
    "BUF": ("Buffalo Bills", "Bills"), "CAR": ("Carolina Panthers", "Panthers"), "CHI": ("Chicago Bears", "Bears"),
    "CIN": ("Cincinnati Bengals", "Bengals"), "CLE": ("Cleveland Browns", "Browns"), "DAL": ("Dallas Cowboys", "Cowboys"),
    "DEN": ("Denver Broncos", "Broncos"), "DET": ("Detroit Lions", "Lions"), "GB": ("Green Bay Packers", "Packers"),
    "HOU": ("Houston Texans", "Texans"), "IND": ("Indianapolis Colts", "Colts"), "JAX": ("Jacksonville Jaguars", "Jaguars"),
    "KC": ("Kansas City Chiefs", "Chiefs"), "LV": ("Las Vegas Raiders", "Raiders"), "LAC": ("Los Angeles Chargers", "Chargers"),
    "LA": ("Los Angeles Rams", "Rams"), "MIA": ("Miami Dolphins", "Dolphins"), "MIN": ("Minnesota Vikings", "Vikings"),
    "NE": ("New England Patriots", "Patriots"), "NO": ("New Orleans Saints", "Saints"), "NYG": ("New York Giants", "Giants"),
    "NYJ": ("New York Jets", "Jets"), "PHI": ("Philadelphia Eagles", "Eagles"), "PIT": ("Pittsburgh Steelers", "Steelers"),
    "SF": ("San Francisco 49ers", "49ers"), "SEA": ("Seattle Seahawks", "Seahawks"), "TB": ("Tampa Bay Buccaneers", "Buccaneers"),
    "TEN": ("Tennessee Titans", "Titans"), "WAS": ("Washington Commanders", "Commanders"),
}
full = lambda t: TEAMS.get(t, (t, t))[0]
nick = lambda t: TEAMS.get(t, (t, t))[1]
slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
team_path = lambda t: f"teams/{slug(full(t))}.html"
week_path = lambda season, wk: f"weeks/{season}-week-{wk}.html"
game_path = lambda season, g: f"games/{season}-week-{g['wk']}-{slug(nick(g['a']))}-vs-{slug(nick(g['h']))}.html"
esc = lambda s: html.escape(str(s), quote=True)
pct = lambda p: f"{round(100 * p)}%"


def fair(p):
    """American odds that break even at probability p."""
    if p is None or p <= 0 or p >= 1: return ""
    return f"−{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def line_txt(team, pts):
    """A team's spread the way a sportsbook shows it."""
    if pts is None: return "no line"
    if abs(pts) < 0.01: return f"{team} pick'em"
    return f"{team} {'+' if pts > 0 else '−'}{abs(pts):g}"


def page_style(repo):
    """The home page's design tokens and base styles, plus a few rules for article pages."""
    src = open(os.path.join(repo, "site", "index.html"), encoding="utf-8").read()
    m = re.search(r"<style>(.*?)</style>", src, re.S)
    extra = """
table.t{width:100%;border-collapse:collapse;font:400 .86rem var(--f-mono);font-variant-numeric:tabular-nums}
.t th,.t td{padding:7px 8px;border-top:1px solid var(--line);text-align:right;white-space:nowrap}
.t th:first-child,.t td:first-child{text-align:left;white-space:normal}
.t th{font:600 .7rem var(--f-body);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.scroll{overflow-x:auto}
.crumbs{font-size:.82rem;color:var(--muted)}
.crumbs a,.panel a,.lede a,li a{color:var(--accent)}
.lede{font-size:1.05rem;margin:.5rem 0 0;max-width:65ch}
.panel h3{font:700 1rem var(--f-display);text-transform:uppercase;letter-spacing:.04em;margin:14px 0 6px}
.panel ul{margin:0;padding-left:1.1rem}
.cta{display:inline-block;margin-top:6px;font-weight:600;color:var(--accent)}
.faq dt{font-weight:600;margin-top:10px}.faq dd{margin:2px 0 0}
main.wrap{grid-template-columns:minmax(0,1fr)}
main.wrap>*,section.panel{min-width:0;overflow-wrap:anywhere}
.t td{white-space:normal}
h1{overflow-wrap:break-word}
"""
    return "<style>" + (m.group(1) if m else "") + extra + "</style>"


class Site:
    def __init__(self, repo, out, base, head_fn, nav_fn, footer_fn, gh):
        self.repo, self.out, self.base, self.head, self.nav, self.footer, self.gh = repo, out, base, head_fn, nav_fn, footer_fn, gh
        self.style = page_style(repo)
        self.pages = []
        self.now = datetime.now(timezone.utc)

    def write(self, path, title, desc, body, crumbs, ld=(), og_type="article", lastmod=None):
        """One full static page. path is relative to the site root; links inside use ../ for nested pages."""
        depth = path.count("/")
        up = "../" * depth
        url = self.base + path
        trail = [("Home", "./")] + crumbs
        crumb_html = " › ".join(f'<a href="{up}{h if h != "./" else ""}">{esc(n)}</a>' if h else esc(n) for n, h in trail)
        bc = {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": i + 1, "name": n, **({"item": self.base + (h if h != "./" else "")} if h else {"item": url})}
            for i, (n, h) in enumerate(trail)]}
        lds = "".join(f'<script type="application/ld+json">{json.dumps(x, ensure_ascii=False)}</script>\n' for x in [bc, *ld])
        doc = (self.head + f"<title>{esc(title)}</title>\n" +
               f'<meta name="description" content="{esc(desc)}">\n<meta name="robots" content="index, follow, max-snippet:-1">\n'
               f'<meta property="og:type" content="{og_type}">\n<meta property="og:site_name" content="Gridline">\n'
               f'<meta property="og:title" content="{esc(title)}">\n<meta property="og:description" content="{esc(desc)}">\n'
               f'<meta property="og:url" content="{url}">\n<meta property="og:image" content="{self.base}og.png?v=2">\n'
               f'<meta name="twitter:card" content="summary_large_image">\n<link rel="canonical" href="{url}">\n'
               f'<link rel="icon" type="image/png" href="{up}icon.png?v=2">\n'
               '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
               '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
               + self.style + "\n" + lds + "</head>\n<body>\n" + self.nav(up) +
               f'\n<main class="wrap">\n<div class="crumbs">{crumb_html}</div>\n{body}\n'
               '<p class="note">Projections are research, not betting advice. Sports betting is not legal in every state. '
               'If gambling stops being fun, call or text 1-800-GAMBLER.</p>\n</main>\n' + self.footer(up) + "\n</body>\n</html>\n")
        f = os.path.join(self.out, path)
        os.makedirs(os.path.dirname(f) or self.out, exist_ok=True)
        open(f, "w", encoding="utf-8").write(doc)
        self.pages.append((path, (lastmod or self.now).strftime("%Y-%m-%d")))


def kickoff_iso(sched, gid):
    r = sched.get(gid)
    if not r or not r.get("gameday"): return None
    t = r.get("gametime") or "13:00"
    return datetime.strptime(f"{r['gameday']} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET).isoformat()


def load_schedule(repo, season):
    """Stadium, kickoff and roof from nflverse's games.csv (in REPO/data after a run), keyed by game_id."""
    import csv
    f = os.path.join(repo, "data", "games.csv")
    if not os.path.exists(f): return {}
    out = {}
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("season") == str(season): out[r["game_id"]] = r
    return out


def build(repo, out, base, head_fn, nav_fn, footer_fn, gh):
    S = json.load(open(os.path.join(repo, "season_data.json")))
    P = json.load(open(os.path.join(repo, "page_data.json"))) if os.path.exists(os.path.join(repo, "page_data.json")) else {}
    games = S["games"]; season = int(games[0]["id"][:4])
    sched = load_schedule(repo, season)
    site = Site(repo, out, base, head_fn, nav_fn, footer_fn, gh)
    gr = S.get("grade", {})
    proj = {p["team"]: p for p in S.get("proj", [])}
    props_by_game = {}
    for p in S.get("props", []): props_by_game.setdefault(p["g"], []).append(p)
    open_weeks = sorted({g["wk"] for g in games if g["hs"] is None})
    cur_wk = open_weeks[0] if open_weeks else max(g["wk"] for g in games)
    rec = lambda wl: f"{wl[0]}-{wl[1]}" if wl else "n/a"
    record_line = (f"Through {gr.get('through', '')}, Gridline has picked {rec(gr.get('su'))} straight up and {rec(gr.get('ats'))} against the spread "
                   f"in {gr.get('games', 0)} graded {season} games, with total leans of 1.5+ points at {rec(gr.get('ou_lean'))}.") if gr else ""

    # ---------- game pages ----------
    def game_page(g):
        a, h = g["a"], g["h"]; A, H = full(a), full(h)
        done = g["hs"] is not None
        sch = sched.get(g["id"], {})
        pm, sl, tl, mt, pt = g.get("pm"), g.get("sl"), g.get("tl"), g.get("mt"), g.get("pt")
        tot = mt if mt is not None else pt
        fav = h if (pm or 0) >= 0 else a
        hp = g.get("hw") or 0.5
        winp = hp if fav == h else 1 - hp
        ps_h = round((tot + (pm or 0)) / 2) if tot else None
        ps_a = round((tot - (pm or 0)) / 2) if tot else None
        vfav = None if sl is None else (h if sl > 0 else a if sl < 0 else None)
        vegas_line = "no line posted" if sl is None else ("pick'em" if vfav is None else line_txt(vfav, -abs(sl)))
        gap = None if sl is None or pm is None else pm - sl
        title = (f"{A} vs {H} Prediction, Week {g['wk']} {season}: Pick, Spread & Over/Under" if not done else
                 f"{nick(a)} vs {nick(h)} Week {g['wk']} {season}: Final Score vs Gridline's Prediction")
        hh, mm_ = (int(x) for x in str(g["time"]).split(":")) if ":" in str(g["time"]) else (13, 0)
        when = f"{g['day']}, {(hh - 1) % 12 + 1}:{mm_:02d} {'PM' if hh >= 12 else 'AM'} ET"
        stadium = sch.get("stadium") or ""
        if not done:
            lede = (f"Gridline projects the <b>{esc(full(fav))}</b> to beat the {esc(full(a if fav == h else h))} "
                    f"<b>{max(ps_a, ps_h)}-{min(ps_a, ps_h)}</b> ({pct(winp)} to win). Vegas has {esc(vegas_line)}"
                    f"{'' if tl is None else f' with the total at {tl:g}'}.") if ps_h is not None else ""
            desc = (f"{A} at {H}, {when}: Gridline projects {nick(fav)} {max(ps_a, ps_h)}-{min(ps_a, ps_h)}, {pct(winp)} to win. "
                    f"Spread and over/under lean vs Vegas{'' if tl is None else f' ({vegas_line}, total {tl:g})'}, weather and player props.")
        else:
            res = g["hs"] - g["as_"]
            right = (pm or 0) * res > 0
            ats = None if sl is None or res == sl or pm is None else ((pm > sl) == (res > sl))
            ou = None if tl is None or tot is None or g["hs"] + g["as_"] == tl else ((tot > tl) == (g["hs"] + g["as_"] > tl))
            ats_txt = "" if ats is None else (", and its side against the spread " + ("covered" if ats else "did not cover"))
            lede = (f"Final: <b>{esc(A)} {g['as_']}, {esc(H)} {g['hs']}</b>. Gridline had {esc(nick(fav))} by {abs(pm):.1f} "
                    f"({'right' if right else 'wrong'} winner){ats_txt}.")
            desc = (f"{A} {g['as_']}, {H} {g['hs']} (Week {g['wk']}, {season}). How Gridline's pregame projection of {nick(fav)} by {abs(pm):.1f} "
                    f"and a {tot:.1f} total did against the result and the Vegas line.")
        rows = []
        if pm is not None: rows.append(("Projected margin", f"{nick(fav)} by {abs(pm):.1f}", vegas_line))
        if tot is not None: rows.append(("Total (Vegas + model adjustments)", f"{tot:.1f}", "—" if tl is None else f"{tl:g}"))
        if pt is not None: rows.append(("Total (stats only)", f"{pt:.1f}", ""))
        rows.append(("Win probability", f"{nick(h)} {pct(hp)} · {nick(a)} {pct(1 - hp)}", ""))
        table = '<div class="scroll"><table class="t"><tr><th></th><th>Gridline</th><th>Vegas</th></tr>' + "".join(
            f"<tr><td>{esc(k)}</td><td>{esc(v)}</td><td>{esc(w)}</td></tr>" for k, v, w in rows) + "</table></div>"
        notes = []
        if gap is not None and not done:
            side = h if gap > 0 else a
            notes.append(f"<li>Spread: Gridline is {abs(gap):.1f} points more bullish on the {esc(nick(side))} than Vegas. "
                         + ("That clears the 3-point gap where the stats model went 54.8% against the spread in 2018 to 2025, a best case; spreads have no proven edge." if abs(gap) >= 3
                            else "That is under the 3-point gap the model needs, so it is no play."))
        if tot is not None and tl is not None and not done:
            dm = tot - tl
            notes.append(f"<li>Total: {'over' if dm > 0 else 'under'} {tl:g} by {abs(dm):.1f} points. " +
                         ("A 1.5+ point lean, the setup that went 55.1% in 2018 to 2025 (a best case)." if abs(dm) >= 1.5 else "Under the 1.5-point lean threshold, so no play."))
        if g.get("fcT") is not None and not done:
            wx_txt = "roof closed" if g.get("fcRoof") else f"forecast {round(g['fcT'])}°F, wind {round(g['fcW'])} mph" + (", rain or snow" if g.get("fcP") else "")
            notes.append(f"<li>Weather: {wx_txt} (as of {esc(g['fcAsOf'])})."
                         + (" Wind of 15+ mph has favored unders: see <a href=\"../research/does-wind-affect-nfl-totals.html\">how wind moves NFL totals</a>." if (g.get('fcW') or 0) >= 15 and not g.get('fcRoof') else "")
                         + (" Rain has been worth about 4 points under the closing total: see <a href=\"../research/does-rain-affect-nfl-over-under.html\">rain and NFL over/unders</a>." if g.get('fcP') else "") + "</li>")
        elif not done and g.get("dome", 0) >= 0.99:
            notes.append("<li>Weather: indoors.</li>")
        if g.get("aq") or g.get("hq"):
            notes.append(f"<li>Quarterbacks: {esc(g.get('aq') or 'TBD')} ({esc(nick(a))}) vs {esc(g.get('hq') or 'TBD')} ({esc(nick(h))})"
                         + (f". Backup starting: {', '.join(esc(x) for x in [g.get('aq') if g.get('aqb') else None, g.get('hq') if g.get('hqb') else None] if x)}" if (g.get('aqb') or g.get('hqb')) else "") + ".</li>")
        odds = g.get("odds") or {}
        if odds.get("best") and not done:
            b = odds["best"]
            def bx(k, team=None):
                x = b.get(k)
                if not x: return ""
                pt_ = f" {x['pt']:+g}" if "pt" in x and k in ("home", "away") else (f" {x['pt']:g}" if "pt" in x else "")
                p = x["price"]; price = f"+{p}" if p > 0 else f"−{abs(p)}"
                return f"{team or ''}{pt_} {price} ({esc(x['book'])})"
            notes.append(f"<li>Best numbers at the books (as of {esc(odds.get('asof', ''))}): {bx('away', nick(a))}; {bx('home', nick(h))}; "
                         f"over{bx('over')}; under{bx('under')}.</li>")
        props = sorted(props_by_game.get(g["id"], []), key=lambda p: -(p.get("td1") or 0))
        prop_html = ""
        if props and not done:
            td = [p for p in props if p.get("td1")][:6]
            qbs = [p for p in props if p.get("py")]
            yd = sorted([p for p in props if p.get("cy") or p.get("ry")], key=lambda p: -((p.get("cy") or 0) + (p.get("ry") or 0)))[:6]
            prop_html = ('<section class="panel"><h2>Player props</h2><h3>Anytime touchdown</h3><div class="scroll"><table class="t"><tr><th>Player</th><th>Chance</th><th>Fair odds</th></tr>'
                         + "".join(f"<tr><td>{esc(p['n'])} ({esc(p['t'])} {esc(p['pos'])})</td><td>{pct(p['td1'])}</td><td>{fair(p['td1'])}</td></tr>" for p in td) + "</table></div>")
            if qbs:
                prop_html += ('<h3>Passing yards</h3><div class="scroll"><table class="t"><tr><th>QB</th><th>Projection</th><th>300+ yards</th></tr>'
                              + "".join(f"<tr><td>{esc(p['n'])} ({esc(p['t'])})</td><td>{p['py']:.0f}</td><td>{pct(p['p300']) if p.get('p300') is not None else ''}</td></tr>" for p in qbs) + "</table></div>")
            if yd:
                prop_html += ('<h3>Rushing and receiving yards</h3><div class="scroll"><table class="t"><tr><th>Player</th><th>Rushing</th><th>Receiving</th></tr>'
                              + "".join(f"<tr><td>{esc(p['n'])} ({esc(p['t'])} {esc(p['pos'])})</td><td>{'' if p.get('ry') is None else format(p['ry'], '.0f')}</td><td>{'' if p.get('cy') is None else format(p['cy'], '.0f')}</td></tr>" for p in yd) + "</table></div>")
            prop_html += '<p class="note">Fair odds are the price where a bet would break even at Gridline\'s chance. The player model is graded against results but has not yet been tested against sportsbook prop lines.</p></section>'
        faq = []
        if not done and ps_h is not None:
            faq.append((f"Who will win {nick(a)} vs {nick(h)}?", f"Gridline projects the {full(fav)} to win {max(ps_a, ps_h)}-{min(ps_a, ps_h)}, with a {pct(winp)} chance to win outright."))
            if tl is not None: faq.append((f"What is the over/under prediction for {nick(a)} vs {nick(h)}?",
                                            f"Gridline's total is {tot:.1f} against a Vegas total of {tl:g}, {'over' if tot > tl else 'under'} by {abs(tot - tl):.1f} points."))
            if sl is not None: faq.append((f"Will the {nick(fav)} cover the spread?",
                                            f"Vegas has {vegas_line}. Gridline's margin is {abs(pm):.1f} for the {nick(fav)}, a gap of {abs(gap):.1f} points from the line."))
        faq.append(("How accurate is Gridline?", record_line or "Gridline grades every game of the season as it happens."))
        faq_html = '<section class="panel"><h2>Questions</h2><dl class="faq">' + "".join(f"<dt>{esc(q)}</dt><dd>{esc(ans)}</dd>" for q, ans in faq) + "</dl></section>"
        body = (f"<header><h1>{esc(nick(a))} vs {esc(nick(h))}</h1>"
                f"<p class=\"lede\">{lede}</p><p class=\"note\">{esc(when)}{' · ' + esc(stadium) if stadium else ''} · Week {g['wk']}, {season} NFL season</p></header>"
                f'<section class="panel"><h2>{"Gridline vs Vegas" if not done else "Pregame projection"}</h2>{table}'
                + (f"<ul>{''.join(notes)}</ul>" if notes else "") + "</section>" + prop_html + faq_html +
                f'<section class="panel"><h2>More</h2><ul><li><a href="../{week_path(season, g["wk"])}">All Week {g["wk"]} predictions</a></li>'
                f'<li><a href="../{team_path(a)}">{esc(A)} {season} projections</a></li><li><a href="../{team_path(h)}">{esc(H)} {season} projections</a></li>'
                f'<li><a href="../season.html">Season board with live odds</a> and the <a href="../model.html">matchup calculator</a> to change the QBs, weather or injuries</li></ul></section>')
        ev = {"@context": "https://schema.org", "@type": "SportsEvent", "name": f"{A} at {H}", "sport": "American football",
              "description": desc, "awayTeam": {"@type": "SportsTeam", "name": A}, "homeTeam": {"@type": "SportsTeam", "name": H},
              "competitor": [{"@type": "SportsTeam", "name": A}, {"@type": "SportsTeam", "name": H}],
              "eventStatus": "https://schema.org/EventScheduled", "eventAttendanceMode": "https://schema.org/OfflineEventAttendanceMode",
              "url": base + game_path(season, g)}
        k = kickoff_iso(sched, g["id"])
        if k: ev["startDate"] = k
        if stadium: ev["location"] = {"@type": "Place", "name": stadium}
        fq = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": ans}} for q, ans in faq]}
        site.write(game_path(season, g), title, desc, body,
                   [(f"Week {g['wk']}", week_path(season, g["wk"])), (f"{nick(a)} vs {nick(h)}", None)], ld=[ev, fq])

    for g in games: game_page(g)

    # ---------- week pages ----------
    weeks = sorted({g["wk"] for g in games})
    for wk in weeks:
        wg = [g for g in games if g["wk"] == wk]
        done_all = all(g["hs"] is not None for g in wg)
        rows = []
        for g in wg:
            pm, tot = g.get("pm") or 0, g.get("mt") if g.get("mt") is not None else g.get("pt")
            fav = g["h"] if pm >= 0 else g["a"]
            ph = round((tot + pm) / 2) if tot else ""; pa = round((tot - pm) / 2) if tot else ""
            res = f"{g['as_']}-{g['hs']}" if g["hs"] is not None else ""
            vline = "" if g.get("sl") is None else ("PK" if g["sl"] == 0 else line_txt(g["h"] if g["sl"] > 0 else g["a"], -abs(g["sl"])))
            rows.append(f'<tr><td><a href="../{game_path(season, g)}">{esc(nick(g["a"]))} at {esc(nick(g["h"]))}</a><br><span class="note">{esc(g["day"])}</span></td>'
                        f"<td>{esc(nick(fav))} {max(pa, ph) if tot else ''}-{min(pa, ph) if tot else ''}</td><td>{esc(vline)}</td>"
                        f"<td>{'' if tot is None else format(tot, '.1f')}</td><td>{'' if g.get('tl') is None else format(g['tl'], 'g')}</td><td>{res}</td></tr>")
        title = f"NFL Week {wk} Predictions {season}: Every Game's Pick, Spread & Over/Under" if not done_all else f"NFL Week {wk} {season} Results vs Gridline's Predictions"
        desc = f"Projected scores, win chances, spread and over/under leans for all {len(wg)} NFL Week {wk} {season} games, from a model tested against the betting market."
        body = (f"<header><h1>Week {wk} predictions</h1><p class=\"lede\">Gridline's projected score for every Week {wk} game of the {season} NFL season, next to the Vegas spread and total. {esc(record_line)}</p></header>"
                '<section class="panel"><div class="scroll"><table class="t"><tr><th>Game</th><th>Projected</th><th>Vegas line</th><th>Gridline total</th><th>Vegas total</th><th>Final</th></tr>'
                + "".join(rows) + "</table></div></section>"
                + '<section class="panel"><h2>Other weeks</h2><p>' + " · ".join(f'<a href="../{week_path(season, w)}">Week {w}</a>' if w != wk else f"<b>Week {w}</b>" for w in weeks) + "</p></section>")
        site.write(week_path(season, wk), title, desc, body, [("Predictions", "predictions.html"), (f"Week {wk}", None)],
                   ld=[{"@context": "https://schema.org", "@type": "ItemList", "name": f"NFL Week {wk} {season} predictions",
                        "itemListElement": [{"@type": "ListItem", "position": i + 1, "url": base + game_path(season, g)} for i, g in enumerate(wg)]}])

    # ---------- team pages ----------
    for t in sorted(TEAMS):
        tg = [g for g in games if t in (g["a"], g["h"])]
        if not tg: continue
        p = proj.get(t, {})
        rows = []
        for g in tg:
            home = g["h"] == t; opp = g["a"] if home else g["h"]
            pm = (g.get("pm") or 0) * (1 if home else -1)
            wp = (g.get("hw") or 0.5) if home else 1 - (g.get("hw") or 0.5)
            res = ""
            if g["hs"] is not None:
                us, them = (g["hs"], g["as_"]) if home else (g["as_"], g["hs"])
                res = f"{'W' if us > them else 'L' if us < them else 'T'} {us}-{them}"
            rows.append(f'<tr><td>Wk {g["wk"]}</td><td><a href="../{game_path(season, g)}">{"vs" if home else "at"} {esc(nick(opp))}</a></td>'
                        f"<td>{pm:+.1f}</td><td>{pct(wp)}</td><td>{res}</td></tr>")
        name = full(t)
        summ = (f"Gridline projects the {name} for {p['exp_wins']:.1f} wins (range {p['w10']:.0f} to {p['w90']:.0f}), "
                f"with a {pct(p['p_playoffs'])} chance to make the playoffs and {pct(p['p_div'])} to win the {esc(p.get('div', 'division'))}.") if p else ""
        vv = (S.get("vsVegas") or {}).get(t)
        vvs = (f" Against the spread in {season}: {vv['covers']}-{vv['fails']}" + (f"-{vv['pushes']}" if vv.get('pushes') else "")
               + f"; overs {vv['overs']}, unders {vv['unders']}.") if vv else ""
        title = f"{name} {season} Predictions: Projected Wins, Playoff Odds & Every Game's Pick"
        desc = (f"{name} {season}: " + (f"{p['exp_wins']:.1f} projected wins, {pct(p['p_playoffs'])} playoff odds. " if p else "") +
                "Gridline's projected margin and win chance for every game, graded as the season goes.")
        body = (f"<header><h1>{esc(name)} {season}</h1><p class=\"lede\">{esc(summ)}{esc(vvs)}</p>"
                + (f"<p class=\"note\">Current record {p['w']:.0f}-{p['l']:.0f}{'' if not p.get('t') else '-' + format(p['t'], '.0f')}. Projections come from 20,000 simulations of the remaining schedule.</p>" if p else "")
                + '</header><section class="panel"><h2>Schedule and projections</h2><div class="scroll"><table class="t"><tr><th>Week</th><th>Game</th><th>Proj. margin</th><th>Win chance</th><th>Result</th></tr>'
                + "".join(rows) + "</table></div></section>"
                + '<section class="panel"><h2>All teams</h2><p>' + " · ".join(f'<a href="../{team_path(x)}">{esc(nick(x))}</a>' for x in sorted(TEAMS, key=nick)) + "</p></section>")
        site.write(team_path(t), title, desc, body, [("Teams", "predictions.html#teams"), (name, None)],
                   ld=[{"@context": "https://schema.org", "@type": "SportsTeam", "name": name, "sport": "American football",
                        "memberOf": {"@type": "SportsOrganization", "name": "National Football League"}}])

    # ---------- research articles (numbers come from the latest build where available) ----------
    wind = P.get("wind") or []; rain = P.get("rain") or {}
    wtab = "".join(f"<tr><td>{w['lo']}{'–' + str(int(w['hi'])) if w['hi'] < 99 else '+'} mph</td><td>{w['n']:,}</td><td>{pct(w['under'])}</td></tr>" for w in wind)
    pub = "2026-10-18"
    ARTICLES = [
        ("does-wind-affect-nfl-totals", "Does Wind Affect NFL Over/Unders? What the Data Says",
         "How often NFL unders hit by kickoff wind speed, measured against the closing total, and how much 15+ mph wind costs quarterbacks.",
         f"""<p class="lede">Yes, but less than most bettors think, and the closing total already prices most of it. Wind only starts to matter around 10 to 15 mph.</p>
<section class="panel"><h2>Unders by kickoff wind</h2><div class="scroll"><table class="t"><tr><th>Kickoff wind</th><th>Games</th><th>Under hit</th></tr>{wtab}</table></div>
<p class="note">Outdoor regular season games, measured against the closing total, pushes excluded.</p></section>
<section class="panel"><h2>What it means</h2><ul><li>Calm games (under 10 mph) go over slightly more often than under.</li>
<li>From 10 to 19 mph, unders have hit about 55 to 56% against the close: a real but modest edge, and only before the market fully moves.</li>
<li>The 20+ mph buckets are small samples; don't read much into them.</li>
<li>Wind of 15+ mph has cost starting quarterbacks about 20 passing yards in Gridline's player model; rushing barely moves.</li>
<li>Gusts matter for kicking and deep throws, but forecasts and models use sustained wind.</li></ul></section>"""),
        ("does-rain-affect-nfl-over-under", "Does Rain Affect NFL Over/Unders? Rain Games vs the Closing Total",
         "NFL games played in rain or snow, measured against the closing total: how often the under hit, and how many points rain is worth.",
         f"""<p class="lede">Rain moves NFL totals more than wind does, and the market under-prices it. Since 2015, outdoor games with rain or snow at kickoff landed {abs(rain.get('wet', {}).get('vs', -3.4)):.1f} points under the closing total on average, and the under hit {pct(rain.get('wet', {}).get('under', 0.645))} of the time ({rain.get('wet', {}).get('n', 152)} games).</p>
<section class="panel"><h2>The numbers</h2><div class="scroll"><table class="t"><tr><th>Conditions</th><th>Games</th><th>Vs closing total</th><th>Under hit</th></tr>
<tr><td>Dry</td><td>{rain.get('dry', {}).get('n', '')}</td><td>{rain.get('dry', {}).get('vs', 0):+.1f}</td><td>{pct(rain.get('dry', {}).get('under', 0.5))}</td></tr>
<tr><td>Rain or snow</td><td>{rain.get('wet', {}).get('n', '')}</td><td>{rain.get('wet', {}).get('vs', 0):+.1f}</td><td>{pct(rain.get('wet', {}).get('under', 0.5))}</td></tr>
<tr><td>Rain or snow + wind 10+ mph</td><td>{rain.get('wetWind', {}).get('n', '')}</td><td>{rain.get('wetWind', {}).get('vs', 0):+.1f}</td><td>{pct(rain.get('wetWind', {}).get('under', 0.5))}</td></tr></table></div></section>
<section class="panel"><h2>How Gridline uses it</h2><p>Adding a rain-or-snow flag to Gridline's totals model (Vegas total plus adjustments) raised its 2+ point leans from 56.5% to 58.5% in walk-forward testing on 2018 to 2025, and its 1.5+ point leans from 54.4% to 55.1%. Rain is worth about 2.8 points off the total in that model, more with wind.</p>
<p class="note">Caveats: history uses the weather actually reported at kickoff, while bets are placed on forecasts, so real results will be weaker; and about 150 rain games is a small sample. Snow alone has barely moved totals (25 games).</p></section>"""),
        ("do-injuries-move-nfl-spreads", "Do NFL Injuries Beat the Spread? Testing Snap-Weighted Injury Reports",
         "Does weighting injured NFL players by snap share or target share improve spread and total predictions? A walk-forward test, 2018 to 2025.",
         """<p class="lede">Not against the line. Vegas prices the injury report before kickoff, and a team's recent form already reflects players who have been out.</p>
<section class="panel"><h2>The test</h2><p>Gridline counted regular starters ruled Out or Doubtful by position group. We tried weighting each absence by the player's snap share over the previous four games, and by targets plus carries for receivers and running backs.</p>
<div class="scroll"><table class="t"><tr><th>Version</th><th>Spread avg miss</th><th>ATS, 3+ pt gap</th><th>Total leans 1.5+</th></tr>
<tr><td>Count of starters out</td><td>10.04</td><td>54.8%</td><td>54.4%</td></tr><tr><td>No injury data at all</td><td>10.03</td><td>55.1%</td><td>54.2%</td></tr>
<tr><td>Snap-share weighted</td><td>10.06</td><td>54.9%</td><td>55.0%</td></tr><tr><td>Snap share + skill opportunity</td><td>10.06</td><td>54.9%</td><td>55.4%</td></tr></table></div></section>
<section class="panel"><h2>What the model learned anyway</h2><ul><li>A receiver or back with a quarter of his team's targets and carries is worth about 2.3 points.</li><li>An every-down linebacker is worth about 2.1 points; an offensive lineman about 0.3.</li><li>Those values are real, but the betting line already contains them.</li></ul></section>"""),
        ("does-team-record-predict-nfl-ats", "Does a Better Record Cover the Spread? NFL Records vs Against-the-Spread Results",
         "Does the team with the better win-loss record cover NFL spreads more often? 2,304 games from 2015 to 2025.",
         """<p class="lede">No. Records predict who wins, but the spread already accounts for them. The better-record team covered 49.9% of the time (2,019 games, week 4 on).</p>
<section class="panel"><h2>Findings</h2><ul><li>Record gap vs final margin: correlation 0.34. Record gap vs the Vegas line: 0.73. The market follows records more closely than results do.</li>
<li>Win-percentage gap of .400 or more: the better team covered only 48.7%.</li><li>Better record but getting points: 51.1% (419 games), noise.</li>
<li>Record gap vs where Gridline's market-adjusted model missed: −0.01. Nothing left to capture.</li></ul></section>"""),
        ("nfl-second-half-trends", "Are NFL Second-Half Trends Real? Halftime Splits vs Results",
         "Do NFL teams that dominate second halves keep doing it? Odd vs even games and early vs late season, 2015 to 2025.",
         """<p class="lede">No. A team's tendency to play better or worse after halftime does not repeat: odd vs even games correlate at −0.01, and the first six games vs the rest at 0.04.</p>
<section class="panel"><h2>Findings</h2><ul><li>Second-half point differential carries over a little (0.23), but only because good teams win both halves.</li>
<li>Once you know a team's full-game point differential, its second-half differential adds nothing.</li>
<li>Fourth-quarter edges and quarter-by-quarter player splits behave the same way; see the <a href="../season.html">season board</a> for the league-wide 4th quarter table.</li></ul></section>"""),
    ]
    for sl_, title, desc, content in ARTICLES:
        body = f"<header><h1>{esc(title.split('?')[0] + '?' if '?' in title else title)}</h1></header>{content}" + \
               '<section class="panel"><h2>More research</h2><ul>' + "".join(f'<li><a href="{s}.html">{esc(t)}</a></li>' for s, t, _, _ in ARTICLES if s != sl_) + "</ul></section>"
        site.write(f"research/{sl_}.html", f"{title} | Gridline", desc, body, [("Research", "research.html"), (title.split(":")[0].split("?")[0], None)],
                   ld=[{"@context": "https://schema.org", "@type": "Article", "headline": title[:110], "description": desc, "datePublished": pub,
                        "dateModified": site.now.strftime("%Y-%m-%d"), "author": {"@type": "Person", "name": "Steven Sutcliffe", "url": "https://github.com/stevopotpie"},
                        "publisher": {"@type": "Organization", "name": "Gridline", "url": base}, "mainEntityOfPage": base + f"research/{sl_}.html", "image": base + "og.png"}])

    # ---------- hubs ----------
    wk_links = "".join(f'<li><a href="{week_path(season, w)}">Week {w}</a>{" (this week)" if w == cur_wk else ""}</li>' for w in weeks)
    tm_links = "".join(f'<li><a href="{team_path(t)}">{esc(full(t))}</a></li>' for t in sorted(TEAMS, key=full))
    cur = [g for g in games if g["wk"] == cur_wk]
    cur_links = "".join(f'<li><a href="{game_path(season, g)}">{esc(full(g["a"]))} vs {esc(full(g["h"]))} prediction</a></li>' for g in cur)
    site.write("predictions.html", f"NFL Predictions {season}: Every Game, Every Week, Every Team | Gridline",
               f"Free NFL predictions for the {season} season: projected scores, spread and over/under picks for every game, team win projections and playoff odds.",
               f"<header><h1>NFL predictions {season}</h1><p class=\"lede\">Projected scores, spread and over/under leans for every game, from a model graded honestly against the betting market. {esc(record_line)}</p></header>"
               f'<section class="panel"><h2>Week {cur_wk} games</h2><ul>{cur_links}</ul></section>'
               f'<section class="panel"><h2>By week</h2><ul>{wk_links}</ul></section>'
               f'<section class="panel" id="teams"><h2>By team</h2><ul>{tm_links}</ul></section>', [("Predictions", None)], og_type="website")
    site.write("research.html", "NFL Betting Research: What Actually Beats the Line | Gridline",
               "Data-backed answers on weather, injuries, records and trends in NFL betting, each tested on games the model had not seen.",
               "<header><h1>Research</h1><p class=\"lede\">Questions bettors ask, answered with walk-forward tests on 2015 to 2025 NFL data. Most popular angles turn out to be priced in already; a few are not.</p></header>"
               '<section class="panel"><ul>' + "".join(f'<li><a href="research/{s}.html">{esc(t)}</a><br><span class="note">{esc(d)}</span></li>' for s, t, d, _ in ARTICLES) + "</ul></section>",
               [("Research", None)], og_type="website")
    return site.pages, cur_wk, [(game_path(season, g), f"{nick(g['a'])} vs {nick(g['h'])}") for g in cur]
