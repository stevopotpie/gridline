"""Assemble the public GitHub Pages site from the built pages.

Usage: python3 -I site/assemble.py REPO_DIR OUT_DIR

The pipeline builds each page as a fragment (styles plus content, no <html> or <head>) so the same file also
works as a Claude artifact. For the public site this script gives each page a full document head, adds the
shared navigation bar and site map footer, and writes sitemap.xml and robots.txt.
"""
import os, sys, shutil, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seo
from datetime import datetime, timezone

REPO, OUT = sys.argv[1], sys.argv[2]
BASE = "https://stevopotpie.github.io/gridline/"
GH = "https://github.com/stevopotpie/gridline"

# (file on the site, built source, nav label, site map label, one line description)
PAGES = [
    ("index.html",    "gridline_ingame.html",  "This week",  "This week's games",   "Every game this week with the line, live scores, win chances and game notes"),
    ("season.html",   "gridline_2026.html",    "Season",     "2026 Season Board",   "Every game graded or projected, book odds, props, playoff odds"),
    ("model.html",    "gridline_model.html",   "Calculator", "Matchup Calculator",  "Set any matchup, line, weather, QBs and injuries"),
    ("bet-log.html",  "gridline_bet_log.html", "Bet log",    "Bet Log",             "Red to green risk meter, a ledger saved in your browser, retro bets and strategy replays"),
    ("how-it-works.html", None,                "How it works", "How it works",      "What Gridline is, how it is built and how it has done"),
    ("about.html",    None,                    "About",      "About Gridline",      "Who built Gridline, why, and what it is and isn't"),
]
# The home page is this week's games (the live board, Nov 18 2026). The old landing page is how-it-works.html, and
# in-game.html (the board's old address) redirects home.
ABOUT = [
    ("About the builder", BASE + "about.html"),
    ("Results and method", f"{GH}#results-at-a-glance"),
    ("What moves the numbers", f"{GH}#what-moves-the-numbers"),
    ("What did not work", f"{GH}#what-did-not-work"),
    ("Player props model", f"{GH}#player-props"),
    ("Run it yourself", f"{GH}#run-it-yourself"),
    ("Source code", GH),
    ("Courtline, the NBA companion model", "https://stevopotpie.github.io/courtline/"),
]

STYLE = """<style>
/* Shared site navigation and site map (added by site/assemble.py) */
body{margin:0}
.gl-nav{background:var(--board,#141a16);color:var(--board-ink,#f1f5f2);font-family:var(--f-body,system-ui,sans-serif)}
.gl-nav .gl-in{max-width:760px;margin:0 auto;padding:10px 16px;display:flex;align-items:center;gap:6px 16px;flex-wrap:wrap}
.gl-nav a{color:inherit;text-decoration:none}
.gl-brand{font:700 1.3rem/1 var(--f-display,"Arial Narrow",sans-serif);text-transform:uppercase;letter-spacing:.03em;margin-right:auto}
.gl-links{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;max-width:100%}
.gl-links::-webkit-scrollbar{display:none}
.gl-links a{font:600 .85rem var(--f-body,system-ui,sans-serif);padding:6px 10px;border-radius:999px;white-space:nowrap;opacity:.85}
.gl-links a:hover{opacity:1;background:rgba(255,255,255,.1)}
.gl-links a[aria-current="page"]{background:var(--accent,#1f6b45);color:var(--accent-ink,#fff);opacity:1}
.gl-nav a:focus-visible,.gl-map a:focus-visible{outline:2px solid var(--accent,#1f6b45);outline-offset:2px}
.gl-map{border-top:1px solid var(--line,#d5dcd7);margin-top:8px;font-family:var(--f-body,system-ui,sans-serif)}
.gl-map .gl-in{max-width:760px;margin:0 auto;padding:22px 16px 36px;display:grid;gap:20px}
@media (min-width:640px){.gl-map .gl-in{grid-template-columns:3fr 2fr}}
.gl-map h2{font:700 1rem var(--f-display,"Arial Narrow",sans-serif);text-transform:uppercase;letter-spacing:.05em;margin:0 0 8px;color:var(--ink,#18201b)}
.gl-map ul{list-style:none;margin:0;padding:0;display:grid;gap:7px}
.gl-map a{color:var(--accent,#1f6b45);font-weight:600;text-decoration:none}
.gl-map a:hover{text-decoration:underline}
.gl-map .gl-d{display:block;color:var(--muted,#5b665f);font-size:.8rem;font-weight:400}
.gl-map .gl-here{color:var(--ink,#18201b)}
.gl-map .gl-foot{grid-column:1/-1;font-size:.8rem;color:var(--muted,#5b665f);display:flex;flex-wrap:wrap;gap:4px 14px}
</style>"""


HUBS = [("predictions.html", "Predictions", "NFL predictions", "Every game, week and team: projected scores and leans"),
        ("research.html", "Research", "Betting research", "Wind, rain, injuries, records and trends, tested on unseen games")]
THIS_WEEK = []      # (path, label) for the current week's game pages, filled by seo.build


def nav(current, up=""):
    here = ' aria-current="page"'
    items = [(f, label) for f, _, label, _, _ in PAGES[:2]] + [(f, label) for f, label, _, _ in HUBS] + [(f, label) for f, _, label, _, _ in PAGES[2:]]
    links = "".join(f'<a href="{up}{"" if f == "index.html" else f}"{here if f == current else ""}>{label}</a>' for f, label in items).replace('href=""', 'href="./"')
    links += '<a href="https://claude.ai/artifact/PnwqjUsuYKyyWzaPuQuHyL">The Bartop</a>'   # shared NFL and NBA question board
    links += '<a href="https://stevopotpie.github.io/courtline/">Courtline (NBA)</a>'
    links += f'<a href="{GH}">GitHub</a>'
    return (f'<nav class="gl-nav" aria-label="Gridline pages"><div class="gl-in">'
            f'<a class="gl-brand" href="{up or "./"}"{here if current == "index.html" else ""}>Gridline</a>'
            f'<div class="gl-links">{links}</div></div></nav>')


def sitemap_footer(current, up=""):
    def item(f, name, desc):
        if f == current:
            return f'<li><span class="gl-here">{name} (you are here)</span><span class="gl-d">{desc}</span></li>'
        return f'<li><a href="{up}{"" if f == "index.html" else f}">{name}</a><span class="gl-d">{desc}</span></li>'.replace('href=""', 'href="./"')
    tools = "".join(item(f, name, desc) for f, _, _, name, desc in PAGES) + "".join(item(f, name, desc) for f, _, name, desc in HUBS)
    about = "".join(f'<li><a href="{u}">{name}</a></li>' for name, u in ABOUT)
    week = "".join(f'<li><a href="{up}{p}">{lbl} prediction</a></li>' for p, lbl in THIS_WEEK)
    return (f'<footer class="gl-map"><div class="gl-in">'
            f'<nav aria-label="Site map"><h2>Site map</h2><ul>{tools}</ul></nav>'
            f'<nav aria-label="About Gridline"><h2>About</h2><ul>{about}</ul></nav>'
            + (f'<nav aria-label="This week" style="grid-column:1/-1"><h2>This week\'s games</h2><ul style="grid-template-columns:repeat(auto-fill,minmax(210px,1fr))">{week}</ul></nav>' if week else '') +
            f'<div class="gl-foot"><a href="{up}about.html">Built by Steven Sutcliffe</a>'
            f'<a href="https://linkedin.com/in/steven-sutcliffe">LinkedIn</a><a href="https://github.com/stevopotpie">GitHub</a>'
            f'<span>Projections are research, not betting advice. Sports betting is not legal in every state. If gambling stops being fun, call or text 1-800-GAMBLER.</span></div>'
            f'</div></footer>')


def meta(f):
    """Link preview (LinkedIn, Slack, texts), search snippet and browser tab icon for one page."""
    title = {"index.html": "Gridline: This Week's NFL Games, Lines and Live Win Chances | The Ultimate Sports Betting Sandbox",
             "how-it-works.html": "Gridline: NFL Predictions, Spread & Over/Under Picks Tested Against Vegas",
             "season.html": "NFL 2026 Season Predictions, Odds & Player Props | Gridline Season Board",
             "model.html": "NFL Matchup Calculator: Spread, Total & Live Win Probability | Gridline",
             "bet-log.html": "Sports Bet Tracker with Win Probability Meter | Gridline Bet Log",
             "about.html": "About Gridline: Built by Steven Sutcliffe"}[f]
    desc = {"index.html": "The Ultimate Sports Betting Sandbox: every NFL game this week with the spread and total, live scores, Gridline's chance that each team wins, and game notes on injuries, weather, history, QBs and coaches.",
            "how-it-works.html": "Free NFL predictions for every game: projected scores, spread and over/under picks and player props from public play by play data, graded honestly against the betting market.",
            "season.html": "Every 2026 NFL game projected or graded: Gridline vs Vegas spreads and totals, the best odds at the books, player props, anytime touchdown odds and playoff chances.",
            "model.html": "Set any NFL matchup, line, weather, quarterbacks and injuries and see the projected spread, total and score, plus a live in-game win probability model.",
            "bet-log.html": "Log your NFL bets and see each one's chance to win on a red to green meter, whether the model beats the price, and your closing line value.",
            "about.html": "Gridline is built by Steven Sutcliffe, a former bartender turned data analyst who puts game day football theories to an honest test against every NFL game since 2015."}[f]
    url = BASE + ("" if f == "index.html" else f)
    ld = ""
    if f == "index.html":
        ld = ('<script type="application/ld+json">' + json.dumps([
            {"@context": "https://schema.org", "@type": "WebSite", "name": "Gridline", "url": BASE,
             "description": desc, "author": {"@type": "Person", "name": "Steven Sutcliffe", "url": "https://github.com/stevopotpie",
                                              "sameAs": ["https://github.com/stevopotpie", "https://linkedin.com/in/steven-sutcliffe"]}},
            {"@context": "https://schema.org", "@type": "Dataset", "name": "Gridline NFL projections",
             "description": "Weekly NFL spread, total and player projections built from nflverse play by play data, with walk-forward backtests 2018 to 2025.",
             "creator": {"@type": "Person", "name": "Steven Sutcliffe"}, "license": "https://opensource.org/licenses/MIT", "url": GH,
             "isBasedOn": "https://github.com/nflverse/nflverse-data"}]) + "</script>\n")
    elif f == "about.html":
        ld = ('<script type="application/ld+json">' + json.dumps({"@context": "https://schema.org", "@type": "AboutPage", "url": BASE + f, "description": desc,
              "mainEntity": {"@type": "Person", "name": "Steven Sutcliffe", "jobTitle": "Data Analyst", "url": BASE + f,
                             "sameAs": ["https://github.com/stevopotpie", "https://linkedin.com/in/steven-sutcliffe"]}}) + "</script>\n")
    else:
        ld = ('<script type="application/ld+json">' + json.dumps({"@context": "https://schema.org", "@type": "WebApplication", "name": title,
              "url": BASE + f, "applicationCategory": "SportsApplication", "operatingSystem": "Any", "description": desc,
              "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}}) + "</script>\n")
    return ld + (f'<title>{title}</title>\n' if f not in ("how-it-works.html", "about.html") else "") + f'<meta name="description" content="{desc}">\n' + (
        f'<meta property="og:type" content="website">\n<meta property="og:site_name" content="Gridline">\n'
        f'<meta property="og:title" content="{title}">\n<meta property="og:description" content="{desc}">\n'
        f'<meta property="og:url" content="{url}">\n<meta property="og:image" content="{BASE}og.png?v=2">\n'
        f'<meta property="og:image:width" content="2400">\n<meta property="og:image:height" content="1254">\n'
        f'<meta property="og:image:alt" content="Gridline: spreads, totals and player props, tested honestly against the betting market">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        f'<link rel="canonical" href="{url}">\n<link rel="icon" type="image/png" href="icon.png?v=2">\n' + PWA)


# Installable app (Nov 15 2026): manifest, phone home screen settings and the service worker, on every page
PWA = ('<link rel="manifest" href="/gridline/manifest.webmanifest">\n<meta name="theme-color" content="#141a16">\n'
       '<meta name="mobile-web-app-capable" content="yes">\n<meta name="apple-mobile-web-app-capable" content="yes">\n'
       '<meta name="apple-mobile-web-app-title" content="Gridline">\n<meta name="apple-mobile-web-app-status-bar-style" content="black">\n'
       '<link rel="apple-touch-icon" href="/gridline/apple-touch-icon.png">\n'
       "<script>if('serviceWorker' in navigator)addEventListener('load',()=>navigator.serviceWorker.register('/gridline/sw.js',{scope:'/gridline/'}).catch(()=>{}))</script>\n")

HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')

os.makedirs(OUT, exist_ok=True)
# static pages for search: games, weeks, teams, research (site/seo.py); fills THIS_WEEK for the footer first
SEO_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + PWA + STYLE + "\n")
extra_pages = []
if os.path.exists(os.path.join(REPO, "season_data.json")):
    extra_pages, cur_wk, THIS_WEEK[:] = seo.build(REPO, OUT, BASE, SEO_HEAD, lambda up: nav("", up), lambda up: sitemap_footer("", up), GH)
_v = re.search(r'<meta name="google-site-verification"[^>]*>\n?', open(os.path.join(REPO, "site", "how-it-works.html"), encoding="utf-8").read())
VERIFY = _v.group(0) if _v else ""
for f, src, *_ in PAGES:
    if src is None:
        html = open(os.path.join(REPO, "site", f), encoding="utf-8").read()
        html = html.replace(VERIFY, "", 1) if f != "index.html" and VERIFY else html
        html = html.replace("</head>", meta(f) + STYLE + "\n</head>", 1)
        html = html.replace("<!-- gridline:nav -->", nav(f), 1).replace("<!-- gridline:sitemap -->", sitemap_footer(f), 1)
    else:
        frag = open(os.path.join(REPO, src), encoding="utf-8").read()
        frag = re.sub(r"<title>.*?</title>\s*", "", frag, count=1)      # the public page gets its search title from meta()
        i, j = frag.find('<div class="wrap">'), frag.find("<script")
        if i < 0 or j < i:
            sys.exit(f"assemble: could not find the page layout in {src}")
        # nav goes right before the page content; the site map after it and before the page's script
        html = HEAD + (VERIFY if f == "index.html" else "") + meta(f) + STYLE + "\n" + frag[:i] + nav(f) + "\n" + frag[i:j] + sitemap_footer(f) + "\n" + frag[j:]
    open(os.path.join(OUT, f), "w", encoding="utf-8").write(html)

# the board's old address sends visitors home
open(os.path.join(OUT, "in-game.html"), "w", encoding="utf-8").write(
    f'<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<title>Gridline: this week\'s games</title>\n<link rel="canonical" href="{BASE}">\n'
    '<meta http-equiv="refresh" content="0; url=./">\n<meta name="robots" content="noindex">\n<p>The live board moved to the <a href="./">Gridline home page</a>.</p>\n')

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
urls = "".join(f"  <url><loc>{BASE}{'' if f == 'index.html' else f}</loc><lastmod>{today}</lastmod></url>\n" for f, *_ in PAGES)
urls += "".join(f"  <url><loc>{BASE}{p}</loc><lastmod>{d}</lastmod></url>\n" for p, d in extra_pages)
open(os.path.join(OUT, "sitemap.xml"), "w").write(
    f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}</urlset>\n')
open(os.path.join(OUT, "robots.txt"), "w").write(f"User-agent: *\nAllow: /\nSitemap: {BASE}sitemap.xml\n")
open(os.path.join(OUT, ".nojekyll"), "w").close()
for asset in ("og.png", "icon.png", "icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png", "manifest.webmanifest", "sw.js"):
    shutil.copy(os.path.join(REPO, "site", asset), os.path.join(OUT, asset))
print("site assembled:", ", ".join(f for f, *_ in PAGES), f"+ {len(extra_pages)} search pages, sitemap.xml, robots.txt")
