"""Assemble the public GitHub Pages site from the built pages.

Usage: python3 -I site/assemble.py REPO_DIR OUT_DIR

The pipeline builds each page as a fragment (styles plus content, no <html> or <head>) so the same file also
works as a Claude artifact. For the public site this script gives each page a full document head, adds the
shared navigation bar and site map footer, and writes sitemap.xml and robots.txt.
"""
import os, sys, shutil
from datetime import datetime, timezone

REPO, OUT = sys.argv[1], sys.argv[2]
BASE = "https://stevopotpie.github.io/gridline/"
GH = "https://github.com/stevopotpie/gridline"

# (file on the site, built source, nav label, site map label, one line description)
PAGES = [
    ("index.html",    None,                    "Home",       "Home",                "What Gridline is and how it has done"),
    ("season.html",   "gridline_2026.html",    "Season",     "2026 Season Board",   "Every game graded or projected, book odds, props, playoff odds"),
    ("model.html",    "gridline_model.html",   "Calculator", "Matchup Calculator",  "Set any matchup, line, weather, QBs and injuries"),
    ("bet-log.html",  "gridline_bet_log.html", "Bet log",    "Bet Log",             "Red to green risk meter and a ledger saved in your browser"),
]
ABOUT = [
    ("Results and method", f"{GH}#results-at-a-glance"),
    ("What moves the numbers", f"{GH}#what-moves-the-numbers"),
    ("What did not work", f"{GH}#what-did-not-work"),
    ("Player props model", f"{GH}#player-props"),
    ("Run it yourself", f"{GH}#run-it-yourself"),
    ("Source code", GH),
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


def nav(current):
    here = ' aria-current="page"'
    links = "".join(f'<a href="{f}"{here if f == current else ""}>{label}</a>'
                    for f, _, label, _, _ in PAGES[1:])
    links += f'<a href="{GH}">GitHub</a>'
    return (f'<nav class="gl-nav" aria-label="Gridline pages"><div class="gl-in">'
            f'<a class="gl-brand" href="./"{here if current == "index.html" else ""}>Gridline</a>'
            f'<div class="gl-links">{links}</div></div></nav>')


def sitemap_footer(current):
    def item(f, name, desc):
        if f == current:
            return f'<li><span class="gl-here">{name} (you are here)</span><span class="gl-d">{desc}</span></li>'
        return f'<li><a href="{"./" if f == "index.html" else f}">{name}</a><span class="gl-d">{desc}</span></li>'
    tools = "".join(item(f, name, desc) for f, _, _, name, desc in PAGES)
    about = "".join(f'<li><a href="{u}">{name}</a></li>' for name, u in ABOUT)
    return (f'<footer class="gl-map"><div class="gl-in">'
            f'<nav aria-label="Site map"><h2>Site map</h2><ul>{tools}</ul></nav>'
            f'<nav aria-label="About Gridline"><h2>About</h2><ul>{about}</ul></nav>'
            f'<div class="gl-foot"><span>Built by Steven Sutcliffe</span>'
            f'<a href="https://linkedin.com/in/steven-sutcliffe">LinkedIn</a><a href="https://github.com/stevopotpie">GitHub</a>'
            f'<span>Research, not betting advice. Sports betting is not legal in every state.</span></div>'
            f'</div></footer>')


def meta(f):
    """Link preview (LinkedIn, Slack, texts), search snippet and browser tab icon for one page."""
    title = {"index.html": "Gridline: an NFL model tested against the betting market",
             "season.html": "Gridline 2026 Season Board", "model.html": "Gridline Matchup Calculator",
             "bet-log.html": "Gridline Bet Log"}[f]
    desc = next(d for p, _, _, _, d in PAGES if p == f)
    if f == "index.html":
        desc = "NFL spreads, totals and player props from public play by play data, tested honestly against the betting market. Built by Steven Sutcliffe."
    url = BASE + ("" if f == "index.html" else f)
    return (f'<meta name="description" content="{desc}">\n' if f != "index.html" else "") + (
        f'<meta property="og:type" content="website">\n<meta property="og:site_name" content="Gridline">\n'
        f'<meta property="og:title" content="{title}">\n<meta property="og:description" content="{desc}">\n'
        f'<meta property="og:url" content="{url}">\n<meta property="og:image" content="{BASE}og.png?v=2">\n'
        f'<meta property="og:image:width" content="2400">\n<meta property="og:image:height" content="1254">\n'
        f'<meta property="og:image:alt" content="Gridline: spreads, totals and player props, tested honestly against the betting market">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        f'<link rel="canonical" href="{url}">\n<link rel="icon" type="image/png" href="icon.png">\n<link rel="apple-touch-icon" href="icon.png">\n')


HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')

os.makedirs(OUT, exist_ok=True)
for f, src, *_ in PAGES:
    if src is None:
        html = open(os.path.join(REPO, "site", "index.html"), encoding="utf-8").read()
        html = html.replace("</head>", meta(f) + STYLE + "\n</head>", 1)
        html = html.replace("<!-- gridline:nav -->", nav(f), 1).replace("<!-- gridline:sitemap -->", sitemap_footer(f), 1)
    else:
        frag = open(os.path.join(REPO, src), encoding="utf-8").read()
        i, j = frag.find('<div class="wrap">'), frag.find("<script")
        if i < 0 or j < i:
            sys.exit(f"assemble: could not find the page layout in {src}")
        # nav goes right before the page content; the site map after it and before the page's script
        html = HEAD + meta(f) + STYLE + "\n" + frag[:i] + nav(f) + "\n" + frag[i:j] + sitemap_footer(f) + "\n" + frag[j:]
    open(os.path.join(OUT, f), "w", encoding="utf-8").write(html)

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
urls = "".join(f"  <url><loc>{BASE}{'' if f == 'index.html' else f}</loc><lastmod>{today}</lastmod></url>\n" for f, *_ in PAGES)
open(os.path.join(OUT, "sitemap.xml"), "w").write(
    f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}</urlset>\n')
open(os.path.join(OUT, "robots.txt"), "w").write(f"User-agent: *\nAllow: /\nSitemap: {BASE}sitemap.xml\n")
open(os.path.join(OUT, ".nojekyll"), "w").close()
for asset in ("og.png", "icon.png"):
    shutil.copy(os.path.join(REPO, "site", asset), os.path.join(OUT, asset))
print("site assembled:", ", ".join(f for f, *_ in PAGES), "+ sitemap.xml, robots.txt")
