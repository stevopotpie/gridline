# Gridline: an NFL spread and totals model

Gridline is a model I built to project NFL point spreads, over/unders, final scores and player props, then test honestly whether any of it beats the betting market. It runs every week of the 2026 season: it pulls public play-by-play data, refits, grades the previous weekend, and publishes two pages, a matchup calculator and a full season board with player projections.

The goal was never to find a magic system. It was to find out which factors actually move outcomes, measure them on games the model had not seen, and keep only what earned its place.

## Try it

No install needed. The pages rebuild automatically from the latest data and odds:

- **[Gridline home](https://stevopotpie.github.io/gridline/)**
- **[2026 season board](https://stevopotpie.github.io/gridline/season.html)**: every game graded or projected, the best number at the books, player props and playoff odds
- **[Matchup calculator](https://stevopotpie.github.io/gridline/model.html)**: set any matchup, line, weather, quarterbacks and injuries
- **[Bet log](https://stevopotpie.github.io/gridline/bet-log.html)**: enter a bet and a meter runs from red to green with its chance to win, and whether Gridline's chance beats the price. Bets are saved in your own browser only, with export and import to back them up.

## Results at a glance

All figures come from walk-forward testing: each season from 2018 to 2025 was predicted by a model trained only on earlier seasons. Breakeven at standard -110 odds is 52.4%.

| Model | Test | Result |
|---|---|---|
| Spread, stats only | Games where the model and Vegas differ by 3+ points | 349-288 (54.8%) |
| Spread, Vegas line plus adjustments | Differ by 1+ point | 550-490 (52.9%) |
| Total, Vegas total plus adjustments | Differ by 1.5+ points | 328-275 (54.4%), winning in 6 of 8 seasons |
| Spread accuracy | Average miss on the final margin | 10.04 points (Vegas: 9.81) |

Two caveats matter. First, I chose the final factors by testing on these same seasons, so the records above are a best case. Second, Vegas is still more accurate on average; the model's value, if any, shows up only in the games where the two disagree. The 2026 season is the real out of sample test, and the season board grades every game as it happens.

## Data

Everything comes from [nflverse](https://github.com/nflverse/nflverse-data), 2015 to 2026:

- Schedules with closing spreads and totals, kickoff temperature, wind and roof type
- Play-by-play (about 540,000 plays) for efficiency, special teams, pace and field position
- Weekly injury reports and depth charts, used to identify regular starters ruled Out or Doubtful

## Method

1. **Pregame features only.** Every rating is an exponentially weighted average of past games, so nothing from the game being predicted leaks into its own projection. Ratings carry over between seasons with a pull toward the league average.
2. **Ridge regression** for the final margin and the game total, plus a second version of each that starts from the Vegas number and learns what the market misses.
3. **Walk-forward backtesting** by season, judged on average miss and on records at different edge sizes against the closing line.
4. **Feature testing one group at a time** against the current model, keeping a factor only if it improved accuracy on held out seasons.

## What moves the numbers

| Factor | Effect |
|---|---|
| Starting QB rating (EPA per dropback, last 12 starts) | The single biggest factor: about 1.4 to 1.8 points of spread and 2 points of total per standard deviation |
| Market power ratings from earlier weeks' closing lines | About 2.4 points per standard deviation; the largest gain from the latest round of testing |
| Scoring form, success rate, passing efficiency | Core team strength |
| Previous game workload | About 0.65 points worse for every 10 extra offensive or defensive snaps the week before |
| Return units (kick returns 70%, punt returns 30%) | About 0.5 points per standard deviation |
| Home field | 2.37 points since 2021, estimated separately from the pre-2021 era |
| Wind above 15 mph | About 0.4 points off the total per standard deviation; unders hit 55% at 10 to 19 mph |
| Domes, pace, explosive plays | Mainly totals |

## What did not work

Some of the most useful findings were negative. Each of these was tested on held out seasons and left out because it made projections worse or no better:

- Turnover margin, red zone touchdown rate, rushing efficiency and sack rates. They swing too much week to week to predict the next game.
- Punter pinning inside the 10, net punting and kicker accuracy. Real skills, but too small to move final scores.
- Garbage time filtering, opponent adjusted efficiency and referee crews.
- Bye weeks, short weeks, and coming off a division game, a shootout or overtime. Fatigue shows up better in snap counts.
- Team specific home field. A team's home edge one year has almost no relationship to the next (correlation 0.02).
- "Trust" and "avoid" labels. How far a team's games land from the line is not a stable trait: a team's misses in its odd numbered games had no relationship to its even numbered games in the same season (correlation -0.04).
- Penalties as a drag on passing yards. The idea: holding wipes out completions and pass interference turns catches into penalty yards, so penalty-prone matchups should cut into a quarterback's passing total. In 3,723 QB games from 2019 to 2025 it ran the other way. Quarterbacks whose offense drew the most penalty yards in a game threw for 12 yards more than the model expected, and those with the fewest threw for 14 fewer, because penalties follow passing volume: more dropbacks mean more chances for a flag. Known before kickoff, neither team's penalty habit improved the passing yards projection (average miss 64.85 to 64.82 yards at best), and a defense's pass interference rate does not carry over at all (first 8 games vs the rest of the season: correlation 0.01). In the team model, combined penalty yards stay in the totals projection but are worth only about 0.2 points per standard deviation.

## Player props

The player model projects passing, rushing and receiving yards, receptions, passing touchdowns and anytime touchdowns for every starter and regular contributor, then turns each projection into the probability of the common prop milestones: 300+ passing yards, 2+ and 3+ passing touchdowns, 100+ rushing or receiving yards, and 1+ or 2+ touchdowns. Each probability is shown with its fair (no vig) American odds.

### Finding the best algorithm

I tested six approaches for each stat, walk forward on 2019 to 2025 (each season predicted by models trained only on earlier seasons), on players a sportsbook would post a line for:

| Average miss | Season average | Recent form (EWMA) | Usage x efficiency | Gradient boosting | Ridge | Ridge + weather |
|---|---|---|---|---|---|---|
| Passing yards | 68.8 | 66.5 | 66.2 | 67.1 | 65.2 | **64.9** |
| Rushing yards | 28.3 | 27.5 | 26.9 | 27.9 | **26.8** | **26.8** |
| Receiving yards | 26.8 | 25.8 | 26.7 | 25.3 | **25.3** | **25.3** |
| Receptions | 1.89 | 1.81 | 1.79 | 1.80 | **1.79** | **1.79** |
| Passing TDs | 0.96 | 0.93 | 0.95 | 0.93 | **0.91** | **0.91** |

Ridge regression won every market. It combines the player's recent production and usage (target share, carry share, red zone touches), shrunk efficiency, the game script implied by the betting line (team implied points and spread), the opponent's yards allowed to that position, home field and kickoff weather. Gradient boosting, the fanciest option, finished behind it: with one game per player per week, the simpler model generalizes better.

For milestones, a normal distribution whose spread widens with the projection beat a direct classifier for yards, and a Poisson distribution won for touchdowns. A final calibration layer fit on the walk forward predictions keeps the percentages honest: players given a 30 to 40% chance to score scored 34.9% of the time.

### Weather and player stats

Measured against what the model expects without weather, 2016 to 2026:

| Condition | Passing yards (QB) | Receiving yards (per player) | Rushing yards |
|---|---|---|---|
| Wind 15+ mph | -20.5 | -3.5 | +1.1 |
| Dome | +5.0 | +1.6 | -0.2 |
| Cold below 40°F | -0.3 | -0.5 | +1.7 |

Wind is the one that matters, and it lands on the passing game. The season board shows each player's own per game averages in domes, outdoors, cold and wind since 2021.

One limit: the free data has no historical prop lines, so the backtest grades projections against actual results, not against what books posted. The way to judge it against the market is to log the posted line and price on each bet and track closing line value.

## Side studies

- **Comebacks.** Of 1,239 second halves where a team trailed by 17 or more since 2015, 2.6% won and 19.3% got back within one score. Live win probability models already price outright comebacks well. Teams already out of contention went 0 for 162.
- **Do teams settle for the loss?** Coaches show a mild, repeatable habit of playing slower when down three scores, but it did not predict outcomes once the situation and team quality were known.
- **Blowout repeat offenders.** Teams that keep losing by 17+ do get blown out more often (19% vs 11%), but the lines already account for it.
- **Fourth quarter scoring.** Teams trailing by 15 or more after three quarters outscore the leader by 1.6 points in the fourth quarter on average, and teams down 9 to 14 score the most (7.2 points), as leaders protect the lead. That matters for live totals and backdoor covers. A team's own fourth quarter edge carries over only modestly (first 8 games vs the rest of the season: correlation 0.22).
- **Volatility.** Early season swings in a team's scores and stats did not carry into the rest of the season (correlation 0.04).

## The weekly pipeline

`run_all.sh` rebuilds everything from scratch in about two minutes:

| Step | Script |
|---|---|
| Live sportsbook lines from the odds snapshots in `odds/` | `odds.py` |
| Team form, weather, injuries, QB changes | `features.py` |
| Special teams and field position | `special_teams.py` |
| Efficiency, pace, QB ratings | `candidates.py` |
| Market ratings and previous game workload | `improvements.py` |
| Fit, backtest, coefficients | `final_model.py` |
| Season projections and simulation | `season.py` |
| Comeback model | `comebacks.py`, `comeback_model.py` |
| Blowout flags and volatility | `blowouts.py`, `volatility.py` |
| Player box scores | `player_stats.py` |
| Player projections and prop probabilities | `player_features.py`, `player_props.py` |
| Fourth quarter scoring by margin | `fourth_quarter.py` |
| Publish the calculator and season board | `build_pages.py` |
| Bet log with a calibrated risk meter | `build_pages.py betlog`, `templates/bet_log.html` |

Scheduled runs grade each weekend on Tuesday, pull in lines and final injury reports on Friday, and add kickoff weather forecasts on Saturday and Sunday mornings. The season board simulates the rest of the season 20,000 times for projected records and playoff odds.

## Bet log and risk meter

The bet log tracks every bet, its result and its closing line value. As a bet is entered, a meter runs from red (long shot) to green (safe) showing its chance to win, and says whether Gridline's chance beats the price the odds imply. Spreads and totals start from the Vegas line and add only the share of Gridline's disagreement that has shown up in results: fit on the 2018 to 2025 backtest, a 3 point gap from the Vegas spread is worth about 52%, not the 59% a raw projection would suggest. Safe means likely, not good value. Opened from the public site or a local file, the log saves bets in that browser (localStorage) with export and import buttons; nothing is sent anywhere.

## Run it yourself

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
bash run_all.sh "$(pwd)"
```

The first run downloads about 250 MB of nflverse data into `data/` and takes a few minutes. It writes the projections, backtest results and coefficients as CSV and JSON files, and builds `gridline_model.html` (the matchup calculator), `gridline_2026.html` (the season board) and `gridline_bet_log.html` (the bet log), which open in any browser. To flag teams on the pages, copy `settings.example.json` to `settings.json` and list team abbreviations under `avoid` or `trust`.

### Repository layout

| Path | What it holds |
|---|---|
| Top level `*.py` and `run_all.sh` | The weekly pipeline, in the order listed above |
| `templates/` | The calculator, season board and bet log pages |
| `site/` | The public site's landing page, and `assemble.py`, which adds the navigation bar, site map, sitemap.xml and robots.txt |
| `odds/` | Sportsbook odds snapshots pulled several times a week from The Odds API |
| `.github/workflows/pages.yml` | Rebuilds everything and publishes the site to GitHub Pages on a schedule and whenever new odds arrive |
| `player_best_models.json` | The winning player model and features for each stat, from `research/player_experiments.py` |
| `research/` | One-off studies: factor ranking, the improvement tests, the player model bake-off, the "do teams quit" study, the trust/avoid test and the penalties vs passing yards study |

## Tools

Python (pandas, NumPy, scikit-learn, SciPy, PyArrow), with HTML and JavaScript for the published pages.

## What I would do next

- Use the real distribution of final margins, which clusters on 3 and 7, for cover probabilities
- Collect posted player prop lines each week to measure the player model against the market
- Add historical rain and snow data to sharpen the weather adjustment

Built by Steven Sutcliffe. [GitHub](https://github.com/stevopotpie) · [LinkedIn](https://linkedin.com/in/steven-sutcliffe)
