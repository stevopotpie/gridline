# Gridline: an NFL spread and totals model

Gridline is a model I built to project NFL point spreads, over/unders and final scores, then test honestly whether any of it beats the betting market. It runs every week of the 2026 season: it pulls public play-by-play data, refits, grades the previous weekend, and publishes two pages, a matchup calculator and a full season board.

The goal was never to find a magic system. It was to find out which factors actually move outcomes, measure them on games the model had not seen, and keep only what earned its place.

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

## Side studies

- **Comebacks.** Of 1,239 second halves where a team trailed by 17 or more since 2015, 2.6% won and 19.3% got back within one score. Live win probability models already price outright comebacks well. Teams already out of contention went 0 for 162.
- **Do teams settle for the loss?** Coaches show a mild, repeatable habit of playing slower when down three scores, but it did not predict outcomes once the situation and team quality were known.
- **Blowout repeat offenders.** Teams that keep losing by 17+ do get blown out more often (19% vs 11%), but the lines already account for it.
- **Volatility.** Early season swings in a team's scores and stats did not carry into the rest of the season (correlation 0.04).

## The weekly pipeline

`run_all.sh` rebuilds everything from scratch in about two minutes:

| Step | Script |
|---|---|
| Team form, weather, injuries, QB changes | `features.py` |
| Special teams and field position | `special_teams.py` |
| Efficiency, pace, QB ratings | `candidates.py` |
| Market ratings and previous game workload | `improvements.py` |
| Fit, backtest, coefficients | `final_model.py` |
| Season projections and simulation | `season.py` |
| Comeback model | `comebacks.py`, `comeback_model.py` |
| Blowout flags and volatility | `blowouts.py`, `volatility.py` |
| Publish the calculator and season board | `build_pages.py` |

Scheduled runs grade each weekend on Tuesday, pull in lines and final injury reports on Friday, and add kickoff weather forecasts on Saturday and Sunday mornings. The season board simulates the rest of the season 20,000 times for projected records and playoff odds.

## Run it yourself

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
bash run_all.sh "$(pwd)"
```

The first run downloads about 250 MB of nflverse data into `data/` and takes a few minutes. It writes the projections, backtest results and coefficients as CSV and JSON files, and builds `gridline_model.html` (the matchup calculator) and `gridline_2026.html` (the season board), which open in any browser. To flag teams on the pages, copy `settings.example.json` to `settings.json` and list team abbreviations under `avoid` or `trust`.

### Repository layout

| Path | What it holds |
|---|---|
| Top level `*.py` and `run_all.sh` | The weekly pipeline, in the order listed above |
| `templates/` | The calculator, season board and bet log pages |
| `research/` | One-off studies: factor ranking, the improvement tests, the "do teams quit" study and the trust/avoid test |

## Tools

Python (pandas, NumPy, scikit-learn, PyArrow), with HTML and JavaScript for the published pages.

## What I would do next

- Track closing line value on every bet to measure skill faster than win and loss records can
- Use the real distribution of final margins, which clusters on 3 and 7, for cover probabilities
- Add historical rain and snow data to sharpen the weather adjustment

Built by Steven Sutcliffe. [GitHub](https://github.com/stevopotpie) · [LinkedIn](https://linkedin.com/in/steven-sutcliffe)
