"""Gridline players, step 2: pregame features for every player game (no leakage).

Every feature uses only games played before the one being predicted.
Imported by player_experiments.py and player_props.py.
"""
import numpy as np
import pandas as pd

STATS = ["pass_att", "completions", "pass_yds", "pass_td", "ints", "carries", "rush_yds", "rush_td",
         "targets", "receptions", "rec_yds", "rec_td", "tds", "rz_carries", "rz_targets", "air_yds"]
HL = 4          # player half life, in games
TEAM_HL = 6     # team and defense half life, in games


def ewm_prior(df, by, cols, hl):
    """Exponentially weighted mean of past rows within each group (shifted so the current game is excluded)."""
    out = df.groupby(by, sort=False)[cols].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1).mean())
    return out


def build(player_games: pd.DataFrame, team_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    pg = player_games.copy()
    tg = team_games.copy()
    pg["game_date"] = pd.to_datetime(pg.game_date)
    tg["game_date"] = pd.to_datetime(tg.game_date)
    pg = pg.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Team shares for each game
    pg = pg.merge(tg[["game_id", "team", "team_dropbacks", "team_carries", "team_targets", "implied_pts",
                      "spread_line", "total_line", "is_home", "points"]], on=["game_id", "team"], how="left")
    pg["tgt_share"] = pg.targets / pg.team_targets.replace(0, np.nan)
    pg["car_share"] = pg.carries / pg.team_carries.replace(0, np.nan)
    pg["pass_share"] = pg.pass_att / pg.team_dropbacks.replace(0, np.nan)
    shares = ["tgt_share", "car_share", "pass_share"]
    # Future games (no box score yet) keep NaN so they never feed later averages
    pg["played"] = pg.pass_yds.notna()
    pg.loc[pg.played, shares] = pg.loc[pg.played, shares].fillna(0)

    # Player history
    hist = STATS + shares
    e = ewm_prior(pg, "player_id", hist, HL)
    pg[[f"e_{c}" for c in hist]] = e.values
    last3 = pg.groupby("player_id", sort=False)[["pass_att", "carries", "targets"]].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    pg[["l3_pass_att", "l3_carries", "l3_targets"]] = last3.values
    pg["n_prior"] = pg.groupby("player_id").cumcount()
    pg["same_season_prior"] = pg.groupby(["player_id", "season"]).cumcount()
    # Season to date average, falling back to the player's EWMA early in the season
    for c in ["pass_yds", "rush_yds", "rec_yds", "receptions", "tds", "pass_td", "completions"]:
        std = pg.groupby(["player_id", "season"], sort=False)[c].transform(lambda s: s.shift(1).expanding().mean())
        pg[f"std_{c}"] = std.fillna(pg[f"e_{c}"])
    # Efficiency, shrunk toward league norms with pseudo counts
    pg["e_ypa"] = (pg.e_pass_yds + 7.0 * 3) / (pg.e_pass_att + 3)
    pg["e_ypc"] = (pg.e_rush_yds + 4.2 * 4) / (pg.e_carries + 4)
    pg["e_ypt"] = (pg.e_rec_yds + 7.5 * 3) / (pg.e_targets + 3)
    pg["e_catch"] = (pg.e_receptions + 0.64 * 3) / (pg.e_targets + 3)

    # Team volume and scoring history
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    tcols = ["team_dropbacks", "team_carries", "team_targets", "points", "implied_pts"]
    te = ewm_prior(tg, "team", tcols, TEAM_HL)
    tg[[f"t_{c}" for c in tcols]] = te.values
    # Defense: what each opponent allows
    tg["allow_pass_yds"] = tg.team_pass_yds
    tg["allow_rush_yds"] = tg.team_rush_yds
    tg["allow_pts"] = tg.points
    tg = tg.sort_values(["opp", "game_date"])
    de = ewm_prior(tg, "opp", ["allow_pass_yds", "allow_rush_yds", "allow_pts", "team_dropbacks", "team_carries"], TEAM_HL)
    tg[["d_pass_yds", "d_rush_yds", "d_pts", "d_dropbacks", "d_carries"]] = de.values
    # Position specific receiving yards allowed (WR, TE, RB)
    pos = pg.assign(grp=pg.position.where(pg.position.isin(["WR", "TE", "RB"]), "WR"))
    pa = pos[pos.played].pivot_table(index=["game_id", "opp", "game_date"], columns="grp", values="rec_yds", aggfunc="sum", fill_value=0).reset_index()
    fut = pg.loc[~pg.played, ["game_id", "opp", "game_date"]].drop_duplicates()
    pa = pd.concat([pa, fut], ignore_index=True).sort_values(["opp", "game_date"])
    pe = ewm_prior(pa, "opp", ["RB", "TE", "WR"], TEAM_HL)
    pa[["d_rec_RB", "d_rec_TE", "d_rec_WR"]] = pe.values
    pg = pg.merge(tg[["game_id", "team"] + [f"t_{c}" for c in tcols] +
                     ["d_pass_yds", "d_rush_yds", "d_pts", "d_dropbacks", "d_carries"]], on=["game_id", "team"], how="left")
    pg = pg.merge(pa[["game_id", "opp", "d_rec_RB", "d_rec_TE", "d_rec_WR"]], on=["game_id", "opp"], how="left")
    pg["d_rec_pos"] = np.select([pg.position == "RB", pg.position == "TE"], [pg.d_rec_RB, pg.d_rec_TE], pg.d_rec_WR)

    # Game script from the market: team spread (positive = favored) and implied points vs the team's norm
    pg["team_spread"] = np.where(pg.is_home == 1, pg.spread_line, -pg.spread_line)
    pg["imp_ratio"] = pg.implied_pts / pg.t_implied_pts
    pg["season_week"] = pg.week
    pg = add_weather(pg, games)
    return pg


def add_weather(pg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Kickoff conditions from the schedule: dome, temperature, wind. Domes and closed roofs get neutral values."""
    w = games[["game_id", "roof", "temp", "wind"]].copy()
    w["dome"] = w.roof.isin(["dome", "closed"]).astype(int)
    w["temp"] = np.where(w.dome == 1, 70, w.temp.fillna(60))
    w["wind"] = np.where(w.dome == 1, 0, w.wind.fillna(6))
    w["cold"] = (w.temp < 40).astype(int)
    w["windy"] = (w.wind >= 15).astype(int)
    w["wind_over10"] = np.clip(w.wind - 10, 0, None)
    w["weather"] = np.select([w.dome == 1, w.windy == 1, w.cold == 1], ["Dome", "Wind 15+", "Cold <40F"], "Outdoor")
    return pg.drop(columns=[c for c in ["roof", "temp", "wind", "dome", "cold", "windy", "wind_over10", "weather"] if c in pg],
                   errors="ignore").merge(w.drop(columns="roof"), on="game_id", how="left")


def eligible(df: pd.DataFrame, market: str) -> pd.Series:
    """Players a sportsbook would post a line for, judged only on pregame usage."""
    if market in ("pass_yds", "pass_td", "completions", "pass_300"):
        listed = df["listed_qb"].fillna(0).astype(bool) if "listed_qb" in df else False
        return ((df.l3_pass_att >= 20) & (df.position == "QB")) | listed
    if market in ("rush_yds", "rush_100"):
        return df.l3_carries >= 8
    if market in ("rec_yds", "receptions", "rec_100"):
        return (df.l3_targets >= 4) & df.position.isin(["WR", "TE", "RB"])
    if market in ("tds", "td_1", "td_2"):
        return ((df.l3_carries >= 5) | (df.l3_targets >= 3)) & df.position.isin(["WR", "TE", "RB"])
    raise ValueError(market)
