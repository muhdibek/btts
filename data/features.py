"""
features.py
-----------
Pre-match feature engineering for the BTTS dataset.

Every feature here is built from matches that kicked off STRICTLY BEFORE the
row it describes. That is the whole job: a leaked feature produces a model
that backtests beautifully and loses money, so the builder makes a single
chronological pass and reads each team's history before appending the current
match to it.

Feature groups
  form      last-5 / last-10 goals for & against, BTTS rate, failed-to-score
            rate and clean-sheet rate, over all competitions in the dataset
  venue     the home team's recent HOME matches and the away team's recent
            AWAY matches, separately — home/away splits matter for BTTS
  elo       pre-match Elo ratings with a goal-difference multiplier and a
            home-advantage term
  h2h       the sides' previous meetings: BTTS rate and average total goals
  rest      days since each side's previous match
  market    overround-free implied probabilities from the closing 1X2 and
            over/under 2.5 prices — a strong feature and the benchmark to beat

Teams with no history yet get NaN rather than 0, so a training script can
drop or impute them deliberately. `home_matches_played` / `away_matches_played`
say how much history each row actually had.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_FORM_WINDOWS   = (5, 10)
DEFAULT_VENUE_WINDOW   = 10
DEFAULT_H2H_WINDOW     = 6

ELO_START              = 1500.0
ELO_K                  = 20.0
ELO_HOME_ADVANTAGE     = 65.0    # rating points, roughly the historical edge
ELO_SEASON_REGRESSION  = 0.25    # pull toward the mean after a long break
ELO_BREAK_DAYS         = 60      # a gap this long is treated as a new season


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

def elo_expected(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """Expected score for A (win = 1, draw = 0.5) against B."""
    return 1.0 / (1.0 + 10 ** (-((rating_a + home_advantage) - rating_b) / 400.0))


def _goal_difference_multiplier(goal_diff: int) -> float:
    """
    World Football Elo margin-of-victory weighting: a rout moves ratings more
    than a one-goal win, with diminishing returns.
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


# ---------------------------------------------------------------------------
# Market prices
# ---------------------------------------------------------------------------

def implied_probabilities(*odds: float) -> list[float]:
    """
    Convert decimal odds for one market into probabilities with the
    bookmaker's overround divided out proportionally.

    Returns NaNs if any leg of the market is missing or non-positive.
    """
    values = [float(o) if o is not None else np.nan for o in odds]
    if any((not np.isfinite(v)) or v <= 1.0 for v in values):
        return [np.nan] * len(values)

    raw   = [1.0 / v for v in values]
    total = sum(raw)
    return [r / total for r in raw]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overround-free implied probabilities from the closing prices.

    Bet365 is used where present, the market average as the fallback.
    """
    df = df.copy()

    def pick(row: pd.Series, primary: str, fallback: str) -> float:
        value = row.get(primary, np.nan)
        if value is None or not np.isfinite(pd.to_numeric(value, errors="coerce")):
            value = row.get(fallback, np.nan)
        return pd.to_numeric(value, errors="coerce")

    p_home, p_draw, p_away, p_over = [], [], [], []
    for _, row in df.iterrows():
        h, d, a = (pick(row, "b365h", "avgh"),
                   pick(row, "b365d", "avgd"),
                   pick(row, "b365a", "avga"))
        ph, pd_, pa = implied_probabilities(h, d, a)

        over, under = (pick(row, "b365over_2.5", "avgover_2.5"),
                       pick(row, "b365under_2.5", "avgunder_2.5"))
        po, _pu = implied_probabilities(over, under)

        p_home.append(ph); p_draw.append(pd_); p_away.append(pa); p_over.append(po)

    df["mkt_p_home"]    = p_home
    df["mkt_p_draw"]    = p_draw
    df["mkt_p_away"]    = p_away
    df["mkt_p_over25"]  = p_over
    # A tight match with plenty of goals expected is the classic BTTS shape.
    df["mkt_closeness"] = 1.0 - (pd.Series(p_home) - pd.Series(p_away)).abs()

    return df


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _summarise(records: Sequence[dict[str, Any]], window: int) -> dict[str, float]:
    """
    Aggregate a team's most recent `window` matches.

    `records` is that team's history oldest-first; only the tail is used.
    Returns NaNs when there is no history at all.
    """
    recent = list(records)[-window:]
    if not recent:
        return {
            "gf_avg": np.nan, "ga_avg": np.nan, "btts_rate": np.nan,
            "fts_rate": np.nan, "cs_rate": np.nan, "matches": 0,
        }

    gf = np.array([r["gf"] for r in recent], dtype=float)
    ga = np.array([r["ga"] for r in recent], dtype=float)
    return {
        "gf_avg":    float(gf.mean()),
        "ga_avg":    float(ga.mean()),
        "btts_rate": float(np.mean([(g > 0) and (c > 0) for g, c in zip(gf, ga)])),
        "fts_rate":  float(np.mean(gf == 0)),
        "cs_rate":   float(np.mean(ga == 0)),
        "matches":   len(recent),
    }


def _h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Order-independent key so a fixture's history is shared both ways."""
    return (team_a, team_b) if team_a <= team_b else (team_b, team_a)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features(
    matches:        pd.DataFrame,
    form_windows:   Sequence[int] = DEFAULT_FORM_WINDOWS,
    venue_window:   int   = DEFAULT_VENUE_WINDOW,
    h2h_window:     int   = DEFAULT_H2H_WINDOW,
    elo_k:          float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    elo_regression: float = ELO_SEASON_REGRESSION,
) -> pd.DataFrame:
    """
    Build the pre-match feature table.

    Args:
        matches:        normalised match table from football_data.load_matches()
        form_windows:   rolling windows for overall form, e.g. (5, 10)
        venue_window:   window for the home-at-home / away-at-away splits
        h2h_window:     how many previous meetings to summarise
        elo_k:          Elo K-factor
        home_advantage: Elo home advantage, in rating points
        elo_regression: fraction pulled back toward 1500 after a long break
                        (a new season); 0 disables it

    Returns:
        The input rows, in kickoff order, with the feature columns appended.

    Note on Elo: ratings are pooled across every division in the dataset, so
    they track promotion and relegation within a country correctly. Clubs from
    different countries never meet here, so their ratings are not comparable
    across leagues — use elo_diff (a within-match quantity), not raw elo.
    """
    if matches.empty:
        return matches.copy()

    df = matches.sort_values("kickoff").reset_index(drop=True)

    history:   dict[str, Deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=50))
    venue_hist: dict[tuple[str, str], Deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=50))
    h2h_hist:  dict[tuple[str, str], Deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))
    elo:       dict[str, float] = defaultdict(lambda: ELO_START)
    last_seen: dict[str, pd.Timestamp] = {}

    rows: list[dict[str, float]] = []

    for match in df.itertuples(index=False):
        home, away = match.home_team, match.away_team
        kickoff    = match.kickoff
        feat: dict[str, float] = {}

        # --- Elo: regress toward the mean after a long break (new season) ---
        for team in (home, away):
            previous = last_seen.get(team)
            if previous is not None and elo_regression > 0:
                if (kickoff - previous).days >= ELO_BREAK_DAYS:
                    elo[team] += (ELO_START - elo[team]) * elo_regression

        home_elo, away_elo = elo[home], elo[away]
        feat["home_elo"]     = home_elo
        feat["away_elo"]     = away_elo
        feat["elo_diff"]     = home_elo - away_elo
        feat["elo_exp_home"] = elo_expected(home_elo, away_elo, home_advantage)

        # --- Overall form over the last N matches ---
        for window in form_windows:
            for side, team in (("home", home), ("away", away)):
                stats = _summarise(history[team], window)
                for key, value in stats.items():
                    if key == "matches":
                        continue
                    feat[f"{side}_{key}_l{window}"] = value

        # --- Venue splits: home side at home, away side away ---
        for side, team, venue in (("home", home, "H"), ("away", away, "A")):
            stats = _summarise(venue_hist[(team, venue)], venue_window)
            for key in ("gf_avg", "ga_avg", "btts_rate"):
                feat[f"{side}_{key}_venue"] = stats[key]

        # --- Head to head ---
        meetings = list(h2h_hist[_h2h_key(home, away)])[-h2h_window:]
        if meetings:
            feat["h2h_btts_rate"]  = float(np.mean([m["btts"] for m in meetings]))
            feat["h2h_avg_goals"]  = float(np.mean([m["total_goals"] for m in meetings]))
            feat["h2h_matches"]    = len(meetings)
        else:
            feat["h2h_btts_rate"]  = np.nan
            feat["h2h_avg_goals"]  = np.nan
            feat["h2h_matches"]    = 0

        # --- Rest days and accumulated history ---
        feat["home_rest_days"] = ((kickoff - last_seen[home]).days
                                  if home in last_seen else np.nan)
        feat["away_rest_days"] = ((kickoff - last_seen[away]).days
                                  if away in last_seen else np.nan)
        feat["home_matches_played"] = len(history[home])
        feat["away_matches_played"] = len(history[away])

        rows.append(feat)

        # ---------------------------------------------------------------
        # Only now, after the features are recorded, does this match enter
        # the history it would otherwise have leaked into.
        # ---------------------------------------------------------------
        hg, ag = int(match.fthg), int(match.ftag)
        btts   = int(hg > 0 and ag > 0)

        history[home].append({"gf": hg, "ga": ag})
        history[away].append({"gf": ag, "ga": hg})
        venue_hist[(home, "H")].append({"gf": hg, "ga": ag})
        venue_hist[(away, "A")].append({"gf": ag, "ga": hg})
        h2h_hist[_h2h_key(home, away)].append({"btts": btts, "total_goals": hg + ag})
        last_seen[home] = last_seen[away] = kickoff

        # --- Elo update ---
        actual     = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        expected   = elo_expected(home_elo, away_elo, home_advantage)
        multiplier = _goal_difference_multiplier(hg - ag)
        change     = elo_k * multiplier * (actual - expected)
        elo[home]  = home_elo + change
        elo[away]  = away_elo - change

    features = pd.DataFrame(rows, index=df.index)
    out      = pd.concat([df, features], axis=1)

    # Derived combinations the model would otherwise have to learn from scratch.
    for window in form_windows:
        out[f"combined_btts_rate_l{window}"] = (
            out[f"home_btts_rate_l{window}"] + out[f"away_btts_rate_l{window}"]
        ) / 2.0
        out[f"expected_goals_proxy_l{window}"] = (
            out[f"home_gf_avg_l{window}"] + out[f"away_ga_avg_l{window}"] +
            out[f"away_gf_avg_l{window}"] + out[f"home_ga_avg_l{window}"]
        ) / 2.0

    return add_market_features(out)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    The model-ready feature columns — everything built by build_features(),
    excluding identifiers, the label, and any column derived from the result.
    """
    non_features = {
        "kickoff", "date", "div", "season", "home_team", "away_team",
        "fthg", "ftag", "ftr", "hthg", "htag", "hs", "a_s", "hst", "ast",
        "hc", "ac", "btts", "total_goals",
        "b365h", "b365d", "b365a", "avgh", "avgd", "avga",
        "b365over_2.5", "b365under_2.5", "avgover_2.5", "avgunder_2.5",
    }
    return [c for c in df.columns if c not in non_features]
