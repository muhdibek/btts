"""
sample_data.py
--------------
Provides realistic sample football match data for the BTTS Dashboard.
In production, replace load_matches() with a real API call to
API-Football or SportMonks.

Data schema mirrors what you'd receive from API-Football v3:
  GET /fixtures?league=39&season=2024&next=10
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- Seed for reproducibility during development ---
random.seed(42)
np.random.seed(42)


def _generate_team_stats(team_name: str, attack: float, defense: float, btts_rate: float) -> dict:
    """
    Build a team stats profile used by the prediction model.
    
    Args:
        team_name:   Club name
        attack:      Goals scored per game (scoring strength)
        defense:     Goals conceded per game (conceding strength)
        btts_rate:   Historical BTTS rate (0.0–1.0)
    """
    return {
        "team": team_name,
        "avg_goals_scored": round(attack + np.random.normal(0, 0.1), 2),
        "avg_goals_conceded": round(defense + np.random.normal(0, 0.1), 2),
        "btts_rate_last20": round(np.clip(btts_rate + np.random.normal(0, 0.05), 0.2, 0.95), 2),
        "clean_sheets_rate": round(np.clip(1 - defense / 3.0, 0.05, 0.75), 2),
        "failed_to_score_rate": round(np.clip(1 - attack / 3.0, 0.05, 0.60), 2),
    }


# --- Static team profiles ---------------------------------------------------
# (team, avg goals scored, avg goals conceded, historical BTTS rate)
# A realistic single-day card spans several leagues, so the pool covers the
# five big European leagues plus the Eredivisie — enough fixtures to build
# 4–6 leg accumulators from one day's matches.
TEAM_PROFILES = [
    # --- Premier League ---
    ("Manchester City",   2.4, 0.9, 0.72),
    ("Arsenal",           2.1, 1.0, 0.66),
    ("Liverpool",         2.3, 1.2, 0.74),
    ("Chelsea",           1.7, 1.3, 0.63),
    ("Tottenham",         1.9, 1.6, 0.68),
    ("Manchester Utd",    1.5, 1.5, 0.60),
    ("Newcastle",         1.8, 1.1, 0.64),
    ("Aston Villa",       1.6, 1.3, 0.61),
    ("Brighton",          1.7, 1.4, 0.63),
    ("West Ham",          1.4, 1.7, 0.57),
    ("Fulham",            1.3, 1.5, 0.55),
    ("Brentford",         1.6, 1.8, 0.65),
    ("Crystal Palace",    1.1, 1.4, 0.51),
    ("Everton",           1.0, 1.6, 0.48),
    ("Wolves",            1.2, 1.5, 0.53),
    ("Nottm Forest",      1.1, 1.3, 0.50),
    ("Bournemouth",       1.4, 1.7, 0.58),
    ("Burnley",           0.9, 2.0, 0.46),
    ("Sheffield Utd",     0.8, 2.2, 0.41),
    ("Luton",             0.9, 2.1, 0.44),

    # --- La Liga ---
    ("Real Madrid",       2.2, 0.9, 0.64),
    ("Barcelona",         2.1, 1.1, 0.67),
    ("Atletico Madrid",   1.7, 1.0, 0.55),
    ("Real Sociedad",     1.5, 1.3, 0.59),
    ("Villarreal",        1.7, 1.6, 0.69),
    ("Valencia",          1.3, 1.4, 0.56),
    ("Sevilla",           1.4, 1.5, 0.61),
    ("Real Betis",        1.5, 1.4, 0.63),
    ("Celta Vigo",        1.4, 1.7, 0.66),
    ("Getafe",            1.0, 1.3, 0.47),

    # --- Serie A ---
    ("Inter",             2.2, 0.9, 0.61),
    ("Milan",             1.8, 1.3, 0.65),
    ("Juventus",          1.6, 1.0, 0.52),
    ("Napoli",            1.8, 1.2, 0.63),
    ("Roma",              1.7, 1.3, 0.64),
    ("Lazio",             1.5, 1.2, 0.57),
    ("Atalanta",          2.0, 1.5, 0.71),
    ("Fiorentina",        1.6, 1.4, 0.63),
    ("Bologna",           1.4, 1.2, 0.53),
    ("Torino",            1.1, 1.2, 0.48),

    # --- Bundesliga ---
    ("Bayern Munich",     2.6, 1.2, 0.72),
    ("Bayer Leverkusen",  2.3, 1.1, 0.68),
    ("Stuttgart",         2.0, 1.4, 0.70),
    ("RB Leipzig",        2.0, 1.3, 0.69),
    ("Dortmund",          1.9, 1.5, 0.73),
    ("Frankfurt",         1.7, 1.6, 0.67),
    ("Freiburg",          1.4, 1.6, 0.64),
    ("Werder Bremen",     1.5, 1.7, 0.66),
    ("Hoffenheim",        1.6, 1.9, 0.70),
    ("Union Berlin",      1.1, 1.5, 0.52),

    # --- Ligue 1 ---
    ("PSG",               2.4, 1.0, 0.66),
    ("Monaco",            2.0, 1.5, 0.71),
    ("Marseille",         1.7, 1.3, 0.63),
    ("Lille",             1.5, 1.1, 0.55),
    ("Lyon",              1.5, 1.5, 0.62),
    ("Nice",              1.2, 0.9, 0.44),
    ("Lens",              1.4, 1.3, 0.58),
    ("Rennes",            1.6, 1.4, 0.63),

    # --- Eredivisie ---
    ("PSV",               2.7, 1.0, 0.68),
    ("Feyenoord",         2.3, 1.2, 0.72),
    ("Ajax",              1.9, 1.4, 0.70),
    ("AZ Alkmaar",        1.8, 1.4, 0.67),
    ("Twente",            1.6, 1.3, 0.63),
    ("Utrecht",           1.5, 1.6, 0.65),
]


def load_matches() -> pd.DataFrame:
    """
    Returns a DataFrame of upcoming fixtures with all features
    needed for BTTS probability estimation.

    Columns:
      match_id          – unique identifier
      home_team         – home club name
      away_team         – away club name
      kickoff           – match datetime string
      kickoff_date      – match date (YYYY-MM-DD), used by the "today only" filter
      league            – competition name
      home_btts_rate    – home team historical BTTS%
      away_btts_rate    – away team historical BTTS%
      home_avg_scored   – home avg goals scored/game
      away_avg_scored   – away avg goals scored/game
      home_avg_conceded – home avg goals conceded/game
      away_avg_conceded – away avg goals conceded/game
      btts_odds         – bookmaker BTTS Yes odds (decimal)
      home_win_odds     – home win odds
      draw_odds         – draw odds
      away_win_odds     – away win odds
    """
    profiles = {name: _generate_team_stats(name, atk, dfs, bts)
                for name, atk, dfs, bts in TEAM_PROFILES}

    # One day's card — every club appears once, so multi-leg slips never
    # double up on the same team.
    fixtures_raw = [
        # --- Premier League ---
        ("Manchester City",  "Arsenal",          "Premier League", 1.60, 4.50, 5.00),
        ("Liverpool",        "Chelsea",          "Premier League", 1.75, 3.80, 4.50),
        ("Tottenham",        "Manchester Utd",   "Premier League", 2.10, 3.40, 3.60),
        ("Newcastle",        "Aston Villa",      "Premier League", 2.00, 3.50, 3.80),
        ("Brighton",         "Brentford",        "Premier League", 2.20, 3.20, 3.40),
        ("Wolves",           "West Ham",         "Premier League", 2.40, 3.10, 3.00),
        ("Crystal Palace",   "Fulham",           "Premier League", 2.50, 3.20, 2.90),
        ("Burnley",          "Sheffield Utd",    "Premier League", 2.20, 3.30, 3.30),
        ("Everton",          "Bournemouth",      "Premier League", 2.30, 3.25, 3.10),
        ("Nottm Forest",     "Luton",            "Premier League", 1.95, 3.40, 3.90),

        # --- La Liga ---
        ("Real Madrid",      "Valencia",         "La Liga",        1.45, 4.60, 6.50),
        ("Barcelona",        "Celta Vigo",       "La Liga",        1.40, 4.80, 7.00),
        ("Villarreal",       "Real Betis",       "La Liga",        2.30, 3.40, 2.95),
        ("Sevilla",          "Real Sociedad",    "La Liga",        2.55, 3.20, 2.80),
        ("Atletico Madrid",  "Getafe",           "La Liga",        1.55, 3.90, 6.00),

        # --- Serie A ---
        ("Atalanta",         "Fiorentina",       "Serie A",        1.90, 3.60, 3.90),
        ("Milan",            "Roma",             "Serie A",        2.15, 3.40, 3.30),
        ("Napoli",           "Bologna",          "Serie A",        1.80, 3.50, 4.40),
        ("Inter",            "Torino",           "Serie A",        1.35, 5.00, 8.00),
        ("Lazio",            "Juventus",         "Serie A",        2.90, 3.10, 2.50),

        # --- Bundesliga ---
        ("Dortmund",         "Hoffenheim",       "Bundesliga",     1.55, 4.30, 5.20),
        ("Bayer Leverkusen", "Frankfurt",        "Bundesliga",     1.50, 4.40, 5.50),
        ("Stuttgart",        "Werder Bremen",    "Bundesliga",     1.70, 4.00, 4.30),
        ("Bayern Munich",    "Union Berlin",     "Bundesliga",     1.25, 6.00, 10.00),
        ("RB Leipzig",       "Freiburg",         "Bundesliga",     1.65, 4.20, 4.60),

        # --- Ligue 1 ---
        ("PSG",              "Lyon",             "Ligue 1",        1.40, 4.90, 7.00),
        ("Monaco",           "Rennes",           "Ligue 1",        1.85, 3.70, 3.80),
        ("Marseille",        "Lens",             "Ligue 1",        2.05, 3.40, 3.60),
        ("Lille",            "Nice",             "Ligue 1",        2.25, 3.10, 3.40),

        # --- Eredivisie ---
        ("PSV",              "Utrecht",          "Eredivisie",     1.35, 5.20, 8.00),
        ("Feyenoord",        "Twente",           "Eredivisie",     1.50, 4.50, 5.50),
        ("Ajax",             "AZ Alkmaar",       "Eredivisie",     1.80, 3.90, 3.90),
    ]

    rows = []
    # All fixtures are scheduled on TODAY's date so the dashboard always has a
    # full same-day card to build slips from (kickoffs spread 12:00 → ~21:45).
    base_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    for i, (home, away, league, h_odds, d_odds, a_odds) in enumerate(fixtures_raw):
        ht = profiles[home]
        at = profiles[away]
        kickoff = base_time + timedelta(minutes=(i % 14) * 45 + (i // 14) * 20)

        # BTTS odds: bookmaker price inverse of implied probability + margin
        # avg_btts ≈ 0.65–0.80 for top matches → implied odds 1.25–1.54
        # Real bookmaker BTTS odds typically 1.60–2.10 after margin
        avg_btts = (ht["btts_rate_last20"] + at["btts_rate_last20"]) / 2
        fair_odds = 1 / avg_btts
        # Add 8–14% bookmaker margin and spread
        margin = np.random.uniform(1.08, 1.14)
        btts_odds = round(np.clip(fair_odds * margin, 1.60, 2.20), 2)

        rows.append({
            "match_id":          f"M{1000 + i}",
            "home_team":         home,
            "away_team":         away,
            "kickoff":           kickoff.strftime("%Y-%m-%d %H:%M"),
            "kickoff_date":      kickoff.strftime("%Y-%m-%d"),
            "league":            league,
            "home_btts_rate":    ht["btts_rate_last20"],
            "away_btts_rate":    at["btts_rate_last20"],
            "home_avg_scored":   ht["avg_goals_scored"],
            "away_avg_scored":   at["avg_goals_scored"],
            "home_avg_conceded": ht["avg_goals_conceded"],
            "away_avg_conceded": at["avg_goals_conceded"],
            "btts_odds":         btts_odds,
            "home_win_odds":     h_odds,
            "draw_odds":         d_odds,
            "away_win_odds":     a_odds,
        })

    return pd.DataFrame(rows)
