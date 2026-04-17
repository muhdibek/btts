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


# --- Static team profiles (used to generate fixture rows) ---
TEAM_PROFILES = [
    ("Manchester City",   2.4, 0.9, 0.78),
    ("Arsenal",           2.1, 1.0, 0.72),
    ("Liverpool",         2.3, 1.2, 0.80),
    ("Chelsea",           1.7, 1.3, 0.68),
    ("Tottenham",         1.9, 1.6, 0.74),
    ("Manchester Utd",    1.5, 1.5, 0.65),
    ("Newcastle",         1.8, 1.1, 0.70),
    ("Aston Villa",       1.6, 1.3, 0.66),
    ("Brighton",          1.7, 1.4, 0.69),
    ("West Ham",          1.4, 1.7, 0.62),
    ("Fulham",            1.3, 1.5, 0.60),
    ("Brentford",         1.6, 1.8, 0.71),
    ("Crystal Palace",    1.1, 1.4, 0.55),
    ("Everton",           1.0, 1.6, 0.52),
    ("Wolves",            1.2, 1.5, 0.58),
    ("Nottm Forest",      1.1, 1.3, 0.54),
    ("Bournemouth",       1.4, 1.7, 0.63),
    ("Burnley",           0.9, 2.0, 0.50),
    ("Sheffield Utd",     0.8, 2.2, 0.45),
    ("Luton",             0.9, 2.1, 0.48),
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

    # Build 12 realistic fixtures
    fixtures_raw = [
        ("Manchester City",  "Arsenal",        "Premier League", 1.60, 4.50, 5.00),
        ("Liverpool",        "Chelsea",         "Premier League", 1.75, 3.80, 4.50),
        ("Tottenham",        "Manchester Utd",  "Premier League", 2.10, 3.40, 3.60),
        ("Newcastle",        "Aston Villa",     "Premier League", 2.00, 3.50, 3.80),
        ("Brighton",         "Brentford",       "Premier League", 2.20, 3.20, 3.40),
        ("Wolves",           "West Ham",        "Premier League", 2.40, 3.10, 3.00),
        ("Crystal Palace",   "Fulham",          "Premier League", 2.50, 3.20, 2.90),
        ("Burnley",          "Sheffield Utd",   "Premier League", 2.20, 3.30, 3.30),
        ("Everton",          "Bournemouth",     "Premier League", 2.30, 3.25, 3.10),
        ("Nottm Forest",     "Luton",           "Premier League", 1.95, 3.40, 3.90),
        ("Arsenal",          "Liverpool",       "Premier League", 2.40, 3.50, 2.90),
        ("Chelsea",          "Tottenham",       "Premier League", 1.90, 3.60, 4.10),
    ]

    rows = []
    base_time = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)

    for i, (home, away, league, h_odds, d_odds, a_odds) in enumerate(fixtures_raw):
        ht = profiles[home]
        at = profiles[away]
        kickoff = base_time + timedelta(hours=i * 2)

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
