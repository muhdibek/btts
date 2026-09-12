"""
api_client.py  (Phase 2 — NOT active in Phase 1)
-------------------------------------------------
API-Football v3 integration.

To activate:
  1. Create a free account at https://www.api-football.com
  2. Copy your API key to a .env file:  API_FOOTBALL_KEY=your_key_here
  3. Replace load_matches() in data/sample_data.py with:
       from utils.api_client import fetch_upcoming_fixtures
       df = fetch_upcoming_fixtures(league_id=39, season=2024, next_n=10)

Endpoint used:
  GET /fixtures?league={league_id}&season={season}&next={next_n}
  GET /fixtures/statistics?fixture={fixture_id}&team={team_id}

Rate limit: 100 requests/day on free tier.
Cache responses locally to stay within limits.
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY      = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL     = "https://v3.football.api-sports.io"
CACHE_DIR    = Path(__file__).parent.parent / ".cache"
CACHE_TTL_S  = 3600  # cache responses for 1 hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "v3.football.api-sports.io"}


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if time.time() - data["_ts"] > CACHE_TTL_S:
        return None
    return data["payload"]


def _save_cache(key: str, payload: dict):
    path = _cache_path(key)
    path.write_text(json.dumps({"_ts": time.time(), "payload": payload}))


def _get(endpoint: str, params: dict) -> dict:
    """Make a GET request with caching."""
    cache_key = endpoint.strip("/").replace("/", "_") + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    cached = _load_cache(cache_key)
    if cached:
        return cached

    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=_headers(), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _save_cache(cache_key, data)
    return data


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_upcoming_fixtures(league_id: int = 39, season: int = 2024, next_n: int = 15) -> pd.DataFrame:
    """
    Fetch upcoming fixtures from API-Football and return a DataFrame
    in the same schema as sample_data.load_matches().
    
    Args:
        league_id:  API-Football league ID (39 = EPL, 140 = La Liga, 78 = Bundesliga)
        season:     Season year (e.g. 2024)
        next_n:     Number of next fixtures to fetch
    
    Returns:
        DataFrame matching the schema expected by score_matches()
    """
    if not API_KEY:
        raise ValueError("API_FOOTBALL_KEY not set in .env file")

    raw = _get("fixtures", {"league": league_id, "season": season, "next": next_n})

    rows = []
    for fix in raw.get("response", []):
        fixture_id = fix["fixture"]["id"]
        home       = fix["teams"]["home"]["name"]
        away       = fix["teams"]["away"]["name"]
        kickoff    = fix["fixture"]["date"][:16].replace("T", " ")
        league_nm  = fix["league"]["name"]

        # Fetch team statistics (last 20 matches)
        home_stats = _fetch_team_stats(fix["teams"]["home"]["id"], league_id, season)
        away_stats = _fetch_team_stats(fix["teams"]["away"]["id"], league_id, season)

        rows.append({
            "match_id":          str(fixture_id),
            "home_team":         home,
            "away_team":         away,
            "kickoff":           kickoff,
            "league":            league_nm,
            "home_btts_rate":    home_stats["btts_rate"],
            "away_btts_rate":    away_stats["btts_rate"],
            "home_avg_scored":   home_stats["avg_scored"],
            "away_avg_scored":   away_stats["avg_scored"],
            "home_avg_conceded": home_stats["avg_conceded"],
            "away_avg_conceded": away_stats["avg_conceded"],
            "btts_odds":         1.90,   # replace with odds API call
            "home_win_odds":     2.10,
            "draw_odds":         3.40,
            "away_win_odds":     3.20,
        })

    return pd.DataFrame(rows)


def _fetch_team_stats(team_id: int, league_id: int, season: int) -> dict:
    """
    Fetch season statistics for a team and compute BTTS-relevant features.
    """
    data = _get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    stats = data.get("response", {})

    played   = stats.get("fixtures", {}).get("played", {}).get("home", 1) + \
               stats.get("fixtures", {}).get("played", {}).get("away", 1)
    played   = max(played, 1)

    goals_f  = stats.get("goals", {}).get("for",     {}).get("total", {}).get("total", 0)
    goals_a  = stats.get("goals", {}).get("against", {}).get("total", {}).get("total", 0)

    # Approximation: BTTS occurred when both teams scored ≥ 1
    # True BTTS rate requires match-level data — use btts_percentage if available
    btts_pct = stats.get("biggest", {}).get("streak", {})  # placeholder
    # Better approach: loop through fixtures and count BTTS
    btts_rate = min(0.90, (goals_f / played) / 2.5 * 0.8)  # rough proxy

    return {
        "avg_scored":   round(goals_f / played, 2),
        "avg_conceded": round(goals_a / played, 2),
        "btts_rate":    round(btts_rate, 2),
    }


# ---------------------------------------------------------------------------
# Odds API stub
# ---------------------------------------------------------------------------

def fetch_btts_odds(fixture_id: int) -> float | None:
    """
    Fetch BTTS Yes odds from API-Football odds endpoint.
    
    Endpoint: GET /odds?fixture={fixture_id}&bookmaker=8&bet=8
      bookmaker=8  → Bet365
      bet=8        → Both Teams To Score
    
    Returns decimal odds for BTTS Yes, or None if unavailable.
    """
    data = _get("odds", {"fixture": fixture_id, "bookmaker": 8, "bet": 8})
    try:
        bets = data["response"][0]["bookmakers"][0]["bets"][0]["values"]
        for val in bets:
            if val["value"].lower() == "yes":
                return float(val["odd"])
    except (IndexError, KeyError, TypeError):
        pass
    return None
