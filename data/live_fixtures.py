"""
live_fixtures.py
----------------
Real fixtures for the dashboard, from the openfootball/football.json feed.

    https://github.com/openfootball/football.json

Why this source: it is public, needs no API key, auto-updates daily, and
carries the full season for the major European leagues — both the fixtures
still to come and the results already played. That second half matters: team
statistics here are computed from actual results rather than invented.

What it does NOT carry is BOOKMAKER ODDS. Nothing in this module returns a
market price, and the dashboard fills that gap with fair odds derived from the
model's own probability (1 / p, no margin). They are labelled as such wherever
they appear, because treating them as prices would be a lie in the user's
favour: a real bookmaker pays less than fair odds, never more.

Schema returned matches data/sample_data.load_matches(), so the dashboard's
scoring, filtering and slip building work unchanged.
"""

from __future__ import annotations

import json
import time
from datetime import date as date_cls
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests


BASE_URL = ("https://raw.githubusercontent.com/openfootball/football.json/"
            "master/{season}/{code}.json")

CACHE_DIR   = Path(__file__).resolve().parent / ".cache_live"
CACHE_TTL_S = 3600          # the feed updates about once a day
TIMEOUT_S   = 20

# openfootball league codes → display names
LEAGUES: dict[str, str] = {
    "en.1": "Premier League",
    "en.2": "Championship",
    "es.1": "La Liga",
    "de.1": "Bundesliga",
    "it.1": "Serie A",
    "fr.1": "Ligue 1",
    "nl.1": "Eredivisie",
    "pt.1": "Liga Portugal",
}

DEFAULT_LEAGUES = ["en.1", "es.1", "de.1", "it.1", "fr.1", "nl.1"]

# How many of a team's most recent matches feed its statistics.
FORM_WINDOW = 20

# Fallbacks for a club with no completed matches yet (a promoted side in
# August). These are league-average-ish, deliberately unexciting numbers.
FALLBACK_STATS = {
    "btts_rate":    0.50,
    "avg_scored":   1.35,
    "avg_conceded": 1.35,
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _cache_path(season: str, code: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{season}_{code}.json"


def fetch_league(season: str, code: str, force: bool = False) -> dict:
    """
    Fetch one league-season file, using a cached copy when it is fresh.

    Raises:
        requests.HTTPError: the season/league file does not exist upstream
    """
    path = _cache_path(season, code)

    if path.exists() and not force:
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_S:
            return json.loads(path.read_text())

    response = requests.get(BASE_URL.format(season=season, code=code),
                            timeout=TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload))
    return payload


def _full_time_score(score) -> tuple[list, bool]:
    """
    Read a match's full-time score, and say whether the feed gave a usable one.

    Scores normally arrive as {"ht": [...], "ft": [...]}. A minority arrive as
    a bare list instead — and across every league-season checked, that form is
    ALWAYS exactly [0, 0], dozens of times, never any other scoreline. Genuine
    goalless draws appear in the dict form, so these are placeholders for a
    result the feed does not have.

    They are therefore treated as unknown, not as 0-0. Counting them would
    push every team's BTTS rate down and invent clean sheets that never
    happened — a silent bias in exactly the statistic this app reports.

    Returns:
        (full_time_goals, score_was_malformed)
    """
    if isinstance(score, dict):
        full_time = score.get("ft") or []
        return (list(full_time) if len(full_time) == 2 else [], False)

    if isinstance(score, list):
        return [], True

    return [], False


def load_league(season: str, code: str, force: bool = False) -> pd.DataFrame:
    """
    One league-season as a flat table.

    Returns columns: kickoff, date, league, code, season, home_team, away_team,
    fthg, ftag, played. Unplayed fixtures carry NaN goals and played=False.
    """
    payload = fetch_league(season, code, force=force)
    rows = []

    for match in payload.get("matches", []):
        if not match.get("date") or not match.get("team1") or not match.get("team2"):
            continue

        full_time, malformed = _full_time_score(match.get("score"))
        played = len(full_time) == 2

        # Not every league in the feed publishes kickoff times. Recording a
        # missing time as midnight would show "00:00" as if it were real, so
        # the flag travels with the row and the display shows --:-- instead.
        time_known = bool(match.get("time"))
        stamp = f"{match['date']} {match.get('time') or '00:00'}"
        rows.append({
            "kickoff":   pd.to_datetime(stamp, errors="coerce"),
            "time_known": time_known,
            "date":      match["date"],
            "league":    LEAGUES.get(code, payload.get("name", code)),
            "code":      code,
            "season":    season,
            "round":     match.get("round"),
            "home_team": match["team1"].strip(),
            "away_team": match["team2"].strip(),
            "fthg":      full_time[0] if played else np.nan,
            "ftag":      full_time[1] if played else np.nan,
            "played":    played,
            "score_unknown": malformed,
        })

    frame = pd.DataFrame(rows).dropna(subset=["kickoff"])
    return frame.sort_values("kickoff").reset_index(drop=True)


def malformed_score_count(matches: pd.DataFrame) -> int:
    """How many matches the feed gave an unusable score for — see _full_time_score."""
    if matches.empty or "score_unknown" not in matches.columns:
        return 0
    return int(matches["score_unknown"].sum())


def load_leagues(
    seasons: Sequence[str],
    codes:   Iterable[str] = DEFAULT_LEAGUES,
    force:   bool = False,
) -> pd.DataFrame:
    """
    Load several leagues and seasons into one table, skipping any file the
    feed does not have (a league that has not started, or a season not yet
    published).
    """
    frames = []
    for season in seasons:
        for code in codes:
            try:
                frames.append(load_league(season, code, force=force))
            except Exception as exc:
                print(f"  ! skipped {season}/{code}: {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("kickoff").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team statistics from real results
# ---------------------------------------------------------------------------

def build_team_stats(matches: pd.DataFrame, window: int = FORM_WINDOW,
                     before: pd.Timestamp | None = None) -> dict[str, dict[str, float]]:
    """
    Per-team scoring, conceding and BTTS rates from completed matches.

    Args:
        matches: league table from load_leagues()
        window:  how many of a team's most recent matches to use
        before:  only count matches kicking off strictly before this moment,
                 so a fixture's statistics never include its own result

    Returns:
        {team: {btts_rate, avg_scored, avg_conceded, matches_used}}
    """
    played = matches[matches["played"]]
    if before is not None:
        played = played[played["kickoff"] < before]

    history: dict[str, list[tuple[float, float]]] = {}
    for row in played.sort_values("kickoff").itertuples(index=False):
        history.setdefault(row.home_team, []).append((row.fthg, row.ftag))
        history.setdefault(row.away_team, []).append((row.ftag, row.fthg))

    stats: dict[str, dict[str, float]] = {}
    for team, results in history.items():
        recent = results[-window:]
        scored   = np.array([s for s, _ in recent], dtype=float)
        conceded = np.array([c for _, c in recent], dtype=float)
        stats[team] = {
            "btts_rate":    float(np.mean((scored > 0) & (conceded > 0))),
            "avg_scored":   float(scored.mean()),
            "avg_conceded": float(conceded.mean()),
            "matches_used": len(recent),
        }

    return stats


def _team_stats_or_fallback(stats: dict[str, dict[str, float]], team: str) -> dict[str, float]:
    entry = stats.get(team)
    if entry and entry["matches_used"] > 0:
        return entry
    return {**FALLBACK_STATS, "matches_used": 0}


# ---------------------------------------------------------------------------
# Fixture card for the dashboard
# ---------------------------------------------------------------------------

def available_dates(matches: pd.DataFrame, upcoming_only: bool = True,
                    today: date_cls | None = None) -> list[date_cls]:
    """Dates that actually have fixtures, for the dashboard's date picker."""
    if matches.empty:
        return []
    dates = sorted({d.date() for d in pd.to_datetime(matches["kickoff"])})
    if upcoming_only:
        cutoff = today or date_cls.today()
        dates = [d for d in dates if d >= cutoff]
    return dates


def next_matchday(matches: pd.DataFrame, today: date_cls | None = None) -> date_cls | None:
    """
    The next date with fixtures, today included.

    Football has gaps — international breaks, midweek with no league games —
    so a dashboard hard-wired to "today" is blank more often than not. This is
    what it should show instead.
    """
    upcoming = available_dates(matches, upcoming_only=True, today=today)
    return upcoming[0] if upcoming else None


def build_fixture_card(
    matches:   pd.DataFrame,
    on_date:   date_cls | None = None,
    window:    int = FORM_WINDOW,
    unplayed_only: bool = True,
) -> pd.DataFrame:
    """
    Build one day's fixtures in the dashboard's schema.

    Team statistics come from results before that day's first kickoff, so a
    card never describes itself.

    Note the odds columns are NaN: this feed carries no bookmaker prices. The
    dashboard fills them with fair odds derived from the model probability and
    labels them as such.
    """
    empty = pd.DataFrame(columns=[
        "match_id", "home_team", "away_team", "kickoff", "kickoff_date", "league",
        "home_btts_rate", "away_btts_rate", "home_avg_scored", "away_avg_scored",
        "home_avg_conceded", "away_avg_conceded", "btts_odds",
        "home_win_odds", "draw_odds", "away_win_odds",
        "home_matches_used", "away_matches_used",
    ])
    if matches.empty:
        return empty

    target = on_date or next_matchday(matches)
    if target is None:
        return empty

    kickoffs = pd.to_datetime(matches["kickoff"])
    card = matches[kickoffs.dt.date == target]
    if unplayed_only:
        card = card[~card["played"]]
    if card.empty:
        return empty

    stats = build_team_stats(matches, window=window,
                             before=pd.to_datetime(card["kickoff"]).min())

    rows = []
    for i, match in enumerate(card.sort_values("kickoff").itertuples(index=False)):
        home = _team_stats_or_fallback(stats, match.home_team)
        away = _team_stats_or_fallback(stats, match.away_team)

        rows.append({
            "match_id":          f"{match.code}-{target:%Y%m%d}-{i:02d}",
            "home_team":         match.home_team,
            "away_team":         match.away_team,
            "kickoff":           (pd.to_datetime(match.kickoff).strftime("%Y-%m-%d %H:%M")
                                  if getattr(match, "time_known", True)
                                  else pd.to_datetime(match.kickoff).strftime("%Y-%m-%d --:--")),
            "kickoff_date":      f"{target:%Y-%m-%d}",
            "league":            match.league,
            "home_btts_rate":    round(home["btts_rate"], 2),
            "away_btts_rate":    round(away["btts_rate"], 2),
            "home_avg_scored":   round(home["avg_scored"], 2),
            "away_avg_scored":   round(away["avg_scored"], 2),
            "home_avg_conceded": round(home["avg_conceded"], 2),
            "away_avg_conceded": round(away["avg_conceded"], 2),
            # No market prices in this feed — filled in as fair odds downstream.
            "btts_odds":         np.nan,
            "home_win_odds":     np.nan,
            "draw_odds":         np.nan,
            "away_win_odds":     np.nan,
            "home_matches_used": home["matches_used"],
            "away_matches_used": away["matches_used"],
        })

    return pd.DataFrame(rows)


def fair_odds(probability: float, floor: float = 1.01, cap: float = 50.0) -> float:
    """
    The break-even price for a probability: 1 / p, with no bookmaker margin.

    A real book prices below this. Fair odds are useful for comparing legs and
    for sizing a hypothetical return, and useless as a promise of payout.
    """
    if not np.isfinite(probability) or probability <= 0:
        return np.nan
    return float(np.clip(round(1.0 / probability, 2), floor, cap))


def feed_status(seasons: Sequence[str], codes: Iterable[str] = DEFAULT_LEAGUES) -> dict:
    """When the local cache was last refreshed, for display in the UI."""
    paths = [_cache_path(s, c) for s in seasons for c in codes]
    stamps = [p.stat().st_mtime for p in paths if p.exists()]
    return {
        "cached_files": len(stamps),
        "last_fetched": (pd.Timestamp(max(stamps), unit="s") if stamps else None),
    }
