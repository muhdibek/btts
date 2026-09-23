"""
football_data.py
----------------
Ingestion for football-data.co.uk season CSVs — the labelled history the
BTTS model needs for training, calibration and backtesting.

Why this source:
  - free, no API key, no rate limit
  - full-time scores, so the BTTS label is exact (FTHG > 0 and FTAG > 0)
  - closing bookmaker odds in the same row, which gives both a strong feature
    and the benchmark any model has to beat
  - 20+ seasons across the major European leagues

File layout on the site:
    https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
    season = 4-digit code, e.g. 2425 for the 2024/25 season
    league = division code, e.g. E0 for the Premier League

Downloads are cached under data/raw/ so a re-run costs nothing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests


BASE_URL     = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
CACHE_DIR    = Path(__file__).resolve().parent / "raw"
REQUEST_TIMEOUT_S = 30


# Division codes → human names. The main-league files carry the richest odds.
LEAGUES: dict[str, str] = {
    "E0":  "Premier League",
    "E1":  "Championship",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "D1":  "Bundesliga",
    "D2":  "Bundesliga 2",
    "I1":  "Serie A",
    "I2":  "Serie B",
    "F1":  "Ligue 1",
    "F2":  "Ligue 2",
    "N1":  "Eredivisie",
    "B1":  "Jupiler League",
    "P1":  "Liga Portugal",
    "T1":  "Super Lig",
    "G1":  "Greek Super League",
    "SC0": "Scottish Premiership",
}

# Columns we keep. The site adds and drops bookmakers between seasons, so
# anything missing from a given file is filled with NaN rather than failing.
CORE_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full time goals / result
    "HTHG", "HTAG",                 # half time goals
    "HS", "AS", "HST", "AST",       # shots / shots on target
    "HC", "AC",                     # corners
]

# Odds columns: Bet365 where available, market average as the fallback.
ODDS_COLUMNS = [
    "B365H", "B365D", "B365A",      # 1X2
    "AvgH", "AvgD", "AvgA",
    "B365>2.5", "B365<2.5",         # over / under 2.5 goals
    "Avg>2.5", "Avg<2.5",
]

NUMERIC_COLUMNS = [c for c in CORE_COLUMNS + ODDS_COLUMNS
                   if c not in ("Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR")]


# ---------------------------------------------------------------------------
# Season codes
# ---------------------------------------------------------------------------

def season_code(start_year: int) -> str:
    """
    Convert a season's starting year to the site's 4-digit code.

        2024 → '2425'   (the 2024/25 season)
        1999 → '9900'
    """
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Human-readable season label, e.g. 2024 → '2024/25'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------

def _cache_path(league: str, start_year: int, cache_dir: Path) -> Path:
    return cache_dir / f"{league}_{season_code(start_year)}.csv"


def download_season(
    league:     str,
    start_year: int,
    cache_dir:  Path | str = CACHE_DIR,
    force:      bool = False,
) -> Path:
    """
    Fetch one league-season CSV into the cache and return its path.

    A cached file is reused unless `force` is set. Note that the current
    season's file changes as matches are played — pass force=True to refresh it.

    Raises:
        requests.HTTPError: the season/league combination does not exist
                            (the site 404s, or serves an HTML error page)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(league, start_year, cache_dir)

    if path.exists() and not force:
        return path

    url  = BASE_URL.format(season=season_code(start_year), league=league)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()

    # The site answers 200 with an HTML error page for a missing season, so
    # check that what came back actually looks like the expected CSV.
    head = resp.content[:200].lstrip().lower()
    if head.startswith(b"<"):
        raise requests.HTTPError(f"{url} did not return a CSV (got an HTML page)")

    path.write_bytes(resp.content)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_kickoff(df: pd.DataFrame) -> pd.Series:
    """
    Build a kickoff timestamp from the Date (and Time, when the file has it).

    Dates are day-first and appear as both dd/mm/yy and dd/mm/yyyy, sometimes
    within one file, so they are parsed per-row rather than with a fixed format.
    """
    # format="mixed" because dd/mm/yy and dd/mm/yyyy both occur, sometimes in
    # the same file — a single inferred format turns the odd ones into NaT.
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")

    if "Time" in df.columns:
        times   = df["Time"].fillna("00:00").astype(str)
        stamps  = pd.to_datetime(
            dates.dt.strftime("%Y-%m-%d") + " " + times,
            errors="coerce",
        )
        # Fall back to the date alone wherever the time failed to parse.
        return stamps.fillna(dates)

    return dates


def parse_season_csv(path: Path | str, start_year: int | None = None) -> pd.DataFrame:
    """
    Read one cached season CSV into the normalised match schema.

    Returns columns:
        kickoff, date, div, season, home_team, away_team,
        fthg, ftag, ftr, hthg, htag, hs, a_s, hst, ast, hc, ac,
        the odds columns, and the btts label.

    Rows without a full-time score (fixtures not yet played, or malformed
    trailing rows the site sometimes leaves in) are dropped — an unplayed
    match has no label and must never reach a training set.
    """
    path = Path(path)
    # Some older files carry stray non-UTF8 bytes in team names.
    raw = pd.read_csv(io.BytesIO(path.read_bytes()), encoding="latin-1",
                      on_bad_lines="skip")

    for col in CORE_COLUMNS + ODDS_COLUMNS:
        if col not in raw.columns:
            raw[col] = pd.NA

    for col in NUMERIC_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    kickoff = _parse_kickoff(raw)

    out = pd.DataFrame({
        "kickoff":   kickoff,
        "div":       raw["Div"],
        "home_team": raw["HomeTeam"].astype("string").str.strip(),
        "away_team": raw["AwayTeam"].astype("string").str.strip(),
        "fthg":      raw["FTHG"],
        "ftag":      raw["FTAG"],
        "ftr":       raw["FTR"],
        "hthg":      raw["HTHG"],
        "htag":      raw["HTAG"],
        "hs":        raw["HS"],
        "a_s":       raw["AS"],      # 'AS' renamed: `df.as` is a Python keyword
        "hst":       raw["HST"],
        "ast":       raw["AST"],
        "hc":        raw["HC"],
        "ac":        raw["AC"],
    })

    for col in ODDS_COLUMNS:
        out[col.lower().replace(">", "over_").replace("<", "under_")] = raw[col]

    if start_year is not None:
        out["season"] = season_label(start_year)
    else:
        out["season"] = pd.NA

    out = out.dropna(subset=["kickoff", "home_team", "away_team", "fthg", "ftag"])
    out = out[(out["home_team"] != "") & (out["away_team"] != "")]

    out["fthg"] = out["fthg"].astype(int)
    out["ftag"] = out["ftag"].astype(int)

    # --- the label ---
    out["btts"]        = ((out["fthg"] > 0) & (out["ftag"] > 0)).astype(int)
    out["total_goals"] = out["fthg"] + out["ftag"]
    out["date"]        = out["kickoff"].dt.date

    return out.sort_values("kickoff").reset_index(drop=True)


def load_matches(
    leagues:    Sequence[str],
    seasons:    Iterable[int],
    cache_dir:  Path | str = CACHE_DIR,
    force:      bool = False,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """
    Download (or reuse) every league-season combination and return one
    chronologically sorted match table.

    Args:
        leagues:      division codes, e.g. ["E0", "SP1"]
        seasons:      season start years, e.g. range(2014, 2025)
        cache_dir:    where CSVs are cached
        force:        re-download even if cached
        skip_missing: skip league-seasons the site does not have (a newer
                      division, or a season before the file series starts)
                      instead of raising

    Returns:
        DataFrame in the normalised schema, sorted by kickoff.
    """
    frames = []
    for league in leagues:
        for year in seasons:
            try:
                path = download_season(league, year, cache_dir=cache_dir, force=force)
            except Exception as exc:                      # network or missing file
                if skip_missing:
                    print(f"  ! skipped {league} {season_label(year)}: {exc}")
                    continue
                raise
            frames.append(parse_season_csv(path, start_year=year))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("kickoff").reset_index(drop=True)
