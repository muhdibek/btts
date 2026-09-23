"""
market_data.py
--------------
Historical matches WITH bookmaker odds.

Source: https://github.com/xgabora/Club-Football-Match-Data-2000-2025 —
~239,000 matches from 27 countries, 2000 to the present, carrying closing 1X2
and over/under 2.5 prices (both the market average and the best price
available), plus pre-computed Elo and form.

Why this matters more than another results file: every model in this repo has
so far been scored against the base rate, and the base rate is a weak opponent.
A closing price is not. With these odds the real question can finally be asked —
would the model have beaten the market? — which is the only version of "is there
an edge" that decides anything.

What it still does not give is odds for matches that have NOT been played. No
free source reachable from here publishes forward prices, so the live dashboard
remains fair-odds only; this is for backtesting.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


MATCHES_URL = ("https://raw.githubusercontent.com/xgabora/"
               "Club-Football-Match-Data-2000-2025/main/data/Matches.csv")

CACHE_DIR   = Path(__file__).resolve().parent / ".cache_live"
CACHE_TTL_S = 7 * 86400
TIMEOUT_S   = 180

# The divisions the rest of the app models.
BIG_LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "N1"]


def _cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "club_matches_with_odds.csv"


def fetch_matches(force: bool = False) -> Path:
    """Download the matches file (~69 MB) into the cache."""
    path = _cache_path()
    if path.exists() and not force and (time.time() - path.stat().st_mtime) < CACHE_TTL_S:
        return path

    with requests.get(MATCHES_URL, timeout=TIMEOUT_S, stream=True) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return path


def load_matches(
    path:      Path | str | None = None,
    divisions: list[str] | None = None,
    since:     str | None = "2010-01-01",
    require_odds: bool = True,
) -> pd.DataFrame:
    """
    Load played matches with their closing prices, in this repo's schema.

    Args:
        path:         local CSV (defaults to the cached download)
        divisions:    division codes to keep, e.g. ["E0"]; None keeps all
        since:        drop matches before this date
        require_odds: keep only rows that actually carry a 1X2 price

    Returns:
        kickoff, div, home_team, away_team, fthg, ftag, btts, plus
        odds_home/draw/away (market average), max_home/draw/away (best price),
        odds_over25 / odds_under25, and the source's Elo and form columns.
    """
    source = Path(path) if path else fetch_matches()
    raw = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

    out = pd.DataFrame({
        "kickoff":   pd.to_datetime(
            raw["MatchDate"].astype(str) + " " + raw["MatchTime"].fillna("00:00:00").astype(str),
            errors="coerce"),
        "div":       raw["Division"].astype("string").str.strip(),
        "home_team": raw["HomeTeam"].astype("string").str.strip(),
        "away_team": raw["AwayTeam"].astype("string").str.strip(),
        "fthg":      pd.to_numeric(raw["FTHome"], errors="coerce"),
        "ftag":      pd.to_numeric(raw["FTAway"], errors="coerce"),
        "home_elo":  pd.to_numeric(raw.get("HomeElo"), errors="coerce"),
        "away_elo":  pd.to_numeric(raw.get("AwayElo"), errors="coerce"),
        "form5_home": pd.to_numeric(raw.get("Form5Home"), errors="coerce"),
        "form5_away": pd.to_numeric(raw.get("Form5Away"), errors="coerce"),
    })

    for target, column in (("odds_home", "OddHome"), ("odds_draw", "OddDraw"),
                           ("odds_away", "OddAway"), ("max_home", "MaxHome"),
                           ("max_draw", "MaxDraw"), ("max_away", "MaxAway"),
                           ("odds_over25", "Over25"), ("odds_under25", "Under25"),
                           ("max_over25", "MaxOver25"), ("max_under25", "MaxUnder25")):
        out[target] = pd.to_numeric(raw.get(column), errors="coerce")

    out = out.dropna(subset=["kickoff", "home_team", "away_team", "fthg", "ftag"])
    if since:
        out = out[out["kickoff"] >= pd.Timestamp(since)]
    if divisions:
        out = out[out["div"].isin(divisions)]
    if require_odds:
        out = out.dropna(subset=["odds_home", "odds_draw", "odds_away"])

    out["fthg"] = out["fthg"].astype(int)
    out["ftag"] = out["ftag"].astype(int)
    out["btts"] = ((out["fthg"] > 0) & (out["ftag"] > 0)).astype(int)
    out["played"] = True

    return out.sort_values("kickoff").reset_index(drop=True)


def market_probabilities(matches: pd.DataFrame, best_price: bool = False) -> pd.DataFrame:
    """
    Closing prices converted to probabilities, with the overround removed
    proportionally.

    Args:
        best_price: use the best price available across books (Max*) rather
                    than the market average. The best price carries a smaller
                    overround — sometimes none at all — so it is the harder
                    benchmark and the one a bettor would actually take.
    """
    prefix = "max" if best_price else "odds"
    home = matches[f"{prefix}_home"].to_numpy(dtype=float)
    draw = matches[f"{prefix}_draw"].to_numpy(dtype=float)
    away = matches[f"{prefix}_away"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.vstack([1 / home, 1 / draw, 1 / away])
        overround = raw.sum(axis=0)
        normalised = raw / overround

    return pd.DataFrame({
        "mkt_p_home":  normalised[0],
        "mkt_p_draw":  normalised[1],
        "mkt_p_away":  normalised[2],
        "overround":   overround - 1.0,
    }, index=matches.index)
