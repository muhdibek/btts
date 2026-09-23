"""
international.py
----------------
National-team results and pricing for manually chosen fixtures.

Source: https://github.com/martj42/international_results — 49,000+ international
matches from 1872 to the present, with a neutral-venue flag and the competition
each was played in. It is results only: **no source reachable here lists
upcoming international fixtures**, which is why this market is a manual pairing
rather than a card. You name the two teams; the model prices that match.

Two things make internationals different from league football, and both are
handled explicitly:

  Neutral venues   Tournament matches are routinely played on neutral ground,
                   where "home" is just a label on the fixture. The fitted model
                   drops its home-advantage term for those.
  Thin, old data   A national team plays a handful of matches a year, so the
                   useful history is long in years and short in matches. The fit
                   uses a date window plus exponential time decay, and friendlies
                   can be down-weighted — they are played with rotated squads and
                   predict competitive matches poorly.

Read the output with more caution than the club pages. The bake-off validated
these models on club leagues; nothing here has been backtested on international
football.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from models.goal_models import PoissonModel


RESULTS_URL = ("https://raw.githubusercontent.com/martj42/international_results/"
               "master/results.csv")

CACHE_DIR   = Path(__file__).resolve().parent / ".cache_live"
CACHE_TTL_S = 86400          # the file gains a few rows a week at most
TIMEOUT_S   = 60

DEFAULT_SINCE       = "2022-01-01"   # ~2 World Cup cycles of current squads
DEFAULT_HALF_LIFE   = 540.0          # days; ~18 months
FRIENDLY_WEIGHT     = 0.5            # friendlies count half
MIN_MATCHES_TO_RATE = 4              # below this a team's rating is noise
RIDGE               = 0.75           # shrinkage on ratings, for thin samples


def _cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "international_results.csv"


def fetch_results(force: bool = False) -> pd.DataFrame:
    """Download (or reuse) the results file. ~3.7 MB, cached for a day."""
    path = _cache_path()

    if path.exists() and not force and (time.time() - path.stat().st_mtime) < CACHE_TTL_S:
        return pd.read_csv(path)

    response = requests.get(RESULTS_URL, timeout=TIMEOUT_S)
    response.raise_for_status()
    path.write_bytes(response.content)
    return pd.read_csv(path)


def load_results(since: str = DEFAULT_SINCE, force: bool = False) -> pd.DataFrame:
    """
    Normalised international results from `since` onwards.

    Returns: kickoff, home_team, away_team, fthg, ftag, neutral, tournament,
    friendly, played.
    """
    raw = fetch_results(force=force)

    out = pd.DataFrame({
        "kickoff":    pd.to_datetime(raw["date"], errors="coerce"),
        "home_team":  raw["home_team"].astype("string").str.strip(),
        "away_team":  raw["away_team"].astype("string").str.strip(),
        "fthg":       pd.to_numeric(raw["home_score"], errors="coerce"),
        "ftag":       pd.to_numeric(raw["away_score"], errors="coerce"),
        "tournament": raw["tournament"].astype("string"),
        # The file stores TRUE/True/true across its history.
        "neutral":    raw["neutral"].astype(str).str.upper().eq("TRUE").astype(int),
    })

    out = out.dropna(subset=["kickoff", "home_team", "away_team", "fthg", "ftag"])
    out = out[out["kickoff"] >= pd.Timestamp(since)]
    out["fthg"]     = out["fthg"].astype(int)
    out["ftag"]     = out["ftag"].astype(int)
    out["friendly"] = out["tournament"].str.contains("friendly", case=False, na=False).astype(int)
    out["played"]   = True

    return out.sort_values("kickoff").reset_index(drop=True)


def team_match_counts(results: pd.DataFrame) -> pd.Series:
    """How many matches each team has in the window."""
    appearances = pd.concat([results["home_team"], results["away_team"]])
    return appearances.value_counts()


def rateable_teams(results: pd.DataFrame,
                   min_matches: int = MIN_MATCHES_TO_RATE) -> list[str]:
    """
    Teams with enough recent matches to rate.

    A side with one or two results in four years cannot be rated; offering it
    in the picker would produce a confident-looking number built on nothing.
    """
    counts = team_match_counts(results)
    return sorted(counts[counts >= min_matches].index.tolist())


def fit_international_model(
    results:         pd.DataFrame,
    half_life_days:  float | None = DEFAULT_HALF_LIFE,
    friendly_weight: float = FRIENDLY_WEIGHT,
) -> PoissonModel:
    """
    Fit attack/defence ratings to international results.

    Friendlies are down-weighted by repeating competitive matches rather than
    by reaching into the optimiser: duplicating a row is exactly equivalent to
    doubling its likelihood weight, and it keeps the model class unchanged.
    """
    frame = results.copy()

    if friendly_weight < 1.0:
        competitive = frame[frame["friendly"] == 0]
        repeats     = max(int(round(1.0 / max(friendly_weight, 0.01))) - 1, 0)
        frame = pd.concat([frame] + [competitive] * repeats, ignore_index=True)
        frame = frame.sort_values("kickoff").reset_index(drop=True)

    # Hundreds of national teams means hundreds of parameters, so both budgets
    # are raised; ridge shrinkage keeps a side with four matches from earning
    # an extreme rating on almost no evidence.
    return PoissonModel(half_life_days=half_life_days, max_iter=4000,
                        max_fun=400_000, ridge=RIDGE).fit(frame)


def recent_form(results: pd.DataFrame, team: str, window: int = 8) -> pd.DataFrame:
    """That team's most recent matches, newest first, for display."""
    played = results[(results["home_team"] == team) | (results["away_team"] == team)]
    rows = []
    for match in played.sort_values("kickoff", ascending=False).head(window).itertuples(index=False):
        at_home = match.home_team == team
        scored, conceded = ((match.fthg, match.ftag) if at_home else (match.ftag, match.fthg))
        rows.append({
            "date":       match.kickoff.date(),
            "opponent":   match.away_team if at_home else match.home_team,
            "venue":      "N" if match.neutral else ("H" if at_home else "A"),
            "score":      f"{scored}-{conceded}",
            "result":     "W" if scored > conceded else ("D" if scored == conceded else "L"),
            "tournament": match.tournament,
        })
    return pd.DataFrame(rows)


def price_fixture(model: PoissonModel, home_team: str, away_team: str,
                  neutral: bool = True) -> dict[str, float]:
    """
    Price one manually chosen international fixture.

    Neutral defaults to True: a made-up pairing has no host until you say
    otherwise, and assuming a home advantage nobody has earned would tilt
    every number toward whichever team was typed first.
    """
    markets = model.predict_markets(home_team, away_team, neutral=neutral)
    lam_home, lam_away = model.fit_result.rates(home_team, away_team, neutral=neutral)
    return {
        **markets,
        "xg_home": round(lam_home, 2),
        "xg_away": round(lam_away, 2),
    }
