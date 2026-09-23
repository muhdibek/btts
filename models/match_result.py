"""
match_result.py
---------------
Match-result (1X2) probabilities for live fixtures.

This is the market the bake-off found signal in. Over 8,770 out-of-sample
Premier League matches, the Poisson family scored AUC 0.672 and +6.9% skill
against the base rate on the home-win market, while the same models on the same
matches managed AUC 0.513 on BTTS — a coin flip. So the dashboard's numbers mean
something here in a way they never did for both-teams-to-score.

One caveat carries through everything below: **skill against the base rate is
not an edge against a bookmaker.** The base rate is a weak opponent; a closing
price is not. No odds source here carries prices for these fixtures, so nothing
in this module has been shown to beat a market, and none of it should be read
as a value bet.

Models are fitted PER LEAGUE. Scoring rates differ enough between competitions
that a single pooled intercept misfits all of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.goal_models import PoissonModel


# A league needs enough completed matches before its ratings mean anything.
MIN_MATCHES_TO_FIT = 80

OUTCOME_LABELS = {
    "p_home": "Home win",
    "p_draw": "Draw",
    "p_away": "Away win",
}


def fit_league_models(
    results: pd.DataFrame,
    before: pd.Timestamp | None = None,
    min_matches: int = MIN_MATCHES_TO_FIT,
    half_life_days: float | None = None,
) -> dict[str, PoissonModel]:
    """
    Fit one Poisson model per league on completed matches.

    Args:
        results:        played matches (league, home_team, away_team, fthg, ftag, kickoff)
        before:         only train on matches kicking off strictly earlier, so a
                        fixture is never predicted by a model that has seen it
        min_matches:    leagues with less history than this are skipped
        half_life_days: optional recency weighting

    Returns:
        {league: fitted model} for every league with enough history.
    """
    played = results[results["played"]] if "played" in results.columns else results
    played = played.dropna(subset=["fthg", "ftag"])
    if before is not None:
        played = played[played["kickoff"] < before]

    models: dict[str, PoissonModel] = {}
    for league, frame in played.groupby("league"):
        if len(frame) < min_matches:
            continue
        models[league] = PoissonModel(half_life_days=half_life_days).fit(
            frame.sort_values("kickoff")
        )
    return models


def predict_fixtures(fixtures: pd.DataFrame,
                     models: dict[str, PoissonModel]) -> pd.DataFrame:
    """
    Add p_home / p_draw / p_away to a fixture card.

    A fixture whose league has no fitted model (too little history) gets NaN
    rather than a guess, and the dashboard drops it with a note.
    """
    rows = []
    for fixture in fixtures.itertuples(index=False):
        model = models.get(fixture.league)
        if model is None:
            rows.append({"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan})
            continue
        markets = model.predict_markets(fixture.home_team, fixture.away_team)
        rows.append({k: markets[k] for k in ("p_home", "p_draw", "p_away")})

    return pd.concat([fixtures.reset_index(drop=True),
                      pd.DataFrame(rows)], axis=1)


def add_best_selection(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Pick each fixture's most likely outcome and name it.

    Adds:
        selection        "Home win" / "Draw" / "Away win"
        selection_team   the club backed, or "Draw"
        selection_prob   that outcome's probability
        selection_label  human-readable, e.g. "Liverpool FC to win"
    """
    out = fixtures.copy()
    columns = ["p_home", "p_draw", "p_away"]

    if out.empty or not set(columns).issubset(out.columns):
        for column in ("selection", "selection_team", "selection_prob", "selection_label"):
            out[column] = np.nan
        return out

    probabilities = out[columns].to_numpy(dtype=float)
    # A row the model could not price stays unpriced.
    unpriced = ~np.isfinite(probabilities).all(axis=1)
    safe = np.where(np.isfinite(probabilities), probabilities, -np.inf)
    best = safe.argmax(axis=1)

    selections, teams, probs, labels = [], [], [], []
    for i, (row, index) in enumerate(zip(out.itertuples(index=False), best)):
        if unpriced[i]:
            selections.append(None); teams.append(None)
            probs.append(np.nan); labels.append(None)
            continue

        key   = columns[index]
        label = OUTCOME_LABELS[key]
        team  = (row.home_team if key == "p_home"
                 else row.away_team if key == "p_away" else "Draw")
        selections.append(label)
        teams.append(team)
        probs.append(float(probabilities[i, index]))
        labels.append("Draw" if key == "p_draw" else f"{team} to win")

    out["selection"]       = selections
    out["selection_team"]  = teams
    out["selection_prob"]  = probs
    out["selection_label"] = labels
    return out


def build_match_result_card(
    fixtures: pd.DataFrame,
    season:   pd.DataFrame,
    half_life_days: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Price a day's fixtures on the match-result market.

    Models are fitted only on matches played before the card's first kickoff.

    Returns:
        (card, meta) — the fixtures with probabilities and a chosen selection,
        and a summary of what was fitted, for display.
    """
    if fixtures.empty:
        return fixtures, {"leagues_fitted": 0, "training_matches": 0, "unpriced": 0}

    cutoff = pd.to_datetime(fixtures["kickoff"].str.slice(0, 10)).min()
    models = fit_league_models(season, before=cutoff, half_life_days=half_life_days)

    priced = add_best_selection(predict_fixtures(fixtures, models))

    played = season[season["played"]] if "played" in season.columns else season
    meta = {
        "leagues_fitted":   len(models),
        "training_matches": int((played["kickoff"] < cutoff).sum()),
        "unpriced":         int(priced["selection_prob"].isna().sum()),
    }
    return priced, meta
