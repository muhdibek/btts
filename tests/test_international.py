"""
test_international.py
---------------------
Tests for national-team pricing.

International football differs from club football in two ways that matter to a
model, and both are asserted here: matches are often played on neutral ground
where no side has a home advantage, and a national team's history is thin
enough that ratings need shrinking toward the mean.

No network: the results frame is built inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.international import (fit_international_model, price_fixture,    # noqa: E402
                                rateable_teams, recent_form, team_match_counts)
from models.goal_models import MAX_RATE_GOALS, PoissonModel                # noqa: E402


def _results(rows: list[tuple]) -> pd.DataFrame:
    """(date, home, away, hg, ag, neutral, friendly) → results frame."""
    df = pd.DataFrame(rows, columns=["kickoff", "home_team", "away_team",
                                     "fthg", "ftag", "neutral", "friendly"])
    df["kickoff"]    = pd.to_datetime(df["kickoff"])
    df["tournament"] = np.where(df["friendly"] == 1, "Friendly", "Qualifier")
    df["played"]     = True
    return df.sort_values("kickoff").reset_index(drop=True)


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    """A small round-robin where Strong beats Weak reliably."""
    rows = []
    date = pd.Timestamp("2024-01-01")
    for i in range(12):
        date += pd.Timedelta(days=30)
        rows.append((date, "Strong", "Weak", 3, 0, 0, 0))
        rows.append((date, "Weak", "Strong", 0, 2, 0, 0))
        rows.append((date, "Strong", "Middling", 2, 1, 0, 0))
        rows.append((date, "Middling", "Weak", 2, 0, 0, 0))
    return _results(rows)


# ---------------------------------------------------------------------------
# Neutral venues
# ---------------------------------------------------------------------------

def test_home_advantage_is_dropped_at_a_neutral_venue(league):
    model = fit_international_model(league, half_life_days=None, friendly_weight=1.0)
    at_home = price_fixture(model, "Strong", "Middling", neutral=False)
    neutral = price_fixture(model, "Strong", "Middling", neutral=True)
    assert at_home["p_home"] > neutral["p_home"]
    assert at_home["xg_home"] > neutral["xg_home"]


def test_neutral_flag_is_learned_not_assumed():
    """
    With every match on neutral ground the fit has no home matches to learn an
    advantage from, so a neutral prediction must not invent one.
    """
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(10):
        date += pd.Timedelta(days=20)
        rows.append((date, "Alpha", "Bravo", 1, 1, 1, 0))
        rows.append((date, "Bravo", "Alpha", 1, 1, 1, 0))
    model = fit_international_model(_results(rows), half_life_days=None, friendly_weight=1.0)
    priced = price_fixture(model, "Alpha", "Bravo", neutral=True)
    assert priced["p_home"] == pytest.approx(priced["p_away"], abs=0.03)


def test_pricing_ranks_the_stronger_side_higher(league):
    model = fit_international_model(league, half_life_days=None, friendly_weight=1.0)
    priced = price_fixture(model, "Strong", "Weak", neutral=True)
    assert priced["p_home"] > priced["p_away"]
    assert priced["xg_home"] > priced["xg_away"]


def test_probabilities_are_coherent(league):
    model = fit_international_model(league, half_life_days=None, friendly_weight=1.0)
    priced = price_fixture(model, "Strong", "Middling", neutral=True)
    assert priced["p_home"] + priced["p_draw"] + priced["p_away"] == pytest.approx(1.0)
    assert 0.0 <= priced["p_btts"] <= 1.0


# ---------------------------------------------------------------------------
# Extrapolation guard
# ---------------------------------------------------------------------------

def test_expected_goals_cannot_run_away():
    """
    The best side against the weakest is a pairing the ratings never saw. Left
    alone the exponential produces double-figure expected goals, which is not
    football and degenerates the scoreline grid.
    """
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(10):
        date += pd.Timedelta(days=20)
        rows.append((date, "Titan", "Minnow", 9, 0, 0, 0))
        rows.append((date, "Minnow", "Titan", 0, 9, 0, 0))
    model = fit_international_model(_results(rows), half_life_days=None, friendly_weight=1.0)
    priced = price_fixture(model, "Titan", "Minnow", neutral=True)
    assert priced["xg_home"] <= MAX_RATE_GOALS
    assert priced["p_home"] + priced["p_draw"] + priced["p_away"] == pytest.approx(1.0)


def test_ridge_shrinks_ratings_toward_the_mean(league):
    """A team with almost no history should not earn an extreme rating."""
    plain    = PoissonModel(ridge=0.0).fit(league)
    shrunken = PoissonModel(ridge=2.0).fit(league)
    spread   = lambda fit: max(fit.attack.values()) - min(fit.attack.values())
    assert spread(shrunken.fit_result) < spread(plain.fit_result)


def test_club_fits_are_unchanged_by_default(league):
    """Ridge defaults to zero so the backtested club numbers stay reproducible."""
    assert PoissonModel().ridge == 0.0


# ---------------------------------------------------------------------------
# Team selection and form
# ---------------------------------------------------------------------------

def test_rateable_teams_excludes_thin_histories():
    rows = [(pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i),
             "Regular", "Alsoregular", 1, 1, 0, 0) for i in range(6)]
    rows.append((pd.Timestamp("2024-06-01"), "Oneoff", "Regular", 0, 3, 0, 1))
    results = _results(rows)

    teams = rateable_teams(results, min_matches=4)
    assert "Regular" in teams
    assert "Oneoff" not in teams
    assert team_match_counts(results)["Oneoff"] == 1


def test_recent_form_reads_from_the_team_s_own_side(league):
    form = recent_form(league, "Weak", window=4)
    assert len(form) == 4
    assert set(form["venue"]) <= {"H", "A", "N"}
    assert (form["opponent"] != "Weak").all()
    # Weak never wins in this fixture set.
    assert "W" not in set(form["result"])


def test_friendly_weighting_changes_the_fit():
    """
    Competitive results should count for more than friendlies.

    Both orientations of each fixture are included deliberately. With one side
    always at home, the strength difference is absorbed by the home-advantage
    term instead of the team ratings, and a neutral-venue prediction cannot see
    it at all.
    """
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(8):
        date += pd.Timedelta(days=20)
        # competitive: Alpha is the better side, home and away
        rows.append((date, "Alpha", "Bravo", 3, 0, 0, 0))
        rows.append((date, "Bravo", "Alpha", 0, 3, 0, 0))
        # friendlies say the opposite
        rows.append((date, "Alpha", "Bravo", 0, 3, 0, 1))
        rows.append((date, "Bravo", "Alpha", 3, 0, 0, 1))
    results = _results(rows)

    equal  = fit_international_model(results, half_life_days=None, friendly_weight=1.0)
    tilted = fit_international_model(results, half_life_days=None, friendly_weight=0.25)

    # Under equal weighting the two sides cancel out; down-weighting friendlies
    # lets the competitive record show.
    assert equal.fit_result.attack["Alpha"] == pytest.approx(
        equal.fit_result.attack["Bravo"], abs=0.05)
    assert tilted.fit_result.attack["Alpha"] > tilted.fit_result.attack["Bravo"]
    assert (price_fixture(tilted, "Alpha", "Bravo", neutral=True)["p_home"] >
            price_fixture(equal,  "Alpha", "Bravo", neutral=True)["p_home"])
