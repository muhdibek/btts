"""
test_odds_link.py
-----------------
Tests for joining an odds feed to a fixture card.

The failure this guards against is silent: providers spell clubs differently,
a naive join drops the rows that disagree, and the page then shows prices for
part of its card with nothing saying which part. So the tests cover the
spellings that actually differ between feeds, and they check that what does
NOT match is reported rather than quietly lost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.odds_link import (add_market_comparison, coverage_note,          # noqa: E402
                            implied_probability, link_odds, name_similarity,
                            normalise_name)


def _card(rows):
    df = pd.DataFrame(rows, columns=["kickoff", "home_team", "away_team"])
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["prob"] = 0.6
    return df


def _odds(rows):
    df = pd.DataFrame(rows, columns=["kickoff", "home_team", "away_team",
                                     "max_home", "max_draw", "max_away"])
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    return df


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Manchester Utd", "Manchester United FC"),
    ("Man United", "Manchester United"),
    ("Tottenham Hotspur FC", "Tottenham"),
    ("Málaga CF", "Malaga"),
    ("Borussia Mönchengladbach", "Gladbach"),
    ("Internazionale Milano", "Inter"),
    ("Brighton & Hove Albion FC", "Brighton and Hove Albion"),
    ("TSG 1899 Hoffenheim", "Hoffenheim"),
])
def test_the_same_club_spelled_differently_matches(a, b):
    assert name_similarity(a, b) >= 0.82, f"{a!r} vs {b!r} = {name_similarity(a, b):.2f}"


@pytest.mark.parametrize("a,b", [
    ("Manchester United", "Manchester City"),
    ("Real Madrid", "Real Sociedad"),
    ("Nottingham Forest", "Norwich City"),
])
def test_different_clubs_do_not_match(a, b):
    assert name_similarity(a, b) < 0.82, f"{a!r} vs {b!r} = {name_similarity(a, b):.2f}"


def test_normalisation_strips_noise_not_identity():
    assert normalise_name("AFC Bournemouth") == "bournemouth"
    assert normalise_name("1. FC Köln") == "1 koln"
    assert normalise_name("") == ""
    assert normalise_name(None) == ""


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def test_prices_attach_across_different_spellings():
    card = _card([("2026-10-10 15:00", "Manchester Utd", "Leeds United FC")])
    odds = _odds([("2026-10-10 15:00", "Manchester United", "Leeds", 2.1, 3.4, 3.6)])

    linked, unmatched = link_odds(card, odds)
    assert bool(linked["odds_matched"].iloc[0])
    assert linked["max_home"].iloc[0] == pytest.approx(2.1)
    assert unmatched.empty


def test_an_unmatched_fixture_is_reported_not_dropped():
    """The card keeps every row; the missing price is visible as a NaN."""
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea"),
                  ("2026-10-10 17:30", "Bodo/Glimt", "Rosenborg")])
    odds = _odds([("2026-10-10 15:00", "Arsenal FC", "Chelsea FC", 1.9, 3.5, 4.0)])

    linked, unmatched = link_odds(card, odds)
    assert len(linked) == 2                       # nothing dropped
    assert list(linked["odds_matched"]) == [True, False]
    assert np.isnan(linked["max_home"].iloc[1])
    assert unmatched["home_team"].tolist() == ["Bodo/Glimt"]


def test_a_reverse_fixture_on_another_day_does_not_match():
    """Same two clubs, other leg — the same-day guard keeps them apart."""
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    odds = _odds([("2027-02-14 15:00", "Chelsea", "Arsenal", 2.4, 3.3, 2.9)])

    linked, _ = link_odds(card, odds)
    assert not bool(linked["odds_matched"].iloc[0])


def test_home_and_away_must_both_match():
    """A shared home team is not a fixture match."""
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    odds = _odds([("2026-10-10 15:00", "Arsenal", "Everton", 1.5, 4.0, 6.0)])

    linked, _ = link_odds(card, odds)
    assert not bool(linked["odds_matched"].iloc[0])


def test_the_best_candidate_wins_when_several_are_close():
    card = _card([("2026-10-10 15:00", "Real Madrid", "Real Sociedad")])
    odds = _odds([("2026-10-10 15:00", "Real Betis", "Real Sociedad", 3.0, 3.2, 2.4),
                  ("2026-10-10 15:00", "Real Madrid CF", "Real Sociedad", 1.6, 4.0, 5.0)])

    linked, _ = link_odds(card, odds)
    assert linked["max_home"].iloc[0] == pytest.approx(1.6)


def test_no_odds_leaves_the_card_intact():
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    linked, unmatched = link_odds(card, pd.DataFrame())
    assert len(linked) == 1
    assert not bool(linked["odds_matched"].iloc[0])
    assert len(unmatched) == 1


# ---------------------------------------------------------------------------
# Model against market
# ---------------------------------------------------------------------------

def test_implied_probability_inverts_the_price():
    assert implied_probability(pd.Series([2.0]))[0] == pytest.approx(0.5)
    assert np.isnan(implied_probability(pd.Series([1.0]))[0])     # no such price
    assert np.isnan(implied_probability(pd.Series([np.nan]))[0])


def test_edge_is_the_gap_between_model_and_price():
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    card["prob"] = 0.62
    odds = _odds([("2026-10-10 15:00", "Arsenal", "Chelsea", 2.0, 3.5, 4.0)])

    linked, _ = link_odds(card, odds)
    compared = add_market_comparison(linked)
    assert compared["market_prob"].iloc[0] == pytest.approx(0.5)
    assert compared["edge"].iloc[0] == pytest.approx(0.12)


def test_comparison_survives_a_card_with_no_prices():
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    linked, _ = link_odds(card, pd.DataFrame())
    compared = add_market_comparison(linked)
    assert compared["market_prob"].isna().all()
    assert compared["edge"].isna().all()


# ---------------------------------------------------------------------------
# Saying it out loud
# ---------------------------------------------------------------------------

def test_coverage_note_states_partial_coverage():
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea"),
                  ("2026-10-10 17:30", "Bodo/Glimt", "Rosenborg")])
    odds = _odds([("2026-10-10 15:00", "Arsenal FC", "Chelsea FC", 1.9, 3.5, 4.0)])

    note = coverage_note(link_odds(card, odds)[0])
    assert "1 of 2" in note and "unmatched" in note


def test_coverage_note_states_full_and_empty_coverage():
    card = _card([("2026-10-10 15:00", "Arsenal", "Chelsea")])
    odds = _odds([("2026-10-10 15:00", "Arsenal", "Chelsea", 1.9, 3.5, 4.0)])
    assert "all 1 fixtures" in coverage_note(link_odds(card, odds)[0])
    assert "No odds matched" in coverage_note(link_odds(card, pd.DataFrame())[0])
