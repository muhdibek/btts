"""
test_apifootball.py
-------------------
Tests for the APIFootball client.

The client was written where the API host is unreachable, so these tests pin
the parsing against payloads shaped like the documented API — and, just as
importantly, against payloads that are NOT, since a field rename upstream
should cost one column rather than the whole run.

No network: every payload here is inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.apifootball import (ApiFootballError, api_key, best_prices,     # noqa: E402
                              normalise_fixtures, normalise_odds)


EVENT = {
    "match_id": "1234", "country_name": "England", "league_name": "Premier League",
    "match_date": "2026-10-10", "match_time": "15:00", "match_status": "",
    "match_hometeam_name": "Arsenal", "match_awayteam_name": "Leeds",
    "match_hometeam_score": "", "match_awayteam_score": "",
}
PLAYED = {**EVENT, "match_id": "1235", "match_status": "Finished",
          "match_hometeam_score": "2", "match_awayteam_score": "1"}

ODDS_ROW = {
    "match_id": "1234", "odd_bookmakers": "Bet365",
    "odd_1": "1.85", "odd_x": "3.60", "odd_2": "4.20",
    "o_over_25": "1.95", "o_under_25": "1.90",
    "bts_yes": "1.80", "bts_no": "1.95",
}


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    with pytest.raises(ApiFootballError, match="No API key"):
        api_key()


def test_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("APIFOOTBALL_KEY", "abc123")
    assert api_key() == "abc123"
    assert api_key("explicit") == "explicit"      # an explicit key still wins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def test_fixtures_map_into_the_repo_schema():
    out = normalise_fixtures(pd.DataFrame([EVENT, PLAYED]))
    assert list(out["home_team"]) == ["Arsenal", "Arsenal"]
    assert out["kickoff"].iloc[0] == pd.Timestamp("2026-10-10 15:00")
    assert out["league"].iloc[0] == "Premier League"


def test_an_unplayed_fixture_has_no_score():
    out = normalise_fixtures(pd.DataFrame([EVENT]))
    assert out["played"].iloc[0] == False          # noqa: E712 — numpy bool
    assert np.isnan(out["fthg"].iloc[0])


def test_a_played_match_is_marked_and_scored():
    out = normalise_fixtures(pd.DataFrame([PLAYED]))
    assert bool(out["played"].iloc[0])
    assert out["fthg"].iloc[0] == 2 and out["ftag"].iloc[0] == 1


def test_a_renamed_field_costs_one_column_not_the_run():
    """
    The client is unverified against the live API, so an unexpected payload
    must degrade rather than raise — one NaN column beats a stack trace.
    """
    renamed = {k.replace("match_hometeam_name", "home_name"): v for k, v in EVENT.items()}
    out = normalise_fixtures(pd.DataFrame([renamed]))
    assert out["home_team"].isna().all()
    assert out["away_team"].iloc[0] == "Leeds"     # the rest still parses


def test_empty_payloads_return_empty_frames():
    assert normalise_fixtures(pd.DataFrame()).empty
    assert normalise_odds(pd.DataFrame()).empty
    assert best_prices(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def test_odds_are_parsed_as_numbers():
    out = normalise_odds(pd.DataFrame([ODDS_ROW]))
    assert out["odds_home"].iloc[0] == pytest.approx(1.85)
    assert out["odds_btts_yes"].iloc[0] == pytest.approx(1.80)
    assert out["bookmaker"].iloc[0] == "Bet365"


def test_best_price_takes_the_maximum_across_bookmakers():
    rows = [ODDS_ROW,
            {**ODDS_ROW, "odd_bookmakers": "Pinnacle", "odd_1": "1.92", "odd_2": "4.10"}]
    best = best_prices(normalise_odds(pd.DataFrame(rows)))
    assert len(best) == 1
    assert best["max_home"].iloc[0] == pytest.approx(1.92)   # Pinnacle's price
    assert best["max_away"].iloc[0] == pytest.approx(4.20)   # Bet365's
    assert best["match_id"].iloc[0] == "1234"


def test_unpriced_markets_stay_missing():
    """A bookmaker that does not price BTTS must not become a zero."""
    row = {k: v for k, v in ODDS_ROW.items() if not k.startswith("bts")}
    out = normalise_odds(pd.DataFrame([row]))
    assert out["odds_btts_yes"].isna().all()
    assert out["odds_home"].iloc[0] == pytest.approx(1.85)
