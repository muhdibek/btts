"""
test_providers.py
-----------------
Tests for the multi-provider football API client.

Every provider here is unverified against its live host — the environment this
was written in cannot reach any of them — so the tests pin the two things that
can be verified offline: that each provider's AUTH STYLE puts the key where
that provider expects it, and that parsing degrades rather than raises when a
payload does not look the way the documentation promised.

A fake transport stands in for the network; nothing here makes a request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import providers as P                                        # noqa: E402
from data.providers import (APIFOOTBALL, FIVEDOLLAR, Provider,         # noqa: E402
                            ProviderError, best_prices, get_provider,
                            normalise_fixtures, normalise_odds, request)


class FakeResponse:
    def __init__(self, payload, text="", status=200):
        self._payload, self.text, self.status = payload, text, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture()
def capture(monkeypatch):
    """Record the request the client would have made."""
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params or {}, headers=headers or {})
        return FakeResponse(seen.get("payload", []))

    monkeypatch.setattr(P.requests, "get", fake_get)
    return seen


# ---------------------------------------------------------------------------
# Auth styles — the thing that differs most between providers
# ---------------------------------------------------------------------------

def test_bearer_provider_sends_an_authorization_header(capture, monkeypatch):
    monkeypatch.setenv("FIVEDOLLAR_API_KEY", "fb_test_123")
    request(FIVEDOLLAR, "fixtures")
    assert capture["headers"]["Authorization"] == "Bearer fb_test_123"
    assert "fb_test_123" not in str(capture["params"])      # not in the URL
    assert capture["url"].endswith("/v1/fixtures")


def test_query_key_provider_sends_the_key_as_a_parameter(capture, monkeypatch):
    monkeypatch.setenv("APIFOOTBALL_KEY", "af_test_456")
    request(APIFOOTBALL, "fixtures")
    assert capture["params"]["APIkey"] == "af_test_456"
    assert capture["params"]["action"] == "get_events"      # action style, not a path
    assert capture["headers"] == {}


def test_a_missing_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("FIVEDOLLAR_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="FIVEDOLLAR_API_KEY"):
        FIVEDOLLAR.key()


def test_an_unknown_endpoint_is_refused(monkeypatch):
    monkeypatch.setenv("FIVEDOLLAR_API_KEY", "x")
    with pytest.raises(ProviderError, match="no endpoint"):
        request(FIVEDOLLAR, "transfers")


def test_unknown_provider_lists_what_exists():
    with pytest.raises(ProviderError, match="apifootball"):
        get_provider("nope")


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------

def test_an_error_body_behind_http_200_is_caught(monkeypatch):
    monkeypatch.setenv("APIFOOTBALL_KEY", "k")
    monkeypatch.setattr(P.requests, "get", lambda *a, **k: FakeResponse(
        {"error": 404, "message": "No data found"}))
    with pytest.raises(ProviderError, match="No data found"):
        request(APIFOOTBALL, "fixtures")


def test_a_wrapped_list_is_unwrapped(monkeypatch):
    """REST providers usually wrap: {"data": [...]}."""
    monkeypatch.setenv("FIVEDOLLAR_API_KEY", "k")
    monkeypatch.setattr(P.requests, "get", lambda *a, **k: FakeResponse(
        {"data": [{"id": "1"}, {"id": "2"}], "meta": {"page": 1}}))
    assert request(FIVEDOLLAR, "fixtures") == [{"id": "1"}, {"id": "2"}]


def test_non_json_is_reported_with_the_key_redacted(monkeypatch):
    monkeypatch.setenv("APIFOOTBALL_KEY", "secret_key_value")
    monkeypatch.setattr(P.requests, "get", lambda *a, **k: FakeResponse(
        None, text="<html>secret_key_value rejected</html>"))
    with pytest.raises(ProviderError) as excinfo:
        request(APIFOOTBALL, "fixtures")
    assert "secret_key_value" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Mapping — tolerant by design, because the mappings are unverified
# ---------------------------------------------------------------------------

FIVEDOLLAR_FIXTURE = {
    "id": "88", "date": "2026-10-10", "time": "15:00", "league": "Premier League",
    "home_team": "Arsenal", "away_team": "Leeds", "home_score": None,
    "away_score": None, "status": "scheduled",
}


def test_fixtures_map_per_provider():
    out = normalise_fixtures(pd.DataFrame([FIVEDOLLAR_FIXTURE]), FIVEDOLLAR)
    assert out["home_team"].iloc[0] == "Arsenal"
    assert out["kickoff"].iloc[0] == pd.Timestamp("2026-10-10 15:00")
    assert not bool(out["played"].iloc[0])


def test_a_played_fixture_is_detected_from_its_scores():
    played = {**FIVEDOLLAR_FIXTURE, "home_score": "3", "away_score": "1"}
    out = normalise_fixtures(pd.DataFrame([played]), FIVEDOLLAR)
    assert bool(out["played"].iloc[0])
    assert out["fthg"].iloc[0] == 3


def test_a_renamed_field_costs_one_column_not_the_run():
    renamed = {k.replace("home_team", "homeTeam"): v for k, v in FIVEDOLLAR_FIXTURE.items()}
    out = normalise_fixtures(pd.DataFrame([renamed]), FIVEDOLLAR)
    assert out["home_team"].isna().all()
    assert out["away_team"].iloc[0] == "Leeds"


def test_odds_parse_and_best_price_wins():
    rows = [{"fixture_id": "88", "bookmaker": "A", "home": "2.00", "draw": "3.4",
             "away": "4.0", "over_2_5": "1.9", "under_2_5": "1.95",
             "btts_yes": "1.8", "btts_no": "2.0"},
            {"fixture_id": "88", "bookmaker": "B", "home": "2.15", "draw": "3.3",
             "away": "3.9", "over_2_5": "1.95", "under_2_5": "1.9",
             "btts_yes": "1.85", "btts_no": "1.95"}]
    odds = normalise_odds(pd.DataFrame(rows), FIVEDOLLAR)
    assert odds["odds_home"].iloc[0] == pytest.approx(2.00)

    best = best_prices(odds)
    assert len(best) == 1
    assert best["max_home"].iloc[0] == pytest.approx(2.15)   # B's price
    assert best["max_away"].iloc[0] == pytest.approx(4.00)   # A's


def test_a_market_the_book_does_not_price_stays_missing():
    row = {"fixture_id": "88", "bookmaker": "A", "home": "2.0", "draw": "3.4", "away": "4.0"}
    odds = normalise_odds(pd.DataFrame([row]), FIVEDOLLAR)
    assert odds["odds_btts_yes"].isna().all()


def test_empty_payloads_stay_empty():
    assert normalise_fixtures(pd.DataFrame(), FIVEDOLLAR).empty
    assert normalise_odds(pd.DataFrame(), APIFOOTBALL).empty
    assert best_prices(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Adding a provider should be a dict
# ---------------------------------------------------------------------------

def test_a_new_provider_needs_no_new_code(capture, monkeypatch):
    custom = Provider(
        name="custom", base_url="https://example.test/api/", key_env="CUSTOM_KEY",
        auth="header", auth_name="X-Api-Key", path_style="path",
        endpoints={"fixtures": "matches"},
        fixture_map={"uid": "match_id", "kickoff_date": "date", "h": "home_team",
                     "a": "away_team"},
    )
    monkeypatch.setenv("CUSTOM_KEY", "ck_1")
    request(custom, "fixtures")
    assert capture["headers"]["X-Api-Key"] == "ck_1"
    assert capture["url"] == "https://example.test/api/matches"

    out = normalise_fixtures(
        pd.DataFrame([{"uid": "7", "kickoff_date": "2026-10-10", "h": "X", "a": "Y"}]),
        custom)
    assert out["home_team"].iloc[0] == "X"


# ---------------------------------------------------------------------------
# The Odds API — a nested response no field map can express
# ---------------------------------------------------------------------------

ODDS_API_EVENT = {
    "id": "e912a", "sport_key": "soccer_epl",
    "commence_time": "2026-10-10T14:00:00Z",
    "home_team": "Arsenal", "away_team": "Leeds United",
    "bookmakers": [
        {"key": "pinnacle", "title": "Pinnacle", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Leeds United", "price": 4.20},     # away listed first
                {"name": "Arsenal", "price": 1.85},
                {"name": "Draw", "price": 3.60}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 2.05, "point": 3.5},   # wrong line
                {"name": "Over", "price": 1.95, "point": 2.5},
                {"name": "Under", "price": 1.90, "point": 2.5}]},
        ]},
        {"key": "bet365", "title": "Bet365", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 1.91},
                {"name": "Leeds United", "price": 4.00},
                {"name": "Draw", "price": 3.50}]},
        ]},
    ],
}


def test_odds_api_outcomes_are_read_by_team_not_by_position():
    """
    Outcomes are named by TEAM and arrive in any order, so taking the first
    entry as the home price is wrong — here the away team is listed first.
    """
    from data.providers import parse_theoddsapi
    rows = parse_theoddsapi([ODDS_API_EVENT])

    pinnacle = rows[rows["bookmaker"] == "Pinnacle"].iloc[0]
    assert pinnacle["odds_home"] == pytest.approx(1.85)
    assert pinnacle["odds_away"] == pytest.approx(4.20)
    assert pinnacle["odds_draw"] == pytest.approx(3.60)


def test_odds_api_keeps_only_the_two_and_a_half_line():
    from data.providers import parse_theoddsapi
    pinnacle = parse_theoddsapi([ODDS_API_EVENT]).iloc[0]
    assert pinnacle["odds_over25"] == pytest.approx(1.95)      # not the 3.5 line
    assert pinnacle["odds_under25"] == pytest.approx(1.90)


def test_odds_api_gives_one_row_per_bookmaker_and_best_price_picks_across_them():
    from data.providers import best_prices, parse_theoddsapi
    rows = parse_theoddsapi([ODDS_API_EVENT])
    assert len(rows) == 2

    best = best_prices(rows)
    assert best["max_home"].iloc[0] == pytest.approx(1.91)     # Bet365's
    assert best["max_away"].iloc[0] == pytest.approx(4.20)     # Pinnacle's


def test_odds_api_tolerates_missing_markets_and_empty_payloads():
    from data.providers import parse_theoddsapi
    bare = {"id": "x", "home_team": "A", "away_team": "B",
            "commence_time": "2026-10-10T14:00:00Z",
            "bookmakers": [{"title": "Book", "markets": []}]}
    row = parse_theoddsapi([bare]).iloc[0]
    assert row["bookmaker"] == "Book"
    assert "odds_home" not in row or pd.isna(row.get("odds_home"))
    assert parse_theoddsapi([]).empty


def test_odds_api_routes_through_its_parser_not_the_field_map():
    from data.providers import THEODDSAPI, normalise_odds
    out = normalise_odds([ODDS_API_EVENT], THEODDSAPI)
    assert out["odds_home"].iloc[0] == pytest.approx(1.85)


def test_sport_key_is_substituted_into_the_path(capture, monkeypatch):
    """The sport key in the URL decides the sport — NFL or the Eredivisie."""
    from data.providers import THEODDSAPI
    monkeypatch.setenv("ODDS_API_KEY", "oa_test")
    request(THEODDSAPI, "odds", sport="soccer_epl", regions="eu")
    assert capture["url"] == "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    assert capture["params"]["apiKey"] == "oa_test"
    assert capture["params"]["regions"] == "eu"
    assert "sport" not in capture["params"]          # consumed by the path


def test_a_missing_path_parameter_is_named(monkeypatch):
    from data.providers import THEODDSAPI
    monkeypatch.setenv("ODDS_API_KEY", "oa_test")
    with pytest.raises(ProviderError, match="missing path parameter 'sport'"):
        request(THEODDSAPI, "odds")


def test_soccer_sport_keys_are_listed_for_the_modelled_leagues():
    from data.providers import ODDS_API_SPORTS
    assert ODDS_API_SPORTS["Premier League"] == "soccer_epl"
    assert all(key.startswith("soccer_") for key in ODDS_API_SPORTS.values())
