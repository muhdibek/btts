"""
test_live_fixtures.py
---------------------
Tests for the live fixture feed.

The important ones cover the feed's two defects, because both fail silently
and both corrupt exactly the statistic the dashboard reports:

  - a minority of matches carry a bare-list score that is ALWAYS [0, 0].
    Counted as results, they invent goalless draws and drag every BTTS rate
    down. They must be treated as unknown.
  - some leagues publish no kickoff time. Defaulting to midnight would display
    "00:00" as though it were real.

Everything here runs against inline payloads — no network.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.live_fixtures import (_full_time_score, available_dates,     # noqa: E402
                                build_fixture_card, build_team_stats,
                                fair_odds, malformed_score_count, next_matchday)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a league table like load_league() returns."""
    df = pd.DataFrame(rows)
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["date"]    = df["kickoff"].dt.strftime("%Y-%m-%d")
    for column, default in (("league", "Test League"), ("code", "tl.1"),
                            ("season", "2026-27"), ("round", "Matchday 1"),
                            ("time_known", True), ("score_unknown", False)):
        if column not in df.columns:
            df[column] = default
    return df.sort_values("kickoff").reset_index(drop=True)


def _played(kickoff, home, away, hg, ag, **kw):
    return {"kickoff": kickoff, "home_team": home, "away_team": away,
            "fthg": hg, "ftag": ag, "played": True, **kw}


def _fixture(kickoff, home, away, **kw):
    return {"kickoff": kickoff, "home_team": home, "away_team": away,
            "fthg": np.nan, "ftag": np.nan, "played": False, **kw}


# ---------------------------------------------------------------------------
# Score parsing — the feed defect
# ---------------------------------------------------------------------------

def test_dict_score_is_a_real_result():
    goals, malformed = _full_time_score({"ht": [1, 0], "ft": [2, 1]})
    assert goals == [2, 1]
    assert malformed is False


def test_genuine_goalless_draw_in_dict_form_is_kept():
    """A real 0-0 arrives as a dict and must count as a played match."""
    goals, malformed = _full_time_score({"ht": [0, 0], "ft": [0, 0]})
    assert goals == [0, 0]
    assert malformed is False


def test_bare_list_score_is_treated_as_unknown():
    """
    The feed's list-form score is always [0, 0] — a placeholder, not a result.
    Taking it at face value would invent a goalless draw.
    """
    goals, malformed = _full_time_score([0, 0])
    assert goals == []
    assert malformed is True


def test_missing_score_is_an_unplayed_fixture():
    goals, malformed = _full_time_score(None)
    assert goals == []
    assert malformed is False


def test_malformed_scores_are_counted_not_hidden():
    df = _frame([
        _played("2026-09-01 15:00", "A", "B", 2, 1),
        _fixture("2026-09-08 15:00", "C", "D", score_unknown=True),
    ])
    assert malformed_score_count(df) == 1


# ---------------------------------------------------------------------------
# Team statistics
# ---------------------------------------------------------------------------

def test_team_stats_come_from_real_results():
    df = _frame([
        _played("2026-09-01 15:00", "Alpha", "Bravo", 2, 1),   # both scored
        _played("2026-09-08 15:00", "Alpha", "Charlie", 1, 0),  # clean sheet
    ])
    stats = build_team_stats(df)
    assert stats["Alpha"]["avg_scored"]   == pytest.approx(1.5)
    assert stats["Alpha"]["avg_conceded"] == pytest.approx(0.5)
    assert stats["Alpha"]["btts_rate"]    == pytest.approx(0.5)
    assert stats["Alpha"]["matches_used"] == 2


def test_team_stats_respect_the_before_cutoff():
    """A fixture's statistics must never include its own result."""
    df = _frame([
        _played("2026-09-01 15:00", "Alpha", "Bravo", 3, 3),
        _played("2026-09-08 15:00", "Alpha", "Bravo", 0, 0),
    ])
    stats = build_team_stats(df, before=pd.Timestamp("2026-09-08 15:00"))
    assert stats["Alpha"]["matches_used"] == 1
    assert stats["Alpha"]["avg_scored"] == pytest.approx(3.0)


def test_team_stats_window_keeps_only_recent_matches():
    rows = [_played(f"2026-0{1 + i // 28}-{1 + i % 28:02d} 15:00", "Alpha", "Bravo", i % 3, 1)
            for i in range(25)]
    stats = build_team_stats(_frame(rows), window=10)
    assert stats["Alpha"]["matches_used"] == 10


def test_card_falls_back_for_a_team_with_no_history():
    """A promoted side with no completed matches gets neutral numbers, not a crash."""
    df = _frame([
        _played("2026-09-01 15:00", "Alpha", "Bravo", 2, 1),
        _fixture("2026-09-08 15:00", "Alpha", "Newcomer"),
    ])
    card = build_fixture_card(df, on_date=date(2026, 9, 8))
    assert len(card) == 1
    assert card.iloc[0]["away_matches_used"] == 0
    assert card.iloc[0]["away_btts_rate"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Matchday selection
# ---------------------------------------------------------------------------

def test_next_matchday_skips_empty_days():
    """Football has gaps; a dashboard pinned to 'today' would show nothing."""
    df = _frame([
        _played("2026-09-20 15:00", "Alpha", "Bravo", 1, 1),
        _fixture("2026-10-09 20:00", "Alpha", "Charlie"),
    ])
    assert next_matchday(df, today=date(2026, 9, 23)) == date(2026, 10, 9)


def test_available_dates_can_include_the_past():
    df = _frame([
        _played("2026-09-20 15:00", "Alpha", "Bravo", 1, 1),
        _fixture("2026-10-09 20:00", "Alpha", "Charlie"),
    ])
    assert available_dates(df, upcoming_only=False) == [date(2026, 9, 20), date(2026, 10, 9)]
    assert available_dates(df, today=date(2026, 9, 23)) == [date(2026, 10, 9)]


def test_card_excludes_matches_already_played():
    df = _frame([
        _played("2026-09-08 13:00", "Alpha", "Bravo", 2, 2),
        _fixture("2026-09-08 15:00", "Charlie", "Delta"),
    ])
    card = build_fixture_card(df, on_date=date(2026, 9, 8))
    assert len(card) == 1
    assert card.iloc[0]["home_team"] == "Charlie"


def test_missing_kickoff_time_is_shown_as_unknown():
    df = _frame([
        _played("2026-09-01 15:00", "Alpha", "Bravo", 1, 1),
        _fixture("2026-09-08 00:00", "Alpha", "Charlie", time_known=False),
    ])
    card = build_fixture_card(df, on_date=date(2026, 9, 8))
    assert card.iloc[0]["kickoff"].endswith("--:--")


def test_card_carries_no_bookmaker_odds():
    """The feed has no prices; the app fills fair odds and says so."""
    df = _frame([
        _played("2026-09-01 15:00", "Alpha", "Bravo", 1, 1),
        _fixture("2026-09-08 15:00", "Alpha", "Charlie"),
    ])
    card = build_fixture_card(df, on_date=date(2026, 9, 8))
    for column in ("btts_odds", "home_win_odds", "draw_odds", "away_win_odds"):
        assert card[column].isna().all(), column


def test_empty_inputs_return_an_empty_card():
    assert build_fixture_card(pd.DataFrame()).empty
    assert next_matchday(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# Fair odds
# ---------------------------------------------------------------------------

def test_fair_odds_are_the_break_even_price():
    assert fair_odds(0.50) == pytest.approx(2.00)
    assert fair_odds(0.625) == pytest.approx(1.60)


def test_fair_odds_reject_impossible_probabilities():
    assert np.isnan(fair_odds(0.0))
    assert np.isnan(fair_odds(np.nan))
    assert fair_odds(1.0) == pytest.approx(1.01)      # floored, never below evens
