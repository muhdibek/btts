"""
test_dataset.py
---------------
Tests for the BTTS dataset pipeline.

The important ones are the leakage tests. A feature that has seen the result
it is supposed to predict makes a model look excellent offline and lose money
in production, and it is invisible in the accuracy numbers — so the invariant
("every feature comes from strictly earlier matches") is asserted directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.features import (build_features, elo_expected,                # noqa: E402
                           implied_probabilities)
from data.football_data import parse_season_csv, season_code            # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,B365H,B365D,B365A,B365>2.5,B365<2.5
E0,01/08/2024,15:00,Alpha,Bravo,2,1,H,1,0,H,14,9,6,3,7,4,2.10,3.40,3.50,1.80,2.00
E0,08/08/2024,17:30,Charlie,Delta,1,0,H,0,0,D,11,12,4,5,5,6,2.50,3.30,2.80,2.05,1.75
E0,15/08/24,15:00,Alpha,Charlie,0,0,D,0,0,D,8,10,2,3,4,5,1.90,3.50,4.00,2.20,1.65
E0,22/08/2024,15:00,Bravo,Delta,3,2,H,2,1,H,16,13,8,6,8,5,1.75,3.80,4.20,1.60,2.30
E0,29/08/2024,20:00,Alpha,Bravo,1,1,D,0,1,A,12,11,5,4,6,6,2.00,3.45,3.70,1.85,1.95
E0,05/09/2024,15:00,Charlie,Alpha,,,,,,,,,,,,,,,,,
"""


@pytest.fixture()
def season_csv(tmp_path: Path) -> Path:
    path = tmp_path / "E0_2425.csv"
    path.write_text(RAW_CSV)
    return path


@pytest.fixture()
def matches(season_csv: Path) -> pd.DataFrame:
    return parse_season_csv(season_csv, start_year=2024)


@pytest.fixture()
def dataset(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_season_code():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"
    assert season_code(2009) == "0910"


def test_parse_drops_unplayed_fixtures(matches: pd.DataFrame):
    """The trailing row has no score — it is a fixture, not a result."""
    assert len(matches) == 5
    assert matches["fthg"].notna().all()


def test_parse_handles_both_date_formats(matches: pd.DataFrame):
    """dd/mm/yyyy and dd/mm/yy both appear, sometimes in one file."""
    assert matches["kickoff"].is_monotonic_increasing
    assert str(matches["kickoff"].iloc[2].date()) == "2024-08-15"
    assert matches["kickoff"].iloc[0].hour == 15


def test_btts_label(matches: pd.DataFrame):
    assert matches["btts"].tolist() == [1, 0, 0, 1, 1]
    assert matches["total_goals"].tolist() == [3, 1, 0, 5, 2]


# ---------------------------------------------------------------------------
# Leakage — the tests that matter
# ---------------------------------------------------------------------------

def test_first_appearance_has_no_history(dataset: pd.DataFrame):
    first = dataset.iloc[0]
    assert first["home_matches_played"] == 0
    assert np.isnan(first["home_btts_rate_l5"])
    assert np.isnan(first["away_gf_avg_l10"])
    assert np.isnan(first["home_rest_days"])
    assert first["h2h_matches"] == 0


def test_features_use_only_earlier_matches(dataset: pd.DataFrame):
    """
    For every row, each side's match count must equal the number of earlier
    matches that team played. If the current match leaked in, the count is
    one too high.
    """
    for i, row in dataset.iterrows():
        earlier = dataset.iloc[:i]
        for side in ("home", "away"):
            team = row[f"{side}_team"]
            expected = ((earlier["home_team"] == team) |
                        (earlier["away_team"] == team)).sum()
            assert row[f"{side}_matches_played"] == expected, (
                f"row {i} ({team}) saw {row[f'{side}_matches_played']} prior "
                f"matches, should be {expected}"
            )


def test_form_values_are_computed_from_prior_results(dataset: pd.DataFrame):
    """
    Row 5 is Alpha vs Bravo. Alpha has played 2-1 (win) and 0-0 before it,
    so its last-5 form must summarise exactly those two matches.
    """
    row = dataset.iloc[4]
    assert row["home_gf_avg_l5"]    == pytest.approx(1.0)    # (2 + 0) / 2
    assert row["home_ga_avg_l5"]    == pytest.approx(0.5)    # (1 + 0) / 2
    assert row["home_btts_rate_l5"] == pytest.approx(0.5)    # one of two
    assert row["home_fts_rate_l5"]  == pytest.approx(0.5)    # blanked once
    assert row["home_cs_rate_l5"]   == pytest.approx(0.5)    # clean sheet once
    assert row["home_rest_days"]    == 14                    # 15 Aug → 29 Aug


def test_venue_splits_use_the_right_venue(dataset: pd.DataFrame):
    """
    Same row: Alpha's home matches are 2-1 and 0-0; Bravo's away matches are
    just the 1-2 at Alpha (its other match was at home).
    """
    row = dataset.iloc[4]
    assert row["home_gf_avg_venue"]    == pytest.approx(1.0)
    assert row["home_btts_rate_venue"] == pytest.approx(0.5)
    assert row["away_gf_avg_venue"]    == pytest.approx(1.0)   # scored 1 at Alpha
    assert row["away_ga_avg_venue"]    == pytest.approx(2.0)


def test_head_to_head_excludes_the_current_match(dataset: pd.DataFrame):
    """Alpha vs Bravo met once before (2-1, both scored, 3 goals)."""
    row = dataset.iloc[4]
    assert row["h2h_matches"]   == 1
    assert row["h2h_btts_rate"] == pytest.approx(1.0)
    assert row["h2h_avg_goals"] == pytest.approx(3.0)


def test_row_order_does_not_change_features(matches: pd.DataFrame):
    """Shuffled input must produce the same table — the builder sorts by kickoff."""
    shuffled = matches.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = build_features(matches)
    b = build_features(shuffled)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

def _frame(rows: list[tuple[str, str, str, int, int]]) -> pd.DataFrame:
    """Minimal match table: (date, home, away, home goals, away goals)."""
    df = pd.DataFrame(rows, columns=["kickoff", "home_team", "away_team", "fthg", "ftag"])
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["btts"] = ((df["fthg"] > 0) & (df["ftag"] > 0)).astype(int)
    return df


def test_elo_starts_level(dataset: pd.DataFrame):
    assert dataset.iloc[0]["home_elo"] == pytest.approx(1500.0)
    assert dataset.iloc[0]["away_elo"] == pytest.approx(1500.0)
    assert dataset.iloc[0]["elo_diff"] == pytest.approx(0.0)


def test_elo_update_is_zero_sum():
    """What one side gains the other loses, so a pair's total is conserved."""
    built = build_features(_frame([
        ("2024-08-01", "Alpha", "Bravo", 3, 0),
        ("2024-08-08", "Alpha", "Bravo", 1, 1),
    ]))
    second = built.iloc[1]
    assert second["home_elo"] + second["away_elo"] == pytest.approx(3000.0)
    assert second["home_elo"] > 1500.0            # Alpha won the first meeting


def test_elo_ratings_follow_the_documented_update(dataset: pd.DataFrame):
    """
    Pre-match ratings must be the state BEFORE the row's own result. Alpha
    won 2-1 in match 1 and drew 0-0 in match 3, so its rating going into
    match 5 is the result of exactly those two updates — no more.
    """
    k, ha = 20.0, 65.0

    expected_m1 = elo_expected(1500.0, 1500.0, ha)
    alpha_after_m1 = 1500.0 + k * (1.0 - expected_m1)            # won, 1 goal margin

    # match 3 is Alpha (home) v Charlie, who beat Delta by the same margin
    charlie_after_m2 = 1500.0 + k * (1.0 - expected_m1)
    assert dataset.iloc[2]["home_elo"] == pytest.approx(alpha_after_m1)
    assert dataset.iloc[2]["away_elo"] == pytest.approx(charlie_after_m2)

    expected_m3 = elo_expected(alpha_after_m1, charlie_after_m2, ha)
    alpha_after_m3 = alpha_after_m1 + k * (0.5 - expected_m3)    # drew
    assert dataset.iloc[4]["home_elo"] == pytest.approx(alpha_after_m3)


def test_elo_margin_multiplier_scales_the_update():
    """A rout moves ratings further than a one-goal win."""
    narrow = build_features(_frame([("2024-08-01", "Alpha", "Bravo", 1, 0),
                                    ("2024-08-08", "Alpha", "Bravo", 0, 0)]))
    rout   = build_features(_frame([("2024-08-01", "Alpha", "Bravo", 5, 0),
                                    ("2024-08-08", "Alpha", "Bravo", 0, 0)]))
    assert rout.iloc[1]["elo_diff"] > narrow.iloc[1]["elo_diff"]


def test_elo_rewards_the_winner(dataset: pd.DataFrame):
    """Alpha beat Bravo first up, so Alpha is rated above Bravo next time out."""
    row = dataset.iloc[4]                 # Alpha vs Bravo again
    assert row["home_elo"] > row["away_elo"]
    assert row["elo_diff"] > 0


def test_elo_expected_score():
    assert elo_expected(1500, 1500) == pytest.approx(0.5)
    assert elo_expected(1500, 1500, home_advantage=65) > 0.5
    assert elo_expected(1900, 1500) > 0.9


# ---------------------------------------------------------------------------
# Market prices
# ---------------------------------------------------------------------------

def test_implied_probabilities_remove_the_overround():
    probs = implied_probabilities(2.10, 3.40, 3.50)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2]            # shorter price, likelier outcome


def test_implied_probabilities_reject_missing_prices():
    assert all(np.isnan(p) for p in implied_probabilities(2.0, np.nan, 3.0))
    assert all(np.isnan(p) for p in implied_probabilities(1.0, 2.0))


def test_market_features_present(dataset: pd.DataFrame):
    row = dataset.iloc[0]
    assert row["mkt_p_home"] + row["mkt_p_draw"] + row["mkt_p_away"] == pytest.approx(1.0)
    assert 0.0 < row["mkt_p_over25"] < 1.0


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_dataset_keeps_every_match_and_the_label(matches, dataset):
    assert len(dataset) == len(matches)
    assert dataset["btts"].tolist() == matches["btts"].tolist()
    assert dataset["kickoff"].is_monotonic_increasing
