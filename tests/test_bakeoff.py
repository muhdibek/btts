"""
test_bakeoff.py
---------------
Tests for the model bake-off.

Two kinds here. The first check the mechanics — market derivation, scoring
rules, the date split. The second are the ones that make the harness worth
trusting: data is generated from a KNOWN process, and the model matching that
process has to win. A harness that cannot recover the truth when the truth is
known tells you nothing when it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bakeoff import run_bakeoff, split_by_date                  # noqa: E402
from models.evaluate import (betting_roi, brier_score,                 # noqa: E402
                             calibration_table, expected_calibration_error,
                             log_loss, roc_auc, skill_score)
from models.goal_models import (build_model, markets_from_grid)        # noqa: E402
from models.synthetic import simulate_league, summarise_dispersion     # noqa: E402


# ---------------------------------------------------------------------------
# Market derivation
# ---------------------------------------------------------------------------

def test_markets_from_a_known_grid():
    """A grid with all its mass on 1-1 is BTTS, a draw, and under 2.5."""
    grid = np.zeros((4, 4))
    grid[1, 1] = 1.0
    m = markets_from_grid(grid)
    assert m["p_btts"]   == pytest.approx(1.0)
    assert m["p_draw"]   == pytest.approx(1.0)
    assert m["p_over25"] == pytest.approx(0.0)
    assert m["p_home"]   == pytest.approx(0.0)


def test_markets_are_coherent():
    """1X2 must sum to 1, and a 0-x column can never be BTTS."""
    grid = np.random.default_rng(3).random((6, 6))
    m = markets_from_grid(grid)
    assert m["p_home"] + m["p_draw"] + m["p_away"] == pytest.approx(1.0)
    assert 0.0 <= m["p_btts"] <= 1.0

    no_home_goals = np.zeros((4, 4))
    no_home_goals[0, :] = 0.25
    assert markets_from_grid(no_home_goals)["p_btts"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def test_log_loss_rewards_being_right():
    outcomes = np.array([1, 1, 0, 0])
    confident_right = np.array([0.99, 0.99, 0.01, 0.01])
    confident_wrong = np.array([0.01, 0.01, 0.99, 0.99])
    coin_flip       = np.full(4, 0.5)

    assert log_loss(confident_right, outcomes) < log_loss(coin_flip, outcomes)
    assert log_loss(coin_flip, outcomes) < log_loss(confident_wrong, outcomes)
    assert log_loss(coin_flip, outcomes) == pytest.approx(np.log(2), abs=1e-9)


def test_brier_matches_squared_error():
    outcomes = np.array([1, 0])
    assert brier_score(np.array([0.75, 0.25]), outcomes) == pytest.approx(0.0625)


def test_skill_score_is_zero_against_itself():
    outcomes = np.array([1, 0, 1, 1])
    probs    = np.array([0.7, 0.4, 0.6, 0.8])
    assert skill_score(probs, outcomes, probs) == pytest.approx(0.0)


def test_metrics_ignore_unpriced_rows():
    """A model that cannot price a market must not be scored on guesses."""
    outcomes = np.array([1, 0, 1])
    probs    = np.array([0.8, np.nan, 0.6])
    assert np.isfinite(log_loss(probs, outcomes))
    assert np.isnan(log_loss(np.full(3, np.nan), outcomes))


def test_calibration_table_finds_a_biased_model():
    rng = np.random.default_rng(1)
    outcomes = rng.binomial(1, 0.5, 2000)
    overconfident = np.where(outcomes == 1, 0.9, 0.9)   # always says 90%
    table = calibration_table(overconfident, outcomes)
    assert table["n"].sum() == 2000
    assert table["gap"].iloc[0] == pytest.approx(0.4, abs=0.05)
    assert expected_calibration_error(overconfident, outcomes) == pytest.approx(0.4, abs=0.05)


def test_auc_measures_ranking_not_scaling():
    outcomes = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), outcomes) == pytest.approx(1.0)
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), outcomes) == pytest.approx(0.0)

    # Ranking is scale-free: squashing the probabilities changes log loss but
    # not the order, so AUC is unmoved.
    assert roc_auc(np.array([0.49, 0.50, 0.51, 0.52]), outcomes) == pytest.approx(1.0)


def test_auc_of_a_constant_prediction_is_a_coin_flip():
    """Tied predictions must average to 0.5, not land on an arbitrary value."""
    assert roc_auc(np.full(4, 0.5), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_auc_needs_both_outcomes():
    assert np.isnan(roc_auc(np.array([0.3, 0.6]), np.array([1, 1])))


def test_betting_roi_only_bets_on_an_edge():
    probs    = np.array([0.60, 0.40])
    odds     = np.array([2.00, 2.00])     # implied 50%
    outcomes = np.array([1, 0])

    result = betting_roi(probs, odds, outcomes)
    assert result["bets"] == 1                        # only the 60% pick clears
    assert result["profit"] == pytest.approx(1.0)
    assert result["roi"] == pytest.approx(1.0)

    assert betting_roi(probs, odds, outcomes, edge=0.5)["bets"] == 0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_split_is_chronological():
    matches = simulate_league(n_teams=10, n_seasons=1, seed=2)
    train, test = split_by_date(matches, 0.7)
    assert len(train) + len(test) == len(matches)
    assert train["kickoff"].max() <= test["kickoff"].min()


# ---------------------------------------------------------------------------
# Recovering a known process — the tests that matter
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def poisson_league():
    return simulate_league(n_teams=18, n_seasons=3, dgp="poisson", seed=5)


@pytest.fixture(scope="module")
def overdispersed_league():
    return simulate_league(n_teams=18, n_seasons=3, dgp="negative_binomial",
                           dispersion=2.5, seed=5)


@pytest.fixture(scope="module")
def correlated_league():
    return simulate_league(n_teams=18, n_seasons=6, dgp="dixon_coles",
                           rho=-0.30, seed=5)


def test_generators_produce_the_dispersion_they_claim(poisson_league, overdispersed_league):
    """Overdispersed data must actually be more variable than Poisson data."""
    assert (summarise_dispersion(overdispersed_league)["variance_to_mean"] >
            summarise_dispersion(poisson_league)["variance_to_mean"])


def test_poisson_predictions_track_the_true_probabilities(poisson_league):
    train, test = split_by_date(poisson_league, 0.75)
    model = build_model("poisson").fit(train)
    predicted = model.predict_frame(test)["p_btts"]
    assert predicted.corr(test["true_p_btts"]) > 0.85


def test_negative_binomial_collapses_to_poisson_on_poisson_data(poisson_league):
    """With no overdispersion to model, r should be driven to its ceiling."""
    train, _ = split_by_date(poisson_league, 0.75)
    model = build_model("negative_binomial").fit(train)
    assert model.fit_result.extra["r"] > 50.0


def test_negative_binomial_recovers_overdispersion(overdispersed_league):
    train, _ = split_by_date(overdispersed_league, 0.75)
    model = build_model("negative_binomial").fit(train)
    assert 1.0 < model.fit_result.extra["r"] < 10.0     # true value was 2.5


def test_negative_binomial_wins_on_overdispersed_data(overdispersed_league):
    card, _ = run_bakeoff(overdispersed_league, market="p_btts",
                          model_names=["poisson", "negative_binomial"])
    scores = card.set_index("model")["log_loss"]
    assert scores["negative_binomial"] < scores["poisson"]


def test_dixon_coles_recovers_rho(correlated_league):
    train, _ = split_by_date(correlated_league, 0.75)
    model = build_model("dixon_coles").fit(train)
    assert model.fit_result.extra["rho"] == pytest.approx(-0.30, abs=0.10)


def test_dixon_coles_wins_when_low_scores_are_correlated(correlated_league):
    card, _ = run_bakeoff(correlated_league, market="p_btts",
                          model_names=["poisson", "dixon_coles"])
    scores = card.set_index("model")["log_loss"]
    assert scores["dixon_coles"] < scores["poisson"]


def test_every_model_beats_the_base_rate_on_learnable_data(poisson_league):
    card, _ = run_bakeoff(poisson_league, market="p_btts")
    scored = card[card["log_loss"].notna()].set_index("model")
    for name in ("poisson", "dixon_coles", "negative_binomial", "team_rate"):
        assert scored.loc[name, "log_loss"] < scored.loc["base_rate", "log_loss"], name


def test_no_model_beats_the_oracle_by_much(poisson_league):
    """
    The generating process is the ceiling. A model scoring clearly below it
    means the harness is leaking, not that the model is clairvoyant.
    """
    card, _ = run_bakeoff(poisson_league, market="p_btts")
    scored = card[card["log_loss"].notna()].set_index("model")
    oracle = scored.loc["ORACLE (true process)", "log_loss"]
    best   = scored["log_loss"].min()
    assert best >= oracle - 0.02          # a margin for test-set sampling noise


# ---------------------------------------------------------------------------
# Skellam's honest limitation
# ---------------------------------------------------------------------------

def test_skellam_refuses_to_price_btts(poisson_league):
    """
    A goal-difference model cannot know whether both sides scored: 0-0 and 1-1
    are the same difference. It must report NaN, not a guess.
    """
    train, test = split_by_date(poisson_league, 0.75)
    model = build_model("skellam").fit(train)
    predictions = model.predict_frame(test)
    assert predictions["p_btts"].isna().all()
    assert predictions["p_over25"].isna().all()


def test_skellam_prices_1x2_like_the_poisson_grid(poisson_league):
    """Same rates, so the difference distribution must agree with the grid."""
    train, _ = split_by_date(poisson_league, 0.75)
    skellam = build_model("skellam").fit(train)
    poisson = build_model("poisson").fit(train)

    home, away = train["home_team"].iloc[0], train["away_team"].iloc[0]
    sk = skellam.predict_markets(home, away)
    po = poisson.predict_markets(home, away)

    assert sk["p_home"] + sk["p_draw"] + sk["p_away"] == pytest.approx(1.0)
    assert sk["p_home"] == pytest.approx(po["p_home"], abs=0.01)
    assert sk["p_draw"] == pytest.approx(po["p_draw"], abs=0.01)


def test_bakeoff_reports_skellam_as_unscored(poisson_league):
    card, _ = run_bakeoff(poisson_league, market="p_btts")
    row = card.set_index("model").loc["skellam"]
    assert row["n"] == 0
    assert np.isnan(row["log_loss"])


# ---------------------------------------------------------------------------
# Daily picks and slips
# ---------------------------------------------------------------------------

def _card(probs: list[float]) -> pd.DataFrame:
    """A minimal priced card, most confident first."""
    return pd.DataFrame({
        "kickoff":    [f"2026-10-18 1{i}:00" for i in range(len(probs))],
        "league":     ["Test League"] * len(probs),
        "home_team":  [f"Home{i}" for i in range(len(probs))],
        "away_team":  [f"Away{i}" for i in range(len(probs))],
        "btts_prob":  probs,
        "selection_label": [f"Home{i} to win" for i in range(len(probs))],
    })


def test_picks_are_ranked_by_confidence():
    from models.daily_picks import rank_picks
    picks = rank_picks(_card([0.4, 0.8, 0.6]), top_n=2)
    assert list(picks["btts_prob"]) == [0.8, 0.6]
    assert picks["fair_odds"].iloc[0] == pytest.approx(1.25)


def test_picks_skip_fixtures_that_could_not_be_priced():
    from models.daily_picks import rank_picks
    card = _card([0.7, np.nan, 0.5])
    picks = rank_picks(card, top_n=5)
    assert len(picks) == 2
    assert picks["btts_prob"].notna().all()


def test_confidence_bands_are_ordered():
    from models.daily_picks import confidence_band
    assert confidence_band(0.75)[0].endswith("Strong")
    assert confidence_band(0.65)[0].endswith("Solid")
    assert confidence_band(0.30)[0].endswith("Coin flip")
    assert confidence_band(np.nan)[0] == "—"


def test_daily_slips_come_in_three_shapes():
    from models.daily_picks import build_daily_slips
    slips = build_daily_slips(_card([0.8, 0.75, 0.7, 0.65, 0.6, 0.55]),
                              label_column="selection_label")
    assert [s["legs"] for s in slips] == [2, 3, 5]
    assert [s["name"] for s in slips] == ["Banker double", "Balanced treble", "Long shot"]


def test_slip_probability_is_the_product_of_its_legs():
    from models.daily_picks import build_daily_slips
    slips = build_daily_slips(_card([0.8, 0.5]), label_column="selection_label")
    double = slips[0]
    assert double["combined_prob"] == pytest.approx(40.0)
    assert double["total_fair_odds"] == pytest.approx(2.5)


def test_more_legs_means_lower_probability():
    from models.daily_picks import build_daily_slips
    slips = build_daily_slips(_card([0.8, 0.75, 0.7, 0.65, 0.6, 0.55]),
                              label_column="selection_label")
    probabilities = [s["combined_prob"] for s in slips]
    assert probabilities == sorted(probabilities, reverse=True)


def test_a_card_too_small_for_a_slip_returns_nothing():
    from models.daily_picks import build_daily_slips
    assert build_daily_slips(_card([0.8])) == []
    assert build_daily_slips(pd.DataFrame()) == []
