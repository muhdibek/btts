"""
evaluate.py
-----------
Scoring for probability forecasts.

Accuracy is the wrong yardstick here. A model that says 55% for every match in
a league where 55% of matches see both teams score is "right" most of the time
and completely useless — it cannot tell one match from another. Proper scoring
rules punish that, so the bake-off is scored on:

  log loss   the standard for probability forecasts; punishes confident errors
             hard, which is the failure mode that empties a bankroll
  Brier      squared error on the probability; gentler on confident misses,
             useful as a second opinion
  skill      1 − loss/baseline_loss. Positive means the model beat the base
             rate; zero or negative means it learned nothing usable
  ECE        expected calibration error — the average gap between "said 60%"
             and "happened 60% of the time". A model can rank matches well and
             still be badly calibrated, and calibration is what multiplies
             through an accumulator
  AUC        ranking ability, independent of scaling — it says whether a model
             that loses on log loss has a real signal that calibration could
             rescue, or no signal at all
  ROI        flat-stake return when a price is available, betting only where
             the model's edge over the implied probability clears a threshold

Everything takes plain arrays so it can score any model, not just the ones in
goal_models.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def _clean(probs, outcomes):
    """Drop rows the model could not price (NaN), returning finite pairs."""
    probs    = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(outcomes)
    return probs[mask], outcomes[mask], mask


def log_loss(probs, outcomes) -> float:
    """Mean negative log likelihood. Lower is better; 0.693 = a coin flip."""
    p, y, _ = _clean(probs, outcomes)
    if p.size == 0:
        return np.nan
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(probs, outcomes) -> float:
    """Mean squared error of the probability. Lower is better."""
    p, y, _ = _clean(probs, outcomes)
    if p.size == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def skill_score(probs, outcomes, baseline_probs) -> float:
    """
    Improvement in log loss over a baseline, as a fraction.

        1.0  perfect        0.0  no better than the baseline      < 0  worse
    """
    model    = log_loss(probs, outcomes)
    baseline = log_loss(baseline_probs, outcomes)
    if not np.isfinite(model) or not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return float(1.0 - model / baseline)


def roc_auc(probs, outcomes) -> float:
    """
    Area under the ROC curve — the probability that a randomly chosen match
    where the event happened was ranked above one where it did not.

    This separates two very different failures. A model can rank matches
    correctly and still lose on log loss because its probabilities are badly
    scaled (fixable by calibration), or it can have no ranking ability at all
    (nothing to fix). 0.5 is a coin flip.

    Computed from rank statistics, with ties averaged, so no sklearn needed.
    """
    p, y, _ = _clean(probs, outcomes)
    if p.size == 0:
        return np.nan

    positives = y > 0
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=float)
    ranks[order] = np.arange(1, p.size + 1)

    # Average the ranks within ties, or a model that predicts one constant
    # value would score an arbitrary AUC instead of 0.5.
    sorted_p = p[order]
    start = 0
    for end in range(1, p.size + 1):
        if end == p.size or sorted_p[end] != sorted_p[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_table(probs, outcomes, bins: int = 10) -> pd.DataFrame:
    """
    Predicted vs observed frequency, bucketed.

    The gap in each row is the model's bias at that confidence level — which is
    what compounds when leg probabilities are multiplied together.
    """
    p, y, _ = _clean(probs, outcomes)
    if p.size == 0:
        return pd.DataFrame()

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx   = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append({
            "bin":       f"{edges[b]:.1f}–{edges[b+1]:.1f}",
            "n":         int(sel.sum()),
            "predicted": float(p[sel].mean()),
            "observed":  float(y[sel].mean()),
            "gap":       float(p[sel].mean() - y[sel].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(probs, outcomes, bins: int = 10) -> float:
    """Sample-weighted mean |predicted − observed| across calibration bins."""
    table = calibration_table(probs, outcomes, bins)
    if table.empty:
        return np.nan
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def betting_roi(probs, odds, outcomes, edge: float = 0.0,
                stake: float = 1.0) -> dict[str, float]:
    """
    Flat-stake return from betting whenever the model's edge clears `edge`.

    A bet is placed when  p_model − 1/odds  >  edge. Profit is (odds − 1) × stake
    on a win and −stake otherwise.

    This is the only metric that answers the question that matters, and it is
    also the noisiest: a few hundred bets tell you very little, so read it
    alongside the scoring rules rather than instead of them.
    """
    p    = np.asarray(probs, dtype=float)
    o    = np.asarray(odds, dtype=float)
    y    = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(o) & np.isfinite(y) & (o > 1.0)

    if not mask.any():
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan}

    p, o, y = p[mask], o[mask], y[mask]
    implied = 1.0 / o
    place   = (p - implied) > edge

    if not place.any():
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan}

    won    = y[place] > 0
    profit = float(np.sum(np.where(won, (o[place] - 1.0) * stake, -stake)))
    staked = float(place.sum() * stake)
    return {
        "bets":   int(place.sum()),
        "staked": staked,
        "profit": profit,
        "roi":    profit / staked,
    }


def evaluate(probs, outcomes, baseline_probs=None, odds=None,
             edge: float = 0.0, bins: int = 10) -> dict[str, float]:
    """
    Full scorecard for one model on one market.

    Returns NaNs (and n = 0) when the model cannot price the market at all —
    Skellam on BTTS, for instance — rather than scoring it on a substitute.
    """
    p, y, mask = _clean(probs, outcomes)

    card: dict[str, float] = {
        "n":          int(p.size),
        "log_loss":   log_loss(probs, outcomes),
        "brier":      brier_score(probs, outcomes),
        "auc":        roc_auc(probs, outcomes),
        "ece":        expected_calibration_error(probs, outcomes, bins),
        "mean_pred":  float(p.mean()) if p.size else np.nan,
        "observed":   float(y.mean()) if p.size else np.nan,
    }

    if baseline_probs is not None:
        card["skill_vs_baseline"] = skill_score(probs, outcomes, baseline_probs)

    if odds is not None:
        card.update({f"roi_{k}": v for k, v in
                     betting_roi(probs, odds, outcomes, edge=edge).items()})

    return card
