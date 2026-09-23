"""
btts_model.py
-------------
BTTS probability estimator.

Phase 1 (current): Weighted heuristic model using:
  - Historical BTTS rates of both teams
  - Attacking strength (goals scored per game)
  - Defensive weakness (goals conceded per game)

Phase 2 (upgrade path): Swap predict_btts_probability() to use
  an XGBoost classifier trained on a labelled historical dataset.
  See UPGRADE NOTES at bottom of this file.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# SCORING / CONCEDING NORMALISATION BOUNDS
# Adjust these if you move to a league with different scoring averages.
# ---------------------------------------------------------------------------
MAX_GOALS_SCORED   = 3.0   # top of range for attack normalisation
MAX_GOALS_CONCEDED = 2.5   # top of range for defence normalisation


def _normalise(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalise a value to [0, 1]."""
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def _poisson_btts_probability(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_avg_goals: float = 1.35,
) -> float:
    """
    Estimate BTTS probability using independent Poisson distributions.

    Under Poisson assumptions:
        P(home scores ≥ 1) = 1 - e^(-lambda_home)
        P(away scores ≥ 1) = 1 - e^(-lambda_away)
        P(BTTS) = P(home ≥ 1) × P(away ≥ 1)

    Expected goals (lambda):
        lambda_home = home_attack_rate × away_defense_rate × league_avg
        lambda_away = away_attack_rate × home_defense_rate × league_avg

    Args:
        home_attack:   home team avg goals scored per game
        home_defense:  home team avg goals conceded per game
        away_attack:   away team avg goals scored per game
        away_defense:  away team avg goals conceded per game
        league_avg_goals: league mean goals per team per game (default 1.35)
    """
    # Strength ratios relative to league average
    home_attack_rate  = home_attack  / league_avg_goals
    away_attack_rate  = away_attack  / league_avg_goals
    home_defense_rate = home_defense / league_avg_goals
    away_defense_rate = away_defense / league_avg_goals

    lambda_home = home_attack_rate * away_defense_rate * league_avg_goals
    lambda_away = away_attack_rate * home_defense_rate * league_avg_goals

    # Clamp lambdas to reasonable range
    lambda_home = np.clip(lambda_home, 0.2, 4.0)
    lambda_away = np.clip(lambda_away, 0.2, 4.0)

    p_home_scores = 1 - np.exp(-lambda_home)
    p_away_scores = 1 - np.exp(-lambda_away)

    return float(p_home_scores * p_away_scores)


def predict_btts_probability(row: pd.Series) -> float:
    """
    Blend three probability signals into a final BTTS probability:
      1. Poisson model (xG-based)   – weight 40%
      2. Historical BTTS rates      – weight 45%
      3. Attack/defence composite   – weight 15%

    Returns a float in [0.0, 1.0].

    --- UPGRADE PATH ---
    Replace the body of this function with:
        features = _extract_feature_vector(row)
        return float(xgb_model.predict_proba([features])[0][1])
    See UPGRADE NOTES below.
    """

    # --- Signal 1: Poisson probability ---
    p_poisson = _poisson_btts_probability(
        home_attack=row["home_avg_scored"],
        home_defense=row["home_avg_conceded"],
        away_attack=row["away_avg_scored"],
        away_defense=row["away_avg_conceded"],
    )

    # --- Signal 2: Historical BTTS rate composite ---
    # Simple average of each team's last-20-match BTTS rate
    p_historical = (row["home_btts_rate"] + row["away_btts_rate"]) / 2.0

    # --- Signal 3: Attack strength composite ---
    # If both teams have high scoring + both opponents defend poorly → BTTS likely
    home_attack_norm  = _normalise(row["home_avg_scored"],   0.5, MAX_GOALS_SCORED)
    away_attack_norm  = _normalise(row["away_avg_scored"],   0.5, MAX_GOALS_SCORED)
    home_defense_norm = _normalise(row["home_avg_conceded"], 0.5, MAX_GOALS_CONCEDED)  # high = leaky
    away_defense_norm = _normalise(row["away_avg_conceded"], 0.5, MAX_GOALS_CONCEDED)

    # Both teams score well AND both defences are leaky
    p_composite = (
        (home_attack_norm * away_defense_norm) ** 0.5 *
        (away_attack_norm * home_defense_norm) ** 0.5
    )

    # --- Weighted blend ---
    p_final = (
        0.40 * p_poisson    +
        0.45 * p_historical +
        0.15 * p_composite
    )

    return round(float(np.clip(p_final, 0.0, 1.0)), 4)


def score_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply predict_btts_probability() to every row in the match DataFrame.
    Adds a 'btts_prob' column (float) and 'btts_prob_pct' (display string).
    """
    df = df.copy()
    df["btts_prob"]     = df.apply(predict_btts_probability, axis=1)
    df["btts_prob_pct"] = (df["btts_prob"] * 100).round(1).astype(str) + "%"
    return df


def get_confidence_label(prob: float) -> Tuple[str, str]:
    """
    Returns a (label, colour_hex) tuple for UI display.
    Used to colour-code probability cells in the dashboard table.
    """
    if prob >= 0.80:
        return "🔥 Very High", "#00ff88"
    elif prob >= 0.70:
        return "✅ High",      "#7eff6e"
    elif prob >= 0.62:
        return "⚡ Good",      "#f5d020"
    elif prob >= 0.50:
        return "⚠️ Moderate", "#ff9d00"
    else:
        return "❌ Low",       "#ff4444"


# ---------------------------------------------------------------------------
# UPGRADE NOTES — XGBoost Integration (Phase 2)
# ---------------------------------------------------------------------------
# 1. Collect labelled data:
#    - Each row = one match
#    - Features: home/away avg_scored, avg_conceded, btts_rate, form, H2H
#    - Label: 1 if BTTS occurred, 0 otherwise
#
# 2. Train model:
#    import xgboost as xgb
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, y)
#    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
#    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20)
#
# 3. Save model:
#    import joblib
#    joblib.dump(model, "models/btts_xgb_model.pkl")
#
# 4. Load & predict (replace predict_btts_probability body):
#    model = joblib.load("models/btts_xgb_model.pkl")
#    features = _extract_feature_vector(row)   # build numpy array
#    return float(model.predict_proba([features])[0][1])
# ---------------------------------------------------------------------------
