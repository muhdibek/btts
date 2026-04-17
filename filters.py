"""
filters.py
----------
Match filtering logic for the BTTS Dashboard.

Applies configurable thresholds to reduce the full match list
down to only the high-value BTTS candidates.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# DEFAULT FILTER THRESHOLDS (can be overridden via Streamlit sidebar sliders)
# ---------------------------------------------------------------------------
DEFAULT_MIN_BTTS_PROB      = 0.55   # minimum BTTS probability
DEFAULT_MIN_AVG_SCORED     = 1.10   # at least one team must average ≥ this
DEFAULT_MAX_AVG_CONCEDED   = 1.80   # at least one team must concede ≥ this
DEFAULT_MIN_BTTS_ODDS      = 1.55   # filter out matches with suspiciously low odds
DEFAULT_MAX_BTTS_ODDS      = 2.20   # filter out value-less high-odds matches


def filter_high_probability_matches(
    df: pd.DataFrame,
    min_btts_prob:    float = DEFAULT_MIN_BTTS_PROB,
    min_avg_scored:   float = DEFAULT_MIN_AVG_SCORED,
    max_avg_conceded: float = DEFAULT_MAX_AVG_CONCEDED,
    min_btts_odds:    float = DEFAULT_MIN_BTTS_ODDS,
    max_btts_odds:    float = DEFAULT_MAX_BTTS_ODDS,
) -> pd.DataFrame:
    """
    Filter the match DataFrame to keep only high-probability BTTS candidates.

    Rules:
      1. BTTS probability ≥ min_btts_prob
      2. Either team averages ≥ min_avg_scored goals per game (attacking strength)
      3. Either team concedes ≥ max_avg_conceded goals per game (defensive weakness)
      4. BTTS odds within [min_btts_odds, max_btts_odds]

    Args:
        df:                Full scored match DataFrame (output of score_matches)
        min_btts_prob:     Minimum BTTS probability threshold (0–1)
        min_avg_scored:    Minimum avg goals scored by at least one team
        max_avg_conceded:  At least one team must concede this much per game
        min_btts_odds:     Minimum acceptable BTTS odds (decimal)
        max_btts_odds:     Maximum acceptable BTTS odds (decimal)

    Returns:
        Filtered DataFrame sorted by btts_prob descending.
    """
    mask = (
        # --- Primary probability filter ---
        (df["btts_prob"] >= min_btts_prob)

        # --- Attacking strength: at least one team scores regularly ---
        & (
            (df["home_avg_scored"] >= min_avg_scored) |
            (df["away_avg_scored"] >= min_avg_scored)
        )

        # --- Defensive weakness: at least one team concedes regularly ---
        & (
            (df["home_avg_conceded"] >= max_avg_conceded) |
            (df["away_avg_conceded"] >= max_avg_conceded)
        )

        # --- Odds range filter ---
        & (df["btts_odds"] >= min_btts_odds)
        & (df["btts_odds"] <= max_btts_odds)
    )

    return df[mask].sort_values("btts_prob", ascending=False).reset_index(drop=True)


def get_filter_summary(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """
    Returns a summary dict of filter effectiveness for display in the UI.
    """
    return {
        "total_matches":    len(full_df),
        "filtered_matches": len(filtered_df),
        "removed":          len(full_df) - len(filtered_df),
        "avg_btts_prob":    round(filtered_df["btts_prob"].mean() * 100, 1) if len(filtered_df) else 0,
        "max_btts_prob":    round(filtered_df["btts_prob"].max() * 100, 1) if len(filtered_df) else 0,
    }
