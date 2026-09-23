"""
daily_picks.py
--------------
The matches a model reads most confidently.

"Confident" is not "good bet". Confidence says a fixture looks predictable;
whether a price on it is worth taking is a different question, and one this
repo answered in the negative — see models/market_test.py, where these models
lost to closing odds on every outcome.

So this ranks the card by how strongly the model reads each match, and nothing
more. It used to build accumulators too; that came out when the app stopped
being a betting tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Confidence bands for a single selection.
CONFIDENCE_BANDS = [
    (0.70, "🔥 Strong",   "#00ff88"),
    (0.60, "✅ Solid",    "#7eff6e"),
    (0.50, "⚡ Leaning",  "#f5d020"),
    (0.40, "⚠️ Thin",     "#ff9d00"),
    (0.00, "❌ Coin flip", "#ff4444"),
]

DEFAULT_TOP_N = 5


def confidence_band(probability: float) -> tuple[str, str]:
    """(label, colour) for a selection's probability."""
    if not np.isfinite(probability):
        return "—", "#8892a4"
    for threshold, label, colour in CONFIDENCE_BANDS:
        if probability >= threshold:
            return label, colour
    return "❌ Coin flip", "#ff4444"


def rank_picks(card: pd.DataFrame, top_n: int = DEFAULT_TOP_N,
               prob_column: str = "btts_prob") -> pd.DataFrame:
    """
    The day's most confident selections, best first.

    Args:
        card:        a scored fixture card
        top_n:       how many to return
        prob_column: the probability to rank on

    Returns:
        The top rows with `confidence` and `fair_odds` columns added.
    """
    if card.empty or prob_column not in card.columns:
        return card.head(0)

    ranked = card[card[prob_column].notna()].copy()
    if ranked.empty:
        return ranked

    ranked = ranked.sort_values(prob_column, ascending=False).head(top_n)
    ranked["confidence"] = [confidence_band(p)[0] for p in ranked[prob_column]]
    ranked["fair_odds"]  = (1.0 / ranked[prob_column]).round(2)
    return ranked.reset_index(drop=True)


def summarise_card(card: pd.DataFrame, prob_column: str = "btts_prob") -> dict:
    """Headline numbers for the day, for the top of the page."""
    if card.empty or prob_column not in card.columns:
        return {"fixtures": 0, "best_prob": np.nan, "leagues": 0}

    probabilities = card[prob_column].dropna()
    return {
        "fixtures":  len(card),
        "best_prob": float(probabilities.max()) if len(probabilities) else np.nan,
        "leagues":   int(card["league"].nunique()) if "league" in card.columns else 0,
    }
