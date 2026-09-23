"""
daily_picks.py
--------------
The day's best selections, and ready-made slips built from them.

"Best" here means **the model's most confident**, not the best value. Value is
a comparison against a price, and no odds source reachable from this app covers
these fixtures — so nothing in this module knows whether a selection is
underpriced. A 78% pick the market has at 1.20 is a bad bet and this code
cannot tell you that.

What it can do is rank the card by confidence, and assemble those picks into
accumulators at three risk levels, so the daily output is one short list rather
than a table to squint at.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

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


def _combo_stats(rows: Sequence[Any], prob_column: str) -> tuple[float, float]:
    """Combined probability and total fair odds for a set of legs."""
    probability = 1.0
    for row in rows:
        probability *= float(getattr(row, prob_column))
    return probability, (1.0 / probability if probability > 0 else np.inf)


def build_daily_slips(
    card:        pd.DataFrame,
    prob_column: str = "btts_prob",
    label_column: str | None = None,
    pool:        int = 8,
) -> list[dict]:
    """
    Assemble three slips from the day's most confident picks.

    The three are deliberately different shapes rather than three near-copies
    of the same favourites:

        Banker double   the two most confident legs
        Balanced treble the best three-leg combination by probability
        Long shot       five legs, the highest-probability set at that length

    Each slip reports its combined probability and total FAIR odds — the
    break-even price with no margin. A real book pays less.

    Returns an empty list when the card cannot fill even the smallest slip.
    """
    if card.empty or prob_column not in card.columns:
        return []

    usable = card[card[prob_column].notna()].sort_values(prob_column, ascending=False)
    if len(usable) < 2:
        return []

    candidates = list(usable.head(pool).itertuples(index=False))

    def leg_text(row) -> str:
        if label_column and getattr(row, label_column, None):
            return f"{getattr(row, label_column)}"
        return f"{row.home_team} vs {row.away_team}"

    shapes = [
        ("Banker double",   2, "The two most confident picks on the card"),
        ("Balanced treble", 3, "Best three-leg combination by probability"),
        ("Long shot",       5, "Five legs — low probability by construction"),
    ]

    slips: list[dict] = []
    for name, size, note in shapes:
        if len(candidates) < size:
            continue

        best_combo, best_prob = None, -1.0
        for combo in combinations(candidates, size):
            probability, _ = _combo_stats(combo, prob_column)
            if probability > best_prob:
                best_combo, best_prob = combo, probability

        _, total_fair = _combo_stats(best_combo, prob_column)
        slips.append({
            "name":          name,
            "note":          note,
            "legs":          size,
            "matches":       [f"{row.home_team} vs {row.away_team}" for row in best_combo],
            "selections":    [leg_text(row) for row in best_combo],
            "kickoffs":      [row.kickoff for row in best_combo],
            "leg_probs":     [round(float(getattr(row, prob_column)) * 100, 1)
                              for row in best_combo],
            "leg_fair_odds": [round(1.0 / float(getattr(row, prob_column)), 2)
                              for row in best_combo],
            "combined_prob": round(best_prob * 100, 1),
            "total_fair_odds": round(total_fair, 2),
        })

    return slips


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
