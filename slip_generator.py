"""
slip_generator.py
-----------------
Generates 2-match and 3-match BTTS betting slip combinations
from the filtered high-probability match list.

Slip scoring formula:
    slip_score = combined_btts_probability × total_odds

This ranks slips by both value and likelihood simultaneously.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# DEFAULT SLIP RULES
# ---------------------------------------------------------------------------
DEFAULT_MIN_TOTAL_ODDS     = 3.0    # slip must have total odds ≥ this
DEFAULT_MAX_SLIPS_RETURNED = 15     # cap results to keep UI clean
DEFAULT_SLIP_SIZES         = (2, 3) # generate 2-leg and 3-leg accas


def _calculate_combined_probability(probs: List[float]) -> float:
    """
    Combined BTTS probability assuming match independence.
    P(all BTTS) = P1 × P2 × ... × Pn
    """
    result = 1.0
    for p in probs:
        result *= p
    return round(result, 4)


def _calculate_total_odds(odds: List[float]) -> float:
    """
    Multiply decimal odds for accumulator total.
    total_odds = odds1 × odds2 × ... × oddsn
    """
    result = 1.0
    for o in odds:
        result *= o
    return round(result, 2)


def _slip_score(combined_prob: float, total_odds: float) -> float:
    """
    Score a slip by its expected value proxy:
        score = combined_probability × total_odds

    Higher = better balance of likelihood and return.
    """
    return round(combined_prob * total_odds, 4)


def generate_slips(
    filtered_df: pd.DataFrame,
    min_total_odds:     float = DEFAULT_MIN_TOTAL_ODDS,
    max_slips:          int   = DEFAULT_MAX_SLIPS_RETURNED,
    slip_sizes:         tuple = DEFAULT_SLIP_SIZES,
) -> List[Dict[str, Any]]:
    """
    Generate all valid BTTS accumulator slips from the filtered match list.

    For each combination of 2 or 3 matches:
      - Calculate combined BTTS probability (product)
      - Calculate total odds (product of BTTS odds)
      - Keep only slips where total_odds ≥ min_total_odds
      - Score and rank slips

    Args:
        filtered_df:    High-probability matches (output of filter_high_probability_matches)
        min_total_odds: Minimum acceptable total slip odds
        max_slips:      Maximum number of slips to return
        slip_sizes:     Tuple of leg counts to generate (default: 2-leg and 3-leg)

    Returns:
        List of slip dicts sorted by slip_score descending.
    """
    if len(filtered_df) < 2:
        return []

    slips = []

    for size in slip_sizes:
        if len(filtered_df) < size:
            continue

        for combo in combinations(filtered_df.itertuples(index=False), size):
            match_names   = [f"{row.home_team} vs {row.away_team}" for row in combo]
            btts_probs    = [row.btts_prob   for row in combo]
            btts_odds_lst = [row.btts_odds   for row in combo]
            match_ids     = [row.match_id    for row in combo]
            kickoffs      = [row.kickoff     for row in combo]

            combined_prob = _calculate_combined_probability(btts_probs)
            total_odds    = _calculate_total_odds(btts_odds_lst)

            # --- Apply minimum odds filter ---
            if total_odds < min_total_odds:
                continue

            score = _slip_score(combined_prob, total_odds)

            slips.append({
                "slip_id":          f"SLIP-{len(slips) + 1:03d}",
                "legs":             size,
                "matches":          match_names,
                "match_ids":        match_ids,
                "kickoffs":         kickoffs,
                "btts_probs":       [round(p * 100, 1) for p in btts_probs],
                "btts_odds":        btts_odds_lst,
                "combined_prob":    round(combined_prob * 100, 1),  # as %
                "total_odds":       total_odds,
                "slip_score":       score,
            })

    # --- Sort by slip_score descending, then cap ---
    slips.sort(key=lambda x: x["slip_score"], reverse=True)
    return slips[:max_slips]


def slips_to_dataframe(slips: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten the slip list into a display-friendly DataFrame.
    One row per slip, with match names joined for readability.
    """
    if not slips:
        return pd.DataFrame()

    rows = []
    for slip in slips:
        rows.append({
            "Slip ID":         slip["slip_id"],
            "Legs":            slip["legs"],
            "Matches":         " | ".join(slip["matches"]),
            "BTTS Probs (%)":  " / ".join(str(p) for p in slip["btts_probs"]),
            "BTTS Odds":       " × ".join(str(o) for o in slip["btts_odds"]),
            "Combined Prob":   f"{slip['combined_prob']}%",
            "Total Odds":      slip["total_odds"],
            "Slip Score":      slip["slip_score"],
        })

    return pd.DataFrame(rows)
