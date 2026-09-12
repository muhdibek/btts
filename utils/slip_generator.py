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


# ---------------------------------------------------------------------------
# TARGET-ODDS SLIP BUILDER
# ---------------------------------------------------------------------------
# BTTS prices sit around 1.60–2.20, so a 2- or 3-leg acca tops out near 10.0.
# Hitting a specific payout target (e.g. "a 25 odds slip") needs a variable
# number of legs — typically 4–6 — chosen so the payout clears the target with
# the highest combined probability still available.
# ---------------------------------------------------------------------------

DEFAULT_TARGET_ODDS      = 25.0   # the payout multiple the slip must reach
DEFAULT_MIN_TARGET_LEGS  = 2
DEFAULT_MAX_TARGET_LEGS  = 8
DEFAULT_TARGET_TOLERANCE = 0.05   # accept up to 5% under target if nothing clears it
MAX_CANDIDATE_MATCHES    = 18     # keeps the combination scan bounded


def _combo_teams(combo) -> List[str]:
    """All club names appearing in a candidate slip."""
    teams = []
    for row in combo:
        teams.extend([row.home_team, row.away_team])
    return teams


def _has_repeated_team(combo) -> bool:
    """
    True if any club appears in more than one leg.

    Two legs sharing a club are correlated (and on a single-day card usually
    mean duplicated fixture data), so target slips exclude them by default.
    """
    teams = _combo_teams(combo)
    return len(set(teams)) != len(teams)


def build_target_odds_slips(
    filtered_df: pd.DataFrame,
    target_odds:  float = DEFAULT_TARGET_ODDS,
    min_legs:     int   = DEFAULT_MIN_TARGET_LEGS,
    max_legs:     int   = DEFAULT_MAX_TARGET_LEGS,
    max_slips:    int   = 5,
    tolerance:    float = DEFAULT_TARGET_TOLERANCE,
    unique_teams: bool  = True,
) -> List[Dict[str, Any]]:
    """
    Build the BTTS accumulators that land closest to a payout target.

    Search:
      - every combination of `min_legs`..`max_legs` filtered matches
      - keep those whose total odds reach `target_odds` (or fall no more than
        `tolerance` under it, so a near miss is still offered)
      - rank by: clears target → highest combined probability → fewest legs →
        closest to target

    The highest-probability qualifying slip comes first: once the payout target
    is met, extra odds beyond it only cost win probability.

    Args:
        filtered_df:  High-probability matches (output of filter_high_probability_matches)
        target_odds:  Payout multiple the slip should reach (e.g. 25.0)
        min_legs:     Smallest number of legs to consider
        max_legs:     Largest number of legs to consider
        max_slips:    How many ranked candidates to return
        tolerance:    Fraction of the target a slip may fall short by (0.05 = 5%)
        unique_teams: Reject slips where a club appears in two legs

    Returns:
        List of slip dicts sorted best-first (empty if nothing comes close).
    """
    if filtered_df.empty or max_legs < min_legs:
        return []

    # Scan the strongest candidates only — combination count grows fast.
    candidates_df = (
        filtered_df.sort_values("btts_prob", ascending=False)
                   .head(MAX_CANDIDATE_MATCHES)
    )
    rows = list(candidates_df.itertuples(index=False))
    if len(rows) < min_legs:
        return []

    floor = target_odds * (1.0 - tolerance)
    slips: List[Dict[str, Any]] = []

    for size in range(min_legs, min(max_legs, len(rows)) + 1):
        for combo in combinations(rows, size):
            if unique_teams and _has_repeated_team(combo):
                continue

            total_odds = _calculate_total_odds([row.btts_odds for row in combo])
            if total_odds < floor:
                continue

            btts_probs    = [row.btts_prob for row in combo]
            combined_prob = _calculate_combined_probability(btts_probs)

            slips.append({
                "slip_id":       "",  # assigned after ranking
                "legs":          size,
                "matches":       [f"{row.home_team} vs {row.away_team}" for row in combo],
                "match_ids":     [row.match_id for row in combo],
                "kickoffs":      [row.kickoff for row in combo],
                "btts_probs":    [round(p * 100, 1) for p in btts_probs],
                "btts_odds":     [row.btts_odds for row in combo],
                "combined_prob": round(combined_prob * 100, 1),   # as %
                "combined_prob_raw": combined_prob,               # unrounded, for ranking
                "total_odds":    total_odds,
                "slip_score":    _slip_score(combined_prob, total_odds),
                "target_odds":   target_odds,
                "clears_target": total_odds >= target_odds,
                "odds_gap":      round(total_odds - target_odds, 2),
            })

    slips.sort(key=lambda s: (
        not s["clears_target"],      # slips that reach the target first
        -s["combined_prob_raw"],     # then the likeliest
        s["legs"],                   # then the fewest legs
        abs(s["odds_gap"]),          # then the tightest to target
    ))

    for i, slip in enumerate(slips[:max_slips], 1):
        slip["slip_id"] = f"TGT-{i:03d}"

    return slips[:max_slips]


def build_target_odds_slip(
    filtered_df: pd.DataFrame,
    target_odds: float = DEFAULT_TARGET_ODDS,
    **kwargs: Any,
) -> Dict[str, Any] | None:
    """
    Single best slip for a payout target, or None if none comes close.
    Thin wrapper over build_target_odds_slips().
    """
    slips = build_target_odds_slips(filtered_df, target_odds=target_odds, max_slips=1, **kwargs)
    return slips[0] if slips else None


def potential_return(total_odds: float, stake: float) -> float:
    """Gross return (stake included) for a winning slip."""
    return round(total_odds * stake, 2)
