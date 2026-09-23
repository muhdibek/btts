"""
odds_link.py
------------
Join an odds feed to a fixture card.

Odds and fixtures come from different providers, and providers do not agree on
team names: "Manchester Utd", "Manchester United FC" and "Man United" are one
club. A naive join silently drops those rows, and a dashboard that quietly
shows prices for two thirds of its card is worse than one that shows none,
because nothing on screen says which third is missing.

So the join is explicit: names are normalised, then matched, then anything
still unmatched is REPORTED rather than dropped in silence.

This runs wherever the app runs. The odds themselves need a reachable
provider (see data/apifootball.py); this module only needs the two frames.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


# Club-name noise that carries no identity: legal forms, founding years, and
# the decorations providers add or drop at will.
_NOISE = re.compile(
    r"\b(fc|afc|cf|sc|ac|as|ss|ssc|sv|tsg|vfl|vfb|bsc|rc|rcd|cd|ud|sd|club|"
    r"calcio|futbol|football|soccer|team|1[89]\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

# Abbreviations that are not noise — dropping them loses the club.
_ALIASES = {
    "utd": "united", "man": "manchester", "wolves": "wolverhampton",
    "spurs": "tottenham", "psg": "paris saint germain", "inter": "internazionale",
    "atleti": "atletico", "gladbach": "borussia monchengladbach",
    "dortmund": "borussia dortmund", "bayern": "bayern munchen",
}

MATCH_THRESHOLD = 0.82          # below this, call it unmatched rather than guess


def normalise_name(name: str) -> str:
    """
    Reduce a club name to its identifying words.

    Accents are folded, legal forms and founding years dropped, common
    abbreviations expanded. The result is for comparison only — never show it
    to a reader, who wants the name their source actually uses.
    """
    if not isinstance(name, str):
        return ""

    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().replace("&", " and ").replace("-", " ")
    folded = re.sub(r"[^a-z0-9 ]", " ", folded)
    folded = _NOISE.sub(" ", folded)

    words = [_ALIASES.get(w, w) for w in folded.split()]
    return " ".join(" ".join(words).split())


def name_similarity(left: str, right: str) -> float:
    """
    How alike two club names are, 0–1.

    One name containing the other scores highly on purpose: "Inter" and
    "Internazionale Milano" are the same club, and a plain ratio would not say so.
    """
    a, b = normalise_name(left), normalise_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def _fixture_similarity(home_a, away_a, home_b, away_b) -> float:
    """A fixture matches when BOTH sides do; the weaker side sets the score."""
    return min(name_similarity(home_a, home_b), name_similarity(away_a, away_b))


def link_odds(
    card:      pd.DataFrame,
    odds:      pd.DataFrame,
    threshold: float = MATCH_THRESHOLD,
    same_day:  bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach odds rows to fixture rows.

    Args:
        card:      fixtures with home_team, away_team and kickoff
        odds:      odds with home_team, away_team, kickoff and price columns
        threshold: minimum similarity to accept a pairing
        same_day:  only consider odds kicking off on the same calendar day,
                   which stops a reverse fixture months away from matching

    Returns:
        (linked, unmatched) — the card with price columns attached where a
        match was found, and the fixtures that got none. The second frame is
        the point: it is what the UI should say out loud.
    """
    price_columns = [c for c in odds.columns
                     if c.startswith(("odds_", "max_")) and c in odds.columns]

    if card.empty:
        return card.copy(), card.copy()
    if odds.empty or not price_columns:
        linked = card.copy()
        for column in price_columns:
            linked[column] = np.nan
        linked["odds_matched"] = False
        return linked, card.copy()

    card_dates = pd.to_datetime(card["kickoff"], errors="coerce").dt.date
    odds_dates = pd.to_datetime(odds["kickoff"], errors="coerce").dt.date

    rows, matched_flags = [], []
    for i, fixture in enumerate(card.itertuples(index=False)):
        candidates = odds
        if same_day:
            candidates = odds[odds_dates == card_dates.iloc[i]]

        best_score, best_row = 0.0, None
        for candidate in candidates.itertuples(index=False):
            score = _fixture_similarity(fixture.home_team, fixture.away_team,
                                        candidate.home_team, candidate.away_team)
            if score > best_score:
                best_score, best_row = score, candidate

        if best_row is not None and best_score >= threshold:
            rows.append({c: getattr(best_row, c, np.nan) for c in price_columns})
            matched_flags.append(True)
        else:
            rows.append({c: np.nan for c in price_columns})
            matched_flags.append(False)

    linked = pd.concat([card.reset_index(drop=True),
                        pd.DataFrame(rows)], axis=1)
    linked["odds_matched"] = matched_flags
    return linked, linked[~linked["odds_matched"]].copy()


# ---------------------------------------------------------------------------
# Model against market
# ---------------------------------------------------------------------------

def implied_probability(price: pd.Series | np.ndarray) -> np.ndarray:
    """1 / price, with the bookmaker's margin still in it."""
    price = pd.to_numeric(pd.Series(price), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(price > 1.0, 1.0 / price, np.nan)


def add_market_comparison(linked: pd.DataFrame, model_column: str = "prob",
                          price_column: str = "max_home") -> pd.DataFrame:
    """
    Put the model's number next to the market's, and the gap between them.

    A positive edge is NOT a good bet. It means the model disagrees with the
    price, and the evidence in this repo is that when they disagree the market
    is usually right: this model lost 4.6% flat-staking its disagreements, and
    an independent published model lost 15.2% over 3,834 bets. Read the edge
    as "how far off consensus this call is", which is genuinely informative,
    and not as a signal.
    """
    out = linked.copy()
    if price_column not in out.columns or model_column not in out.columns:
        out["market_prob"] = np.nan
        out["edge"] = np.nan
        return out

    out["market_prob"] = implied_probability(out[price_column])
    out["edge"] = out[model_column] - out["market_prob"]
    return out


def coverage_note(linked: pd.DataFrame) -> str:
    """One line saying how much of the card actually carries a price."""
    if linked.empty or "odds_matched" not in linked.columns:
        return "No odds available for this card."

    matched = int(linked["odds_matched"].sum())
    total   = len(linked)
    if matched == 0:
        return f"No odds matched any of the {total} fixtures on this card."
    if matched == total:
        return f"Prices matched for all {total} fixtures."
    return (f"Prices matched for {matched} of {total} fixtures — "
            f"{total - matched} unmatched, shown without a market column.")
