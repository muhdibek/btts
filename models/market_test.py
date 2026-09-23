"""
market_test.py
--------------
Does the model beat the market?

Every earlier result in this repo was scored against the BASE RATE, which is a
weak opponent — it knows nothing about which teams are playing. A closing price
is the opposite: it aggregates everything the betting public and the books know,
and on this dataset it is almost perfectly calibrated. Beating it is the only
version of "is there an edge" that decides anything.

The test is walk-forward by season: at the start of each season, fit on every
match played before it, predict that season, then move on. Two questions:

  1. Is the model's log loss lower than the market's?
  2. Betting where the model sees value, at the best available price, does the
     bankroll go up?

Question 2 is the one that pays, and it is also the noisier of the two — a few
hundred bets can show a healthy return on pure chance. Read the two together.

    python -m models.market_test --divisions E0 SP1 --since 2014-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.market_data import BIG_LEAGUES, load_matches, market_probabilities  # noqa: E402
from models.evaluate import betting_roi, log_loss, roc_auc                    # noqa: E402
from models.goal_models import PoissonModel                                   # noqa: E402


OUTCOMES = {
    "home": ("mkt_p_home", "p_home", "max_home", lambda d: (d["fthg"] > d["ftag"])),
    "draw": ("mkt_p_draw", "p_draw", "max_draw", lambda d: (d["fthg"] == d["ftag"])),
    "away": ("mkt_p_away", "p_away", "max_away", lambda d: (d["fthg"] < d["ftag"])),
}


def _season_of(kickoff: pd.Series) -> pd.Series:
    """European seasons run August-May; label each match by its starting year."""
    year = kickoff.dt.year
    return np.where(kickoff.dt.month >= 7, year, year - 1)


def walk_forward_predictions(
    matches:  pd.DataFrame,
    min_train: int = 2000,
    half_life_days: float | None = None,
) -> pd.DataFrame:
    """
    Refit per division at the start of each season and predict that season.

    Returns the test rows with p_home / p_draw / p_away attached.
    """
    matches = matches.sort_values("kickoff").reset_index(drop=True)
    matches["season"] = _season_of(matches["kickoff"])

    blocks = []
    for division, division_matches in matches.groupby("div"):
        seasons = sorted(division_matches["season"].unique())
        for season in seasons:
            train = division_matches[division_matches["season"] < season]
            test  = division_matches[division_matches["season"] == season]
            if len(train) < min_train or test.empty:
                continue

            model = PoissonModel(half_life_days=half_life_days).fit(train)
            priced = [model.predict_markets(row.home_team, row.away_team)
                      for row in test.itertuples(index=False)]
            block = test.copy()
            for key in ("p_home", "p_draw", "p_away"):
                block[key] = [p[key] for p in priced]
            blocks.append(block)

    if not blocks:
        raise ValueError("not enough history to walk forward — lower --min-train")

    return pd.concat(blocks, ignore_index=True).sort_values("kickoff").reset_index(drop=True)


def compare_to_market(predictions: pd.DataFrame, edge: float = 0.0,
                      best_price: bool = True) -> pd.DataFrame:
    """
    Score the model against the closing price, outcome by outcome.

    `edge` is how far the model's probability must exceed the market's implied
    probability before a bet is placed.
    """
    market = market_probabilities(predictions, best_price=best_price)
    frame  = pd.concat([predictions.reset_index(drop=True), market], axis=1)

    rows = []
    for name, (market_col, model_col, price_col, hit) in OUTCOMES.items():
        outcome = hit(frame).astype(int).to_numpy()
        model   = frame[model_col].to_numpy(dtype=float)
        implied = frame[market_col].to_numpy(dtype=float)
        price   = frame[price_col].to_numpy(dtype=float)

        roi = betting_roi(model, price, outcome, edge=edge)
        rows.append({
            "outcome":       name,
            "n":             len(frame),
            "model_logloss": log_loss(model, outcome),
            "market_logloss": log_loss(implied, outcome),
            "model_auc":     roc_auc(model, outcome),
            "market_auc":    roc_auc(implied, outcome),
            "bets":          roi["bets"],
            "roi":           roi["roi"],
            "roi_t":         roi["roi_t"],
            "profit":        roi["profit"],
        })

    summary = pd.DataFrame(rows)
    summary["beats_market"] = summary["model_logloss"] < summary["market_logloss"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the model against closing odds.")
    parser.add_argument("--data", type=Path, default=None,
                        help="local Matches.csv (defaults to the cached download)")
    parser.add_argument("--divisions", nargs="+", default=BIG_LEAGUES)
    parser.add_argument("--since", default="2012-01-01")
    parser.add_argument("--min-train", type=int, default=2000)
    parser.add_argument("--half-life", type=float, default=None)
    parser.add_argument("--edge", type=float, default=0.0,
                        help="model probability must exceed the implied probability by this")
    parser.add_argument("--average-price", action="store_true",
                        help="bet at the market average rather than the best price")
    args = parser.parse_args(argv)

    matches = load_matches(path=args.data, divisions=args.divisions, since=args.since)
    print(f"{len(matches):,} matches with closing odds · "
          f"{matches['kickoff'].min():%Y-%m-%d} → {matches['kickoff'].max():%Y-%m-%d} · "
          f"{', '.join(sorted(matches['div'].unique()))}")

    predictions = walk_forward_predictions(matches, min_train=args.min_train,
                                           half_life_days=args.half_life)
    print(f"Walk-forward: {len(predictions):,} out-of-sample matches\n")

    summary = compare_to_market(predictions, edge=args.edge,
                                best_price=not args.average_price)
    print(summary.round(4).to_string(index=False))

    beaten = int(summary["beats_market"].sum())
    print(f"\nModel beats the closing price on {beaten} of 3 outcomes by log loss.")
    print("roi_t is the ROI in standard errors from break-even. Below ~2 it is "
          "indistinguishable from chance, however good the percentage looks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
