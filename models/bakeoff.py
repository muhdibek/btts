"""
bakeoff.py
----------
Run every model in goal_models.py over the same matches and score them
head to head.

    # validate the harness on data whose truth is known
    python -m models.bakeoff --synthetic poisson
    python -m models.bakeoff --synthetic negative_binomial
    python -m models.bakeoff --synthetic dixon_coles

    # run it on real results
    python -m models.bakeoff --data data/processed/btts_dataset.csv

    # other markets, and the calibration detail
    python -m models.bakeoff --synthetic dixon_coles --market p_home --calibration

The split is always by DATE — the models train on the earlier matches and are
scored on the later ones. A random split would let a model learn from matches
that had not been played yet, which flatters every metric.

On synthetic data the table also carries an ORACLE row: the generating
process's own probabilities. No model can beat it, and how close the best model
gets is the honest measure of how much of the remaining loss is irreducible
noise rather than model error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.evaluate import calibration_table, evaluate            # noqa: E402
from models.goal_models import MODELS, build_model                 # noqa: E402
from models.synthetic import simulate_league, summarise_dispersion # noqa: E402


REQUIRED_COLUMNS = ("kickoff", "home_team", "away_team", "fthg", "ftag")

MARKET_OUTCOMES = {
    "p_btts":   lambda df: ((df["fthg"] > 0) & (df["ftag"] > 0)).astype(int),
    "p_over25": lambda df: (df["fthg"] + df["ftag"] >= 3).astype(int),
    "p_home":   lambda df: (df["fthg"] > df["ftag"]).astype(int),
    "p_draw":   lambda df: (df["fthg"] == df["ftag"]).astype(int),
    "p_away":   lambda df: (df["fthg"] < df["ftag"]).astype(int),
}


def split_by_date(matches: pd.DataFrame, fraction: float = 0.75
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest `fraction` of matches to train, the rest to test."""
    ordered = matches.sort_values("kickoff").reset_index(drop=True)
    cut     = int(len(ordered) * fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def run_bakeoff(
    matches:   pd.DataFrame,
    market:    str = "p_btts",
    split:     float = 0.75,
    model_names: list[str] | None = None,
    half_life_days: float | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Fit every model on the training slice and score it on the test slice.

    Returns:
        (scorecard, predictions) — the table, and each model's test-set
        probabilities for the chosen market, so callers can dig further.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing required columns: {missing}")
    if market not in MARKET_OUTCOMES:
        raise ValueError(f"unknown market '{market}'; "
                         f"choose from {', '.join(MARKET_OUTCOMES)}")

    matches = matches.copy()
    matches["kickoff"] = pd.to_datetime(matches["kickoff"])
    train, test = split_by_date(matches, split)

    outcomes = MARKET_OUTCOMES[market](test).to_numpy()
    names    = model_names or list(MODELS)

    # The base rate is the reference every skill score is measured against.
    baseline = build_model("base_rate").fit(train)
    baseline_probs = baseline.predict_frame(test)[market].to_numpy()

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    for name in names:
        kwargs = {}
        if half_life_days is not None and name in ("poisson", "dixon_coles",
                                                   "negative_binomial", "skellam"):
            kwargs["half_life_days"] = half_life_days

        model = build_model(name, **kwargs).fit(train)
        probs = model.predict_frame(test)[market].to_numpy()
        predictions[name] = probs

        card = {"model": name}
        card.update(evaluate(probs, outcomes, baseline_probs=baseline_probs))
        card.update(_fitted_parameters(model))
        rows.append(card)

    # --- oracle: only available when the data came from a known process ---
    truth_column = market.replace("p_", "true_p_")
    if truth_column in test.columns:
        card = {"model": "ORACLE (true process)"}
        card.update(evaluate(test[truth_column].to_numpy(), outcomes,
                             baseline_probs=baseline_probs))
        rows.append(card)
        predictions["oracle"] = test[truth_column].to_numpy()

    scorecard = pd.DataFrame(rows)
    return scorecard.sort_values("log_loss", na_position="last").reset_index(drop=True), predictions


def run_repeated(
    dgp:        str,
    repeats:    int = 5,
    market:     str = "p_btts",
    split:      float = 0.75,
    model_names: list[str] | None = None,
    **simulate_kwargs,
) -> pd.DataFrame:
    """
    Repeat the bake-off over several simulated leagues and aggregate.

    One run settles nothing: with a few hundred test matches, log loss carries
    enough sampling noise that models can swap places — a model can even score
    below the oracle on a lucky sample. Averaging over seeds, and counting how
    often each model actually wins, is what separates a real difference from a
    coin flip.

    Returns one row per model: mean and standard deviation of log loss, and the
    number of runs it came out on top (the oracle excluded, since nothing is
    expected to beat the generating process).
    """
    collected: list[pd.DataFrame] = []
    for seed in range(repeats):
        matches = simulate_league(dgp=dgp, seed=seed, **simulate_kwargs)
        card, _ = run_bakeoff(matches, market=market, split=split,
                              model_names=model_names)
        card["run"] = seed
        collected.append(card)

    everything = pd.concat(collected, ignore_index=True)
    scored     = everything[everything["log_loss"].notna()]

    contenders = scored[~scored["model"].str.startswith("ORACLE")]
    win_counts = (contenders.loc[contenders.groupby("run")["log_loss"].idxmin()]
                  ["model"].value_counts())

    summary = (scored.groupby("model")["log_loss"]
               .agg(mean_log_loss="mean", std_log_loss="std", runs="count")
               .reset_index())
    summary["wins"] = summary["model"].map(win_counts).fillna(0).astype(int)
    return summary.sort_values("mean_log_loss").reset_index(drop=True)


def _fitted_parameters(model) -> dict[str, float]:
    """Surface the parameters worth reading off a fit."""
    fit = getattr(model, "fit_result", None)
    if fit is None:
        return {}

    out: dict[str, float] = {}
    if fit.teams:
        out["home_adv"] = round(float(fit.home_adv), 3)
    for key, value in (fit.extra or {}).items():
        out[key] = round(float(value), 3)
    return out


def format_scorecard(scorecard: pd.DataFrame) -> str:
    """Render the table for the terminal."""
    show = scorecard.copy()
    for col in ("log_loss", "brier", "ece", "mean_pred", "observed",
                "skill_vs_baseline"):
        if col in show.columns:
            show[col] = show[col].astype(float).round(4)
    drop = [c for c in ("roi_staked", "roi_profit") if c in show.columns]
    return show.drop(columns=drop).to_string(index=False, na_rep="—")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score goal models head to head.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path,
                        help="CSV of played matches (kickoff, home_team, away_team, fthg, ftag)")
    source.add_argument("--synthetic", choices=["poisson", "negative_binomial", "dixon_coles"],
                        help="generate matches from a known process instead")

    parser.add_argument("--market", default="p_btts", choices=list(MARKET_OUTCOMES))
    parser.add_argument("--split", type=float, default=0.75,
                        help="fraction of matches (by date) used for training")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"subset to run (default: all — {', '.join(MODELS)})")
    parser.add_argument("--half-life", type=float, default=None,
                        help="days; down-weight older training matches")
    parser.add_argument("--calibration", action="store_true",
                        help="print the calibration table for the best model")
    parser.add_argument("--seasons", type=int, default=3, help="synthetic: seasons")
    parser.add_argument("--teams", type=int, default=20, help="synthetic: clubs")
    parser.add_argument("--dispersion", type=float, default=4.0,
                        help="synthetic negative_binomial: r (smaller = more overdispersed)")
    parser.add_argument("--rho", type=float, default=-0.13,
                        help="synthetic dixon_coles: low-score correlation")
    parser.add_argument("--seed", type=int, default=7, help="synthetic: RNG seed")
    parser.add_argument("--repeats", type=int, default=None,
                        help="synthetic: average over N simulated leagues instead "
                             "of scoring one (single runs are noisy)")
    args = parser.parse_args(argv)

    if args.synthetic and args.repeats:
        print(f"Repeating the bake-off over {args.repeats} simulated leagues — "
              f"process: {args.synthetic}\n")
        summary = run_repeated(
            args.synthetic, repeats=args.repeats, market=args.market,
            split=args.split, model_names=args.models,
            n_teams=args.teams, n_seasons=args.seasons,
            rho=args.rho, dispersion=args.dispersion,
        )
        print(summary.round(4).to_string(index=False))
        print("\nwins: runs where that model had the lowest log loss "
              "(the oracle is excluded — nothing should beat the true process).")
        return 0

    if args.synthetic:
        matches = simulate_league(n_teams=args.teams, n_seasons=args.seasons,
                                  dgp=args.synthetic, rho=args.rho,
                                  dispersion=args.dispersion, seed=args.seed)
        stats = summarise_dispersion(matches)
        print(f"Synthetic league — process: {args.synthetic}")
        print(f"  {len(matches):,} matches · goals mean {stats['mean']:.2f} "
              f"variance {stats['variance']:.2f} "
              f"(variance/mean {stats['variance_to_mean']:.2f}) · "
              f"BTTS {stats['btts_rate']:.1%}")
    else:
        matches = pd.read_csv(args.data)
        print(f"Loaded {len(matches):,} matches from {args.data}")

    scorecard, predictions = run_bakeoff(
        matches, market=args.market, split=args.split,
        model_names=args.models, half_life_days=args.half_life,
    )

    train_n = int(len(matches) * args.split)
    print(f"\nMarket: {args.market} · train {train_n:,} → test {len(matches) - train_n:,} "
          f"(split by date)\n")
    print(format_scorecard(scorecard))
    print("\nlog_loss / brier / ece: lower is better. "
          "skill_vs_baseline: > 0 beats the base rate.")

    if args.calibration:
        best = scorecard.loc[scorecard["log_loss"].idxmin(), "model"]
        if best in predictions:
            _, test = split_by_date(matches.assign(
                kickoff=pd.to_datetime(matches["kickoff"])), args.split)
            outcomes = MARKET_OUTCOMES[args.market](test).to_numpy()
            print(f"\nCalibration — {best}")
            print(calibration_table(predictions[best], outcomes).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
