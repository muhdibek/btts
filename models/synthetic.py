"""
synthetic.py
------------
Match generators with a KNOWN data-generating process.

Real results are the only way to find out which model is right about football.
But before trusting a bake-off on real data, the harness itself has to be shown
to work — and that needs data whose truth is known, which real matches never
are. So each generator here draws goals from a stated process, and the bake-off
can then be checked against three things it must get right:

  1. when goals come from a Poisson process, the Poisson model should win
     (and negative binomial should tie it, having fitted its way back to
     near-Poisson by driving r up)
  2. when goals are overdispersed, negative binomial should beat Poisson
  3. when low scores are correlated, Dixon-Coles should beat Poisson

A harness that fails those is measuring something other than model quality.

Every generator also returns `true_p_btts` — the probability the generating
process itself assigns to the match. Scored as a model, that column is the
ORACLE: the best any model could do, and the noise floor a real log loss should
be read against.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models.goal_models import MAX_GOALS, markets_from_grid


DEFAULT_INTERCEPT = np.log(1.35)     # ≈ league average goals per side
DEFAULT_HOME_ADV  = 0.25


def _nb_pmf(k: np.ndarray, lam: float, r: float) -> np.ndarray:
    from scipy.special import gammaln
    return np.exp(gammaln(k + r) - gammaln(r) - gammaln(k + 1) +
                  r * np.log(r / (r + lam)) + k * np.log(lam / (r + lam)))


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from scipy.special import gammaln
    return np.exp(k * np.log(lam) - lam - gammaln(k + 1))


def _match_grid(lam_home: float, lam_away: float, dgp: str,
                rho: float, r: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """The generating process's own joint scoreline distribution."""
    k = np.arange(max_goals + 1, dtype=float)

    if dgp == "negative_binomial":
        grid = np.outer(_nb_pmf(k, lam_home, r), _nb_pmf(k, lam_away, r))
    else:
        grid = np.outer(_poisson_pmf(k, lam_home), _poisson_pmf(k, lam_away))

    if dgp == "dixon_coles":
        grid[0, 0] *= 1.0 - lam_home * lam_away * rho
        grid[0, 1] *= 1.0 + lam_home * rho
        grid[1, 0] *= 1.0 + lam_away * rho
        grid[1, 1] *= 1.0 - rho
        grid = np.clip(grid, 0.0, None)

    return grid / grid.sum()


def simulate_league(
    n_teams:    int = 20,
    n_seasons:  int = 3,
    dgp:        str = "poisson",
    rho:        float = -0.13,
    dispersion: float = 4.0,
    intercept:  float = DEFAULT_INTERCEPT,
    home_adv:   float = DEFAULT_HOME_ADV,
    seed:       int = 7,
) -> pd.DataFrame:
    """
    Simulate seasons of a league from a stated process.

    Args:
        n_teams:    clubs in the league (double round robin per season)
        n_seasons:  how many seasons to play
        dgp:        'poisson', 'negative_binomial' or 'dixon_coles'
        rho:        low-score correlation, used by the dixon_coles process
        dispersion: r for the negative_binomial process; smaller = more
                    overdispersed (variance λ + λ²/r)
        intercept:  log baseline goals per side
        home_adv:   home advantage on the log scale
        seed:       RNG seed

    Returns:
        DataFrame with kickoff, home_team, away_team, fthg, ftag, btts,
        plus true_p_btts / true_p_over25 / true_p_home from the generator.
    """
    if dgp not in ("poisson", "negative_binomial", "dixon_coles"):
        raise ValueError(f"unknown dgp '{dgp}'")

    rng   = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]

    # Latent strengths, centred so the league average is the intercept.
    attack  = rng.normal(0.0, 0.32, n_teams)
    defence = rng.normal(0.0, 0.26, n_teams)
    attack  -= attack.mean()
    defence -= defence.mean()

    flat = np.arange(MAX_GOALS + 1)
    rows = []
    date = pd.Timestamp("2021-08-07")

    for season in range(n_seasons):
        fixtures = list(itertools.permutations(range(n_teams), 2))
        rng.shuffle(fixtures)
        per_round = max(n_teams // 2, 1)

        for n, (h, a) in enumerate(fixtures):
            lam_home = float(np.exp(intercept + home_adv + attack[h] - defence[a]))
            lam_away = float(np.exp(intercept + attack[a] - defence[h]))

            grid = _match_grid(lam_home, lam_away, dgp, rho, dispersion)
            draw = rng.choice(grid.size, p=grid.ravel())
            hg, ag = divmod(draw, grid.shape[1])

            truth = markets_from_grid(grid)
            kickoff = date + pd.Timedelta(days=7 * (n // per_round))

            rows.append({
                "kickoff":        kickoff,
                "season":         f"S{season + 1}",
                "home_team":      teams[h],
                "away_team":      teams[a],
                "fthg":           int(flat[hg]),
                "ftag":           int(flat[ag]),
                "true_p_btts":    truth["p_btts"],
                "true_p_over25":  truth["p_over25"],
                "true_p_home":    truth["p_home"],
            })

        date = rows[-1]["kickoff"] + pd.Timedelta(days=60)

    df = pd.DataFrame(rows).sort_values("kickoff").reset_index(drop=True)
    df["btts"]        = ((df["fthg"] > 0) & (df["ftag"] > 0)).astype(int)
    df["total_goals"] = df["fthg"] + df["ftag"]
    return df


def summarise_dispersion(df: pd.DataFrame) -> dict[str, float]:
    """
    Mean and variance of goals per side. Poisson implies variance = mean;
    anything meaningfully above that is overdispersion.
    """
    goals = np.concatenate([df["fthg"].to_numpy(), df["ftag"].to_numpy()])
    return {
        "mean":              float(goals.mean()),
        "variance":          float(goals.var()),
        "variance_to_mean":  float(goals.var() / goals.mean()),
        "btts_rate":         float(df["btts"].mean()),
    }
