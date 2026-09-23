"""
goal_models.py
--------------
The scoreline models under test, behind one interface.

Every model here answers the same question — "how will this match's goals be
distributed?" — and every market is then derived from that distribution rather
than modelled separately:

    P(BTTS)    = Σ  P(H=i, A=j)  for i ≥ 1 and j ≥ 1
    P(over2.5) = Σ  P(H=i, A=j)  for i + j ≥ 3
    P(home)    = Σ  P(H=i, A=j)  for i > j

That shared derivation is what makes the comparison fair: the models differ in
their distributional assumption, not in how their output is post-processed.

Implemented
  BaseRate          every match gets the training-set base rate (the floor any
                    model has to clear to be worth anything)
  TeamRate          mean of the two sides' historical BTTS rates — the signal
                    the dashboard's heuristic leans on hardest (45% weight)
  Poisson           log-linear attack/defence/home-advantage, fitted by maximum
                    likelihood; goals independent between sides
  DixonColes        Poisson plus the low-score dependence correction, which
                    reweights exactly the 0-0, 1-0, 0-1 and 1-1 cells that
                    decide BTTS, plus optional time decay on older matches
  NegativeBinomial  same mean structure, but variance λ + λ²/r, for when goals
                    are overdispersed relative to Poisson
  Skellam           models the goal DIFFERENCE. It gives 1X2 and handicaps, but
                    a difference distribution cannot say whether both sides
                    scored (0-0 and 1-1 are the same difference), so it reports
                    NaN for BTTS rather than pretending otherwise.

Fitting note: the attack/defence parameterisation is shift-invariant (adding a
constant to every attack and subtracting it from the mean changes nothing), so
the ratings are centred inside the likelihood to keep the optimiser well posed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import skellam


MAX_GOALS = 10          # scoreline grid: 0..10 covers >99.9% of football matches
MIN_RATE_GOALS = 0.05   # λ floor
MAX_RATE_GOALS = 6.0    # λ ceiling — above this no real fixture lives
MIN_RATE  = 1e-6
MAX_RATE  = 1 - 1e-6


# ---------------------------------------------------------------------------
# Market derivation — shared by every model that produces a scoreline grid
# ---------------------------------------------------------------------------

def markets_from_grid(grid: np.ndarray) -> dict[str, float]:
    """
    Derive the betting markets from a joint scoreline distribution.

    Args:
        grid: (n+1, n+1) array where grid[i, j] = P(home scores i, away scores j)

    Returns:
        p_btts, p_over25, p_home, p_draw, p_away
    """
    total = grid.sum()
    if not np.isfinite(total) or total <= 0:
        return {k: np.nan for k in
                ("p_btts", "p_over25", "p_home", "p_draw", "p_away")}

    grid = grid / total
    n    = grid.shape[0]
    idx  = np.arange(n)
    home_goals = idx[:, None]
    away_goals = idx[None, :]

    return {
        "p_btts":   float(grid[1:, 1:].sum()),
        "p_over25": float(grid[(home_goals + away_goals) >= 3].sum()),
        "p_home":   float(grid[home_goals > away_goals].sum()),
        "p_draw":   float(np.trace(grid)),
        "p_away":   float(grid[home_goals < away_goals].sum()),
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GoalModel:
    """Common interface. Subclasses implement fit() and either grid or markets."""

    name = "base"

    def fit(self, matches: pd.DataFrame) -> "GoalModel":
        raise NotImplementedError

    def predict_grid(self, home_team: str, away_team: str,
                     neutral: bool = False) -> np.ndarray | None:
        """Joint scoreline distribution, or None if the model does not have one."""
        return None

    def predict_markets(self, home_team: str, away_team: str,
                        neutral: bool = False) -> dict[str, float]:
        grid = self.predict_grid(home_team, away_team, neutral=neutral)
        if grid is None:
            return {k: np.nan for k in
                    ("p_btts", "p_over25", "p_home", "p_draw", "p_away")}
        return markets_from_grid(grid)

    def predict_btts(self, home_team: str, away_team: str,
                     neutral: bool = False) -> float:
        return self.predict_markets(home_team, away_team, neutral=neutral)["p_btts"]

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Run predict_markets() over a match table; returns one row per match."""
        rows = [self.predict_markets(m.home_team, m.away_team,
                                     neutral=bool(getattr(m, "neutral", False)))
                for m in matches.itertuples(index=False)]
        return pd.DataFrame(rows, index=matches.index)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class BaseRateModel(GoalModel):
    """
    Predicts the training-set base rate for every match.

    This is the floor. A model that cannot beat it has learned nothing about
    which matches differ from the average one.
    """

    name = "base_rate"

    def __init__(self) -> None:
        self.rates: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "BaseRateModel":
        hg = matches["fthg"].to_numpy()
        ag = matches["ftag"].to_numpy()
        self.rates = {
            "p_btts":   float(np.mean((hg > 0) & (ag > 0))),
            "p_over25": float(np.mean(hg + ag >= 3)),
            "p_home":   float(np.mean(hg > ag)),
            "p_draw":   float(np.mean(hg == ag)),
            "p_away":   float(np.mean(hg < ag)),
        }
        return self

    def predict_markets(self, home_team: str, away_team: str,
                        neutral: bool = False) -> dict[str, float]:
        return dict(self.rates)


class TeamRateModel(GoalModel):
    """
    Mean of the two sides' historical BTTS rates — no distributional
    assumption at all.

    This is the single strongest term in the dashboard's current heuristic
    (45% of the blend), isolated so the fancier models have something real to
    beat rather than only the base rate.
    """

    name = "team_rate"

    def __init__(self) -> None:
        self.team_btts: dict[str, float] = {}
        self.fallback  = 0.5
        self.other: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "TeamRateModel":
        hg = matches["fthg"].to_numpy()
        ag = matches["ftag"].to_numpy()
        btts = (hg > 0) & (ag > 0)

        tally: dict[str, list[int]] = {}
        for team, flag in zip(matches["home_team"], btts):
            tally.setdefault(team, []).append(int(flag))
        for team, flag in zip(matches["away_team"], btts):
            tally.setdefault(team, []).append(int(flag))

        self.team_btts = {team: float(np.mean(flags)) for team, flags in tally.items()}
        self.fallback  = float(btts.mean())
        # The other markets are not what this model is for; report the base rate.
        self.other = BaseRateModel().fit(matches).rates
        return self

    def predict_markets(self, home_team: str, away_team: str,
                        neutral: bool = False) -> dict[str, float]:
        home = self.team_btts.get(home_team, self.fallback)
        away = self.team_btts.get(away_team, self.fallback)
        out  = dict(self.other)
        out["p_btts"] = float(np.clip((home + away) / 2.0, MIN_RATE, MAX_RATE))
        return out


# ---------------------------------------------------------------------------
# Attack / defence rating models
# ---------------------------------------------------------------------------

@dataclass
class RatingFit:
    """Fitted log-linear ratings shared by the Poisson family."""
    teams:      list[str] = field(default_factory=list)
    attack:     dict[str, float] = field(default_factory=dict)
    defence:    dict[str, float] = field(default_factory=dict)
    intercept:  float = 0.0
    home_adv:   float = 0.0
    extra:      dict[str, float] = field(default_factory=dict)
    converged:  bool = False

    def rates(self, home_team: str, away_team: str,
              neutral: bool = False) -> tuple[float, float]:
        """
        Expected goals (λ) for the home and away side.

        `neutral` drops the home-advantage term — for a match on neutral
        ground, "home" is only a label on the fixture.
        """
        atk_h = self.attack.get(home_team, 0.0)
        atk_a = self.attack.get(away_team, 0.0)
        def_h = self.defence.get(home_team, 0.0)
        def_a = self.defence.get(away_team, 0.0)
        advantage = 0.0 if neutral else self.home_adv
        lam_home = np.exp(self.intercept + advantage + atk_h - def_a)
        lam_away = np.exp(self.intercept + atk_a - def_h)

        # Pairings the ratings were never fitted on — the best side in the
        # world against the weakest — extrapolate to expected-goal figures no
        # football match has produced (14 goals, say), which also degenerates
        # the scoreline grid since it only runs to MAX_GOALS. The rates are
        # clamped to a range real matches actually occupy; a fixture pinned to
        # the ceiling is the model saying "far outside what I have seen", not a
        # forecast to take literally.
        return (float(np.clip(lam_home, MIN_RATE_GOALS, MAX_RATE_GOALS)),
                float(np.clip(lam_away, MIN_RATE_GOALS, MAX_RATE_GOALS)))


def _time_weights(matches: pd.DataFrame, half_life_days: float | None) -> np.ndarray:
    """
    Exponential decay so older matches count less.

    half_life_days=None disables it (every match weighted 1).
    """
    if half_life_days is None or "kickoff" not in matches.columns:
        return np.ones(len(matches))

    kickoff = pd.to_datetime(matches["kickoff"])
    age_days = (kickoff.max() - kickoff).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2.0) * age_days.to_numpy() / half_life_days)


class PoissonModel(GoalModel):
    """
    Independent Poisson goals with log-linear attack/defence ratings.

        log λ_home = μ + γ + attack[home] − defence[away]
        log λ_away = μ     + attack[away] − defence[home]

    γ is the home advantage. Fitted by maximum likelihood over the training
    matches, with ratings centred for identifiability.
    """

    name = "poisson"

    def __init__(self, half_life_days: float | None = None, max_goals: int = MAX_GOALS,
                 max_iter: int = 500, max_fun: int | None = None, ridge: float = 0.0):
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        # A club league fits ~40 parameters and converges well inside the
        # defaults. International football fits hundreds, and since the
        # gradient is numerical each iteration costs one evaluation per
        # parameter — so the FUNCTION-EVALUATION budget binds long before the
        # iteration count, and a truncated fit reports converged=False.
        self.max_iter = max_iter
        self.max_fun  = max_fun
        # Ridge shrinkage on the ratings. Default 0 keeps club fits exactly as
        # they were backtested. A small positive value stops a team with three
        # matches from earning an extreme rating it has not evidenced.
        self.ridge = ridge
        self.fit_result = RatingFit()

    # --- likelihood ---------------------------------------------------------

    def _unpack(self, params: np.ndarray, n_teams: int):
        intercept = params[0]
        home_adv  = params[1]
        attack    = params[2:2 + n_teams]
        defence   = params[2 + n_teams:2 + 2 * n_teams]
        # Centre the ratings: the likelihood is invariant to a common shift.
        return intercept, home_adv, attack - attack.mean(), defence - defence.mean()

    def _log_likelihood_terms(self, lam_home, lam_away, hg, ag, params, n_teams):
        """Poisson log-pmf for both sides (factorials dropped — constant in params)."""
        return (hg * np.log(lam_home) - lam_home +
                ag * np.log(lam_away) - lam_away)

    def _negative_log_likelihood(self, params, home_idx, away_idx, hg, ag,
                                 weights, n_teams, neutral=None):
        intercept, home_adv, attack, defence = self._unpack(params, n_teams)
        # On neutral ground there is no home side to advantage.
        advantage = home_adv if neutral is None else home_adv * (1.0 - neutral)
        lam_home = np.exp(intercept + advantage + attack[home_idx] - defence[away_idx])
        lam_away = np.exp(intercept + attack[away_idx] - defence[home_idx])
        lam_home = np.clip(lam_home, 1e-8, 25.0)
        lam_away = np.clip(lam_away, 1e-8, 25.0)

        terms = self._log_likelihood_terms(lam_home, lam_away, hg, ag, params, n_teams)
        penalty = (self.ridge * float(np.sum(attack ** 2) + np.sum(defence ** 2))
                   if self.ridge else 0.0)
        return -float(np.sum(weights * terms)) + penalty

    def _initial_params(self, n_teams: int, hg, ag) -> np.ndarray:
        mean_goals = max(float(np.mean(np.concatenate([hg, ag]))), 0.1)
        start = np.zeros(2 + 2 * n_teams)
        start[0] = np.log(mean_goals)
        start[1] = 0.2
        return start

    def _extra_param_count(self) -> int:
        return 0

    def _store_extra(self, params: np.ndarray, n_teams: int) -> dict[str, float]:
        return {}

    # --- api ----------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {team: i for i, team in enumerate(teams)}
        n     = len(teams)

        home_idx = matches["home_team"].map(index).to_numpy()
        away_idx = matches["away_team"].map(index).to_numpy()
        hg = matches["fthg"].to_numpy(dtype=float)
        ag = matches["ftag"].to_numpy(dtype=float)
        weights = _time_weights(matches, self.half_life_days)
        # Optional: a 'neutral' column marks matches played on neutral ground,
        # which is the norm in international tournaments and never happens in
        # league football. Absent, every match is treated as having a host.
        neutral = (matches["neutral"].astype(float).to_numpy()
                   if "neutral" in matches.columns else None)

        start = self._initial_params(n, hg, ag)
        if self._extra_param_count():
            start = np.concatenate([start, np.zeros(self._extra_param_count())])
            start = self._seed_extra(start, n)

        result = minimize(
            self._negative_log_likelihood,
            start,
            args=(home_idx, away_idx, hg, ag, weights, n, neutral),
            method="L-BFGS-B",
            bounds=self._bounds(n),
            options={"maxiter": self.max_iter,
                     "maxfun": self.max_fun or max(15000, 60 * len(start))},
        )

        intercept, home_adv, attack, defence = self._unpack(result.x, n)
        self.fit_result = RatingFit(
            teams=teams,
            attack={t: float(a) for t, a in zip(teams, attack)},
            defence={t: float(d) for t, d in zip(teams, defence)},
            intercept=float(intercept),
            home_adv=float(home_adv),
            extra=self._store_extra(result.x, n),
            converged=bool(result.success),
        )
        return self

    def _seed_extra(self, start: np.ndarray, n_teams: int) -> np.ndarray:
        return start

    def _bounds(self, n_teams: int) -> list[tuple[float, float]] | None:
        return None

    def _marginal_pmf(self, lam: float) -> np.ndarray:
        """P(X = k) for k = 0..max_goals."""
        k = np.arange(self.max_goals + 1)
        return np.exp(k * np.log(lam) - lam - gammaln(k + 1))

    def predict_grid(self, home_team: str, away_team: str,
                     neutral: bool = False) -> np.ndarray:
        lam_home, lam_away = self.fit_result.rates(home_team, away_team, neutral=neutral)
        return np.outer(self._marginal_pmf(lam_home), self._marginal_pmf(lam_away))


class DixonColesModel(PoissonModel):
    """
    Poisson with the Dixon & Coles (1997) low-score correction.

    Independent Poisson gets the 0-0 / 1-0 / 0-1 / 1-1 cells wrong — real
    matches produce more of some and fewer of others than independence implies.
    The τ correction reweights exactly those four cells:

        τ(0,0) = 1 − λμρ      τ(0,1) = 1 + λρ
        τ(1,0) = 1 + μρ       τ(1,1) = 1 − ρ

    Those are the cells BTTS turns on (three of the four are no-BTTS outcomes),
    so this is the correction most likely to matter for this app specifically.

    `half_life_days` additionally down-weights older matches, as in the paper.
    """

    name = "dixon_coles"

    def _extra_param_count(self) -> int:
        return 1                                   # rho

    def _seed_extra(self, start: np.ndarray, n_teams: int) -> np.ndarray:
        start[-1] = -0.05
        return start

    def _bounds(self, n_teams: int):
        # rho outside roughly (-1, 1) drives tau negative for plausible rates.
        return [(None, None)] * (2 + 2 * n_teams) + [(-0.35, 0.35)]

    @staticmethod
    def _tau(hg, ag, lam_home, lam_away, rho):
        tau = np.ones_like(lam_home)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam_home[m00] * lam_away[m00] * rho
        tau[m01] = 1.0 + lam_home[m01] * rho
        tau[m10] = 1.0 + lam_away[m10] * rho
        tau[m11] = 1.0 - rho
        return np.clip(tau, 1e-8, None)

    def _log_likelihood_terms(self, lam_home, lam_away, hg, ag, params, n_teams):
        rho  = params[-1]
        base = (hg * np.log(lam_home) - lam_home +
                ag * np.log(lam_away) - lam_away)
        return base + np.log(self._tau(hg, ag, lam_home, lam_away, rho))

    def _store_extra(self, params: np.ndarray, n_teams: int) -> dict[str, float]:
        return {"rho": float(params[-1])}

    def predict_grid(self, home_team: str, away_team: str,
                     neutral: bool = False) -> np.ndarray:
        lam_home, lam_away = self.fit_result.rates(home_team, away_team, neutral=neutral)
        grid = np.outer(self._marginal_pmf(lam_home), self._marginal_pmf(lam_away))

        rho = self.fit_result.extra.get("rho", 0.0)
        grid[0, 0] *= 1.0 - lam_home * lam_away * rho
        grid[0, 1] *= 1.0 + lam_home * rho
        grid[1, 0] *= 1.0 + lam_away * rho
        grid[1, 1] *= 1.0 - rho

        grid = np.clip(grid, 0.0, None)
        return grid / grid.sum()


class NegativeBinomialModel(PoissonModel):
    """
    Negative binomial goals: same mean structure, variance λ + λ²/r.

    Poisson forces variance = mean. Football goals are often overdispersed —
    more blowouts and more blanks than Poisson allows — and that misses in both
    tails, which is exactly where BTTS lives (a blank is a no, a blowout is
    usually a yes). r is shared across matches; as r → ∞ this becomes Poisson,
    so the fitted r says directly how much overdispersion the data holds.
    """

    name = "negative_binomial"

    def _extra_param_count(self) -> int:
        return 1                                   # log r

    def _seed_extra(self, start: np.ndarray, n_teams: int) -> np.ndarray:
        start[-1] = np.log(8.0)
        return start

    def _bounds(self, n_teams: int):
        # log r in [log 0.5, log 500]: the top end is numerically Poisson.
        return [(None, None)] * (2 + 2 * n_teams) + [(np.log(0.5), np.log(500.0))]

    @staticmethod
    def _nb_log_pmf(k, lam, r):
        """log P(X = k) with mean lam and dispersion r."""
        return (gammaln(k + r) - gammaln(r) - gammaln(k + 1) +
                r * np.log(r / (r + lam)) + k * np.log(lam / (r + lam)))

    def _log_likelihood_terms(self, lam_home, lam_away, hg, ag, params, n_teams):
        r = np.exp(params[-1])
        return (self._nb_log_pmf(hg, lam_home, r) +
                self._nb_log_pmf(ag, lam_away, r))

    def _store_extra(self, params: np.ndarray, n_teams: int) -> dict[str, float]:
        return {"r": float(np.exp(params[-1]))}

    def _marginal_pmf(self, lam: float) -> np.ndarray:
        r = self.fit_result.extra.get("r", 100.0)
        k = np.arange(self.max_goals + 1, dtype=float)
        return np.exp(self._nb_log_pmf(k, lam, r))


class SkellamModel(PoissonModel):
    """
    Skellam: the distribution of the goal DIFFERENCE, H − A, as the difference
    of two Poissons.

    It prices 1X2 and handicaps straight off P(D > 0), P(D = 0), P(D < 0).

    It cannot price BTTS, and this class does not pretend it can: 0-0 and 1-1
    are both D = 0, so the difference distribution carries no information about
    whether both sides scored. predict_markets() reports NaN for p_btts and
    p_over25, and the bake-off reports it as "not applicable" rather than
    scoring it on a market it structurally cannot serve.
    """

    name = "skellam"

    def predict_grid(self, home_team: str, away_team: str,
                     neutral: bool = False) -> None:
        return None

    def predict_markets(self, home_team: str, away_team: str,
                        neutral: bool = False) -> dict[str, float]:
        lam_home, lam_away = self.fit_result.rates(home_team, away_team, neutral=neutral)
        return {
            "p_btts":   np.nan,          # structurally unavailable — see docstring
            "p_over25": np.nan,
            "p_home":   float(skellam.sf(0, lam_home, lam_away)),    # P(D > 0)
            "p_draw":   float(skellam.pmf(0, lam_home, lam_away)),
            "p_away":   float(skellam.cdf(-1, lam_home, lam_away)),  # P(D < 0)
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, type[GoalModel]] = {
    "base_rate":         BaseRateModel,
    "team_rate":         TeamRateModel,
    "poisson":           PoissonModel,
    "dixon_coles":       DixonColesModel,
    "negative_binomial": NegativeBinomialModel,
    "skellam":           SkellamModel,
}


def build_model(name: str, **kwargs) -> GoalModel:
    """Instantiate a model by registry name."""
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; available: {', '.join(MODELS)}")
    return MODELS[name](**kwargs)
