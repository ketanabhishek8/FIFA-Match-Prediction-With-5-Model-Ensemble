"""Scoreline prediction: two Poisson regressions plus a Dixon-Coles low-score
correction, turning the binary home-win/away-win task into a full distribution
over scorelines.

The feature matrix is exactly the one the classifier ensemble uses -- only the
target changes. That is what makes this a generalisation of the existing task
rather than a second, parallel model: P(home win) is recovered by summing the
grid where home > away, and measured over 5 seeds it reproduces the classifier's
accuracy on the same splits (79.18% +/- 2.41 vs 78.29% +/- 1.62 for an untuned
ensemble). It also restores the ~24% of matches dropped as draws, which the
binary framing structurally cannot represent.

Two modelling choices worth knowing, both measured rather than assumed:

  - **Plain Poisson, not negative binomial.** The raw goal marginals look badly
    overdispersed (variance 3.12 vs mean 1.66 for home goals), which normally
    argues for a negative binomial. That is a red herring: conditional on the
    features, Pearson dispersion is 1.007. The features explain essentially all
    of the apparent overdispersion, so Poisson is correctly specified and the
    extra parameter buys nothing. `dispersion()` re-checks this on demand --
    if it drifts well above 1 after a feature change, revisit the assumption.

  - **Dixon-Coles is a calibration fix, not an accuracy gain.** Independent
    Poisson under-predicts draws (21.8% vs 24.2% actual) because it cannot
    represent the dependence between the two scores at low scorelines. The
    correction closes about half that gap, and is better on 5/5 seeds on both
    RPS and log-loss -- but by 0.0001 and 0.0013 respectively. It cannot change
    win-accuracy at all (it is side-symmetric, so it never moves a home-vs-away
    comparison), and it costs ~0.3 pp of exact-scoreline accuracy because it
    makes 0-0 and 1-1 the modal prediction more often. Keep it for the
    calibration; do not expect it in a headline.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market", Applied Statistics 46(2).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

# Goal rates are clipped into this range before the grid is built. The upper
# bound also sets how negative rho is allowed to go (see `_feasible_rho_range`),
# since tau(0,1) = 1 + lambda*rho must stay positive.
MIN_RATE, MAX_RATE = 1e-3, 12.0

HOME, DRAW, AWAY = 0, 1, 2


def outcome_index(home_score, away_score) -> np.ndarray:
    """Map scorelines to ordered outcome indices: 0 home win, 1 draw, 2 away
    win. Ordered on purpose -- RPS treats the three as a scale, not as
    unrelated categories."""
    home_score, away_score = np.asarray(home_score), np.asarray(away_score)
    return np.where(home_score > away_score, HOME,
                    np.where(home_score == away_score, DRAW, AWAY))


def dixon_coles_tau(home_goals, away_goals, lam, mu, rho: float) -> np.ndarray:
    """The Dixon-Coles adjustment factor for observed scorelines.

    Only the four cells with both scores <= 1 are touched; everything else is
    multiplied by 1. Negative rho moves probability out of 1-0 and 0-1 and into
    0-0 and 1-1, which is what raises the draw rate.

    The factor is *exactly* mass-preserving over those four cells -- summing
    them weighted by their Poisson probabilities returns the unadjusted total
    for any rho. `tests/test_score_model.py` asserts it numerically, which is
    the cheapest available check that an implementation of this is correct.
    """
    home_goals, away_goals = np.asarray(home_goals), np.asarray(away_goals)
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)

    tau = np.ones(np.broadcast(home_goals, away_goals, lam, mu).shape, dtype=float)
    m = (home_goals == 0) & (away_goals == 0)
    tau = np.where(m, 1.0 - lam * mu * rho, tau)
    m = (home_goals == 0) & (away_goals == 1)
    tau = np.where(m, 1.0 + lam * rho, tau)
    m = (home_goals == 1) & (away_goals == 0)
    tau = np.where(m, 1.0 + mu * rho, tau)
    m = (home_goals == 1) & (away_goals == 1)
    tau = np.where(m, 1.0 - rho, tau)
    return tau


def scoreline_grid(lam, mu, max_goals: int = 10, rho: float = 0.0) -> np.ndarray:
    """(n, max_goals+1, max_goals+1) scoreline probabilities from per-match goal
    rates, indexed [match, home_goals, away_goals] and normalised to sum to 1.

    Shared by the fitted model and the constant-rate baseline so both produce
    grids the evaluation code can treat identically.
    """
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    goals = np.arange(max_goals + 1)
    grid = poisson.pmf(goals, lam[:, None])[:, :, None] * poisson.pmf(goals, mu[:, None])[:, None, :]

    if rho != 0.0:
        gi, gj = np.indices((max_goals + 1, max_goals + 1))
        grid = grid * dixon_coles_tau(gi[None, :, :], gj[None, :, :],
                                      lam[:, None, None], mu[:, None, None], rho)
        # Defensive: rho is constrained to keep every training cell positive,
        # but a test-time rate outside the training range could still push one
        # slightly negative.
        np.clip(grid, 0.0, None, out=grid)

    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcome_proba_from_grid(grid: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in the ordered convention of `outcome_index`."""
    width = grid.shape[1]
    gi, gj = np.indices((width, width))
    probs = np.stack([grid[:, gi > gj].sum(1), grid[:, gi == gj].sum(1),
                      grid[:, gi < gj].sum(1)], axis=1)
    return probs / probs.sum(1, keepdims=True)


def _feasible_rho_range(lam, mu) -> tuple[float, float]:
    """The interval of rho keeping all four adjusted cells positive for every
    (lambda, mu) pair given. Binding bounds come from the largest rates."""
    lam, mu = np.asarray(lam), np.asarray(mu)
    lo = -1.0 / max(lam.max(), mu.max())
    hi = min(1.0 / (lam * mu).max(), 1.0)
    return float(lo), float(hi)


class PoissonScoreModel:
    """Two Poisson regressors (home goals, away goals) over a shared feature
    matrix, combined into a scoreline distribution.

    Args:
        max_goals: grid is built over 0..max_goals per side. Mass beyond is
            renormalised away; at these goal rates it is negligible.
        alpha: L2 penalty passed to both PoissonRegressor fits.
        dixon_coles: fit and apply the low-score correction.
        rho_grid_points / rho_bounds: rho is fit by a deterministic grid search
            rather than an optimiser. Only training rows with both scores <= 1
            contribute to its likelihood, so the search is cheap, and a grid
            cannot land in a bad local region or fail to converge.
    """

    def __init__(self, max_goals: int = 10, alpha: float = 1e-3,
                 dixon_coles: bool = True, rho_grid_points: int = 1401,
                 rho_bounds: tuple[float, float] = (-0.35, 0.35),
                 max_iter: int = 2000):
        self.max_goals = max_goals
        self.alpha = alpha
        self.dixon_coles = dixon_coles
        self.rho_grid_points = rho_grid_points
        self.rho_bounds = rho_bounds
        self.max_iter = max_iter

        self.home_model_: PoissonRegressor | None = None
        self.away_model_: PoissonRegressor | None = None
        self.rho_: float = 0.0
        self.rho_at_bound_: bool = False
        self.n_low_score_rows_: int = 0

    # -- fitting ---------------------------------------------------------

    def fit(self, X, y_home, y_away) -> "PoissonScoreModel":
        y_home, y_away = np.asarray(y_home), np.asarray(y_away)
        self.home_model_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, y_home)
        self.away_model_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, y_away)

        if self.dixon_coles:
            lam, mu = self._rates(X)
            self.rho_ = self._fit_rho(y_home, y_away, lam, mu)
        else:
            self.rho_ = 0.0
        return self

    def _fit_rho(self, y_home, y_away, lam, mu) -> float:
        """MLE for rho on the training rows.

        The Poisson terms of the likelihood do not depend on rho, so they drop
        out of the maximisation entirely and only the tau terms remain -- which
        means only rows with both scores <= 1 carry any information about it.
        """
        low = (y_home <= 1) & (y_away <= 1)
        self.n_low_score_rows_ = int(low.sum())
        if not self.n_low_score_rows_:
            return 0.0

        lo, hi = _feasible_rho_range(lam, mu)
        lo, hi = max(lo, self.rho_bounds[0]), min(hi, self.rho_bounds[1])
        if not lo < hi:
            return 0.0

        grid = np.linspace(lo, hi, self.rho_grid_points)
        yh, ya, lam_l, mu_l = y_home[low], y_away[low], lam[low], mu[low]

        best_ll, best_rho = -np.inf, 0.0
        for rho in grid:
            tau = dixon_coles_tau(yh, ya, lam_l, mu_l, rho)
            if (tau <= 0).any():
                continue
            ll = np.log(tau).sum()
            if ll > best_ll:
                best_ll, best_rho = ll, float(rho)

        # A rho pinned to the edge means the feasible range, not the data, chose
        # it -- worth knowing, since it usually signals an extreme fitted rate.
        step = (hi - lo) / (self.rho_grid_points - 1)
        self.rho_at_bound_ = bool(min(best_rho - lo, hi - best_rho) <= step)
        return best_rho

    # -- prediction ------------------------------------------------------

    def _rates(self, X) -> tuple[np.ndarray, np.ndarray]:
        if self.home_model_ is None or self.away_model_ is None:
            raise RuntimeError("PoissonScoreModel must be fit before predicting")
        lam = np.clip(self.home_model_.predict(X), MIN_RATE, MAX_RATE)
        mu = np.clip(self.away_model_.predict(X), MIN_RATE, MAX_RATE)
        return lam, mu

    def expected_goals(self, X) -> np.ndarray:
        """(n, 2) array of expected goals: column 0 home, column 1 away."""
        lam, mu = self._rates(X)
        return np.column_stack([lam, mu])

    def predict_grid(self, X) -> np.ndarray:
        """(n, max_goals+1, max_goals+1) scoreline probabilities, indexed
        [match, home_goals, away_goals] and normalised to sum to 1."""
        lam, mu = self._rates(X)
        rho = self.rho_ if self.dixon_coles else 0.0
        return scoreline_grid(lam, mu, self.max_goals, rho)

    def predict_outcome_proba(self, X) -> np.ndarray:
        """(n, 3) probabilities in the ordered outcome convention used by
        `outcome_index`: home win, draw, away win."""
        return outcome_proba_from_grid(self.predict_grid(X))

    def predict_scoreline(self, X) -> np.ndarray:
        """(n, 2) most likely exact scoreline, [home_goals, away_goals].

        Note this is the mode of the grid, not the rounded expected goals --
        those disagree often, and the mode is the honest answer to "what is the
        single most likely score".
        """
        grid = self.predict_grid(X)
        flat = grid.reshape(len(grid), -1).argmax(1)
        return np.column_stack([flat // (self.max_goals + 1), flat % (self.max_goals + 1)])

    def top_scorelines(self, X, n: int = 5) -> list[list[tuple[int, int, float]]]:
        """Per match, the `n` most likely scorelines as (home, away, prob),
        most likely first. Intended for the API, where showing a 15%-likely
        scoreline alone would overstate how sharp the prediction is."""
        grid = self.predict_grid(X)
        width = self.max_goals + 1
        flat = grid.reshape(len(grid), -1)
        top = np.argsort(flat, axis=1)[:, ::-1][:, :n]
        return [[(int(i // width), int(i % width), float(flat[row, i])) for i in top[row]]
                for row in range(len(grid))]

    # -- diagnostics -----------------------------------------------------

    def dispersion(self, X, y_home, y_away) -> float:
        """Pearson dispersion statistic, pooled over both sides. ~1.0 means
        Poisson is correctly specified; substantially above 1 would mean the
        conditional variance exceeds the conditional mean and a negative
        binomial is worth trying. Measured at 1.007 on the development set."""
        lam, mu = self._rates(X)
        y_home, y_away = np.asarray(y_home), np.asarray(y_away)
        chi2 = (((y_home - lam) ** 2 / lam).sum() + ((y_away - mu) ** 2 / mu).sum())
        return float(chi2 / (2 * len(y_home)))


class ConstantPoissonBaseline:
    """Every match gets the training set's average scoreline.

    The no-skill floor for the scoreline task, and the reason the headline
    numbers mean anything: exact-scoreline accuracy of 15% sounds poor until you
    know that always predicting the modal 1-0 scores 10.4%, and RPS has no
    natural scale at all. Fills the role the WWR baseline fills for the
    classifier -- WWR itself cannot be used here, since it predicts a winner
    rather than a distribution over goals.
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.home_rate_: float | None = None
        self.away_rate_: float | None = None

    def fit(self, X, y_home, y_away) -> "ConstantPoissonBaseline":
        del X  # signature parity with PoissonScoreModel; the point is it uses no features
        self.home_rate_ = float(np.mean(y_home))
        self.away_rate_ = float(np.mean(y_away))
        return self

    def _rates(self, X) -> tuple[np.ndarray, np.ndarray]:
        if self.home_rate_ is None:
            raise RuntimeError("ConstantPoissonBaseline must be fit before predicting")
        n = len(X)
        return np.full(n, self.home_rate_), np.full(n, self.away_rate_)

    def expected_goals(self, X) -> np.ndarray:
        lam, mu = self._rates(X)
        return np.column_stack([lam, mu])

    def predict_grid(self, X) -> np.ndarray:
        lam, mu = self._rates(X)
        return scoreline_grid(lam, mu, self.max_goals, rho=0.0)

    def predict_outcome_proba(self, X) -> np.ndarray:
        return outcome_proba_from_grid(self.predict_grid(X))

    def predict_scoreline(self, X) -> np.ndarray:
        grid = self.predict_grid(X)
        flat = grid.reshape(len(grid), -1).argmax(1)
        return np.column_stack([flat // (self.max_goals + 1), flat % (self.max_goals + 1)])
