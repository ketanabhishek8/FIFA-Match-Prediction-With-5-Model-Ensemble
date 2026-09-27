"""Tests for the scoreline model (src/models/score_model.py).

Most of what can go wrong here is silent. A Dixon-Coles implementation with a
sign error, a transposed grid, or a mirror that swaps features without swapping
the two goal targets all produce output that looks entirely reasonable -- a
normalised distribution, plausible scorelines, a believable accuracy. So the
properties are asserted directly rather than inferred from a headline metric:

  - tau is exactly mass-preserving over the four low-score cells
  - negative rho moves probability into draws, which is the entire point of it
  - the grid is oriented [match, home_goals, away_goals], not transposed
  - swapping the two sides swaps the outcome probabilities symmetrically

Run with: pytest tests/test_score_model.py -v
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.evaluation.evaluate import (
    derived_win_accuracy,
    outcome_log_loss,
    ranked_probability_score,
)
from src.models.score_model import (
    ConstantPoissonBaseline,
    PoissonScoreModel,
    dixon_coles_tau,
    outcome_index,
    outcome_proba_from_grid,
    scoreline_grid,
)
from src.pipeline import mirror_targets


@pytest.fixture
def toy_training_set():
    """A feature that genuinely drives goals: as x rises the home side scores
    more and concedes less, so a fitted model has real signal to find."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 1))
    y_home = rng.poisson(np.exp(0.4 + 0.5 * x[:, 0]))
    y_away = rng.poisson(np.exp(0.3 - 0.5 * x[:, 0]))
    return x, y_home, y_away


# --- Dixon-Coles correctness --------------------------------------------

@pytest.mark.parametrize("rho", [-0.2, -0.066, 0.0, 0.1])
@pytest.mark.parametrize("lam,mu", [(1.66, 1.16), (0.4, 0.4), (2.5, 0.9)])
def test_tau_preserves_low_score_mass(rho, lam, mu):
    """The four adjusted cells, weighted by their Poisson probabilities, sum to
    exactly what they summed to before the adjustment -- for any rho.

    This is the cheapest available check that a Dixon-Coles implementation is
    right: a sign error or a swapped lambda/mu breaks it immediately.
    """
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    before = sum(poisson.pmf(h, lam) * poisson.pmf(a, mu) for h, a in cells)
    after = sum(
        poisson.pmf(h, lam) * poisson.pmf(a, mu)
        * dixon_coles_tau(np.array([h]), np.array([a]), np.array([lam]), np.array([mu]), rho)[0]
        for h, a in cells
    )
    assert after == pytest.approx(before, rel=1e-12)


def test_tau_leaves_high_scorelines_alone():
    lam, mu = np.array([1.5]), np.array([1.2])
    for h, a in [(2, 0), (0, 2), (1, 2), (3, 3)]:
        tau = dixon_coles_tau(np.array([h]), np.array([a]), lam, mu, rho=-0.1)
        assert tau[0] == 1.0


def test_negative_rho_raises_draw_probability():
    """The whole reason the correction is applied: independent Poisson
    under-predicts draws, and negative rho moves mass from 1-0 and 0-1 into
    0-0 and 1-1."""
    lam, mu = np.array([1.6]), np.array([1.2])
    plain = outcome_proba_from_grid(scoreline_grid(lam, mu, rho=0.0))
    corrected = outcome_proba_from_grid(scoreline_grid(lam, mu, rho=-0.1))
    assert corrected[0, 1] > plain[0, 1]


def test_positive_rho_lowers_draw_probability():
    lam, mu = np.array([1.6]), np.array([1.2])
    plain = outcome_proba_from_grid(scoreline_grid(lam, mu, rho=0.0))
    corrected = outcome_proba_from_grid(scoreline_grid(lam, mu, rho=0.1))
    assert corrected[0, 1] < plain[0, 1]


# --- grid mechanics ------------------------------------------------------

def test_grid_is_normalised_and_non_negative():
    lam, mu = np.array([0.5, 2.0, 4.0]), np.array([1.0, 0.2, 3.0])
    grid = scoreline_grid(lam, mu, max_goals=10, rho=-0.05)
    assert np.all(grid >= 0)
    assert grid.sum(axis=(1, 2)) == pytest.approx(np.ones(3))


def test_grid_axes_are_home_then_away():
    """A transposed grid is the classic silent bug here: it still normalises,
    still looks plausible, and simply reverses every prediction. A team that
    scores far more than it concedes must put more mass on 2-0 than on 0-2."""
    grid = scoreline_grid(np.array([3.0]), np.array([0.3]), max_goals=10)
    assert grid[0, 2, 0] > grid[0, 0, 2]

    probs = outcome_proba_from_grid(grid)
    assert probs[0, 0] > probs[0, 2]  # home win more likely than away win


def test_outcome_probabilities_sum_to_one():
    lam, mu = np.array([1.66, 0.8]), np.array([1.16, 2.2])
    probs = outcome_proba_from_grid(scoreline_grid(lam, mu, rho=-0.07))
    assert probs.sum(axis=1) == pytest.approx(np.ones(2))


def test_equal_rates_give_symmetric_outcomes():
    probs = outcome_proba_from_grid(scoreline_grid(np.array([1.4]), np.array([1.4])))
    assert probs[0, 0] == pytest.approx(probs[0, 2])


def test_outcome_index_orders_home_draw_away():
    idx = outcome_index([2, 1, 0], [0, 1, 3])
    assert list(idx) == [0, 1, 2]


# --- the fitted model ----------------------------------------------------

def test_model_learns_the_signal(toy_training_set):
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel(dixon_coles=False).fit(x, y_home, y_away)

    strong, weak = np.array([[2.0]]), np.array([[-2.0]])
    assert model.expected_goals(strong)[0, 0] > model.expected_goals(weak)[0, 0]
    assert model.predict_outcome_proba(strong)[0, 0] > model.predict_outcome_proba(weak)[0, 0]


def test_dispersion_is_about_one_on_poisson_data(toy_training_set):
    """Guards the modelling decision to use plain Poisson rather than a negative
    binomial: on genuinely Poisson data the statistic must land near 1, so a
    value well above 1 on real data would be evidence, not noise."""
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel(dixon_coles=False).fit(x, y_home, y_away)
    assert model.dispersion(x, y_home, y_away) == pytest.approx(1.0, abs=0.15)


def test_rho_is_fit_and_within_feasible_range(toy_training_set):
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel(dixon_coles=True).fit(x, y_home, y_away)

    assert model.n_low_score_rows_ > 0
    lam, mu = model.expected_goals(x).T
    assert model.rho_ > -1.0 / max(lam.max(), mu.max())
    # Data generated with independent Poissons has no low-score dependence, so
    # the fitted rho should be near zero rather than picking up a phantom one.
    assert abs(model.rho_) < 0.1


def test_dixon_coles_off_means_rho_zero(toy_training_set):
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel(dixon_coles=False).fit(x, y_home, y_away)
    assert model.rho_ == 0.0


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="must be fit"):
        PoissonScoreModel().predict_grid(np.array([[0.0]]))
    with pytest.raises(RuntimeError, match="must be fit"):
        ConstantPoissonBaseline().predict_grid(np.array([[0.0]]))


def test_predict_scoreline_is_the_grid_mode(toy_training_set):
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel().fit(x, y_home, y_away)
    grid = model.predict_grid(x[:20])
    scores = model.predict_scoreline(x[:20])
    for row in range(20):
        assert grid[row].max() == pytest.approx(grid[row, scores[row, 0], scores[row, 1]])


def test_top_scorelines_are_ranked_and_sum_below_one(toy_training_set):
    x, y_home, y_away = toy_training_set
    model = PoissonScoreModel().fit(x, y_home, y_away)
    top = model.top_scorelines(x[:5], n=4)

    for match in top:
        probs = [p for _, _, p in match]
        assert probs == sorted(probs, reverse=True)
        assert 0 < sum(probs) < 1  # a handful of scorelines is never the whole distribution


def test_baseline_ignores_features(toy_training_set):
    """The floor has to be a genuine floor: if it responded to features it
    would not be measuring what a no-skill forecast achieves."""
    x, y_home, y_away = toy_training_set
    baseline = ConstantPoissonBaseline().fit(x, y_home, y_away)
    eg = baseline.expected_goals(x)
    assert np.allclose(eg, eg[0])
    assert eg[0, 0] == pytest.approx(y_home.mean())


# --- metrics -------------------------------------------------------------

def test_rps_rewards_being_close_on_the_ordered_scale():
    """RPS, unlike log-loss, cares how far wrong a forecast was. Home actually
    won: putting the mass on a draw must beat putting it on an away win."""
    obs = np.array([0])
    near = ranked_probability_score(np.array([[0.2, 0.7, 0.1]]), obs)
    far = ranked_probability_score(np.array([[0.2, 0.1, 0.7]]), obs)
    assert near < far


def test_rps_is_zero_for_a_perfect_confident_forecast():
    assert ranked_probability_score(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == pytest.approx(0.0)


def test_log_loss_punishes_confident_and_wrong():
    obs = np.array([2])
    assert outcome_log_loss(np.array([[0.9, 0.05, 0.05]]), obs) > \
           outcome_log_loss(np.array([[0.4, 0.3, 0.3]]), obs)


def test_derived_win_accuracy_ignores_draws():
    """Draw rows are excluded rather than counted wrong -- the binary classifier
    is never asked about them, so including them would compare two different
    questions."""
    probs = np.array([[0.7, 0.2, 0.1],    # home favoured, home won   -> correct
                      [0.1, 0.2, 0.7],    # away favoured, home won   -> wrong
                      [0.3, 0.4, 0.3]])   # a draw, must not be scored
    out = derived_win_accuracy(probs, [2, 1, 1], [0, 0, 1])
    assert out["n_decided"] == 2
    assert out["derived_win_accuracy"] == pytest.approx(0.5)


def test_derived_win_accuracy_handles_an_all_draw_split():
    out = derived_win_accuracy(np.array([[0.3, 0.4, 0.3]]), [1], [1])
    assert out["n_decided"] == 0
    assert np.isnan(out["derived_win_accuracy"])


# --- mirroring carries the right target ----------------------------------
#
# prepare_split() mirrors the FEATURES and hands back a mask; each task applies
# that mask to its own target. Getting this wrong is silent: the model trains on
# side-swapped features paired with un-swapped goals, learns a muddle, and still
# produces a perfectly plausible-looking distribution.

def test_mirror_targets_swaps_home_and_away_goals():
    home = pd.Series([2, 0, 1], index=[10, 11, 12])
    away = pd.Series([1, 3, 1], index=[10, 11, 12])
    mask = np.array([True, False, True])

    y_home = mirror_targets(home, mask, swap_with=away)
    y_away = mirror_targets(away, mask, swap_with=home)

    # originals first, then one mirrored copy per masked row, sides exchanged
    assert list(y_home) == [2, 0, 1, 1, 1]
    assert list(y_away) == [1, 3, 1, 2, 1]
    # the mirror of a 2-1 is a 1-2
    assert (y_home.iloc[3], y_away.iloc[3]) == (1, 2)


def test_mirror_targets_flips_a_classifier_label():
    y = pd.Series([1, 0, 1])
    assert list(mirror_targets(y, np.array([True, False, True]), flip=True)) == [1, 0, 1, 0, 0]


def test_mirror_targets_is_a_no_op_when_nothing_is_mirrored():
    y = pd.Series([1, 0, 1], index=[7, 8, 9])
    out = mirror_targets(y, np.zeros(3, dtype=bool), flip=True)
    assert list(out) == [1, 0, 1]
