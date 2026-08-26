import numpy as np
import pytest

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics import population as pop

SPACE = [1.0, 2.0, 2.0, 3.0, 4.0]


def test_fraction_better():
    assert pop.fraction_better(4.0, SPACE) == 0.0
    assert pop.fraction_better(2.0, SPACE) == pytest.approx(2 / 5)  # ties don't count
    assert pop.fraction_better(0.0, SPACE) == 1.0


def test_expected_best_of_random_sample_closed_form():
    assert pop.expected_best_of_random_sample(SPACE, 1) == pytest.approx(np.mean(SPACE))
    # n -> large converges to the population maximum
    assert pop.expected_best_of_random_sample(SPACE, 500) == pytest.approx(
        4.0, abs=1e-9
    )
    # hand computation for n=2 with ties: P(max<=v) = F(v)^2
    # F = [.2, .6, .8, 1.0] at v = [1,2,3,4]
    expected = 1 * 0.04 + 2 * (0.36 - 0.04) + 3 * (0.64 - 0.36) + 4 * (1 - 0.64)
    assert pop.expected_best_of_random_sample(SPACE, 2) == pytest.approx(expected)


def test_expected_best_matches_monte_carlo():
    rng = np.random.default_rng(0)
    space = rng.normal(size=200)
    draws = rng.choice(space, size=(200_000, 7), replace=True).max(axis=1)
    assert pop.expected_best_of_random_sample(space, 7) == pytest.approx(
        draws.mean(), abs=5e-3
    )


def test_gain_and_relative_improvement():
    assert pop.gain_over_random_search(4.0, SPACE, 1) == pytest.approx(4.0 - 2.4)
    assert pop.relative_improvement(3.6, SPACE) == pytest.approx(0.5)
    with pytest.raises(UndefinedMetric):
        pop.relative_improvement(1.0, [-1.0, 1.0])
    with pytest.raises(ValueError):
        pop.expected_best_of_random_sample(SPACE, 0)
    with pytest.raises(ValueError):
        pop.fraction_better(1.0, [])
