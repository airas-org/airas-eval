import numpy as np
import pytest
from scipy import stats as sps

from airas_eval.metrics import stats


def test_mean_std_matches_numpy():
    values = [0.91, 0.93, 0.92, 0.95]
    out = stats.mean_std(values)
    assert out["mean"] == pytest.approx(np.mean(values))
    assert out["std"] == pytest.approx(np.std(values, ddof=1))
    assert (out["min"], out["max"], out["n"]) == (0.91, 0.95, 4.0)
    assert stats.mean_std([1.0])["std"] == 0.0
    with pytest.raises(ValueError):
        stats.mean_std([])


def test_paired_permutation_test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=200)
    b = a - 0.05
    assert stats.paired_permutation_test(a, b) < 0.01
    assert stats.paired_permutation_test(a, a) == pytest.approx(1.0)


def test_paired_permutation_test_agrees_with_scipy_on_null_data():
    rng = np.random.default_rng(1)
    a = rng.normal(size=60)
    b = rng.normal(size=60)
    ours = stats.paired_permutation_test(a, b, n_resamples=20000)
    ref = sps.permutation_test(
        (a, b),
        lambda x, y, axis: np.mean(x - y, axis=axis),
        permutation_type="samples",
        n_resamples=20000,
        random_state=1,
    ).pvalue
    assert ours == pytest.approx(ref, abs=0.03)


def test_paired_permutation_test_rejects_misaligned():
    with pytest.raises(ValueError):
        stats.paired_permutation_test([1.0, 2.0], [1.0])
