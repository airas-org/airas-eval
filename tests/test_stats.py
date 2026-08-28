import numpy as np
import pytest
from scipy import stats as sps

from airas_eval.metrics import stats


def test_summarize_matches_numpy_and_scipy():
    values = [0.91, 0.93, 0.92, 0.95]
    out = stats.summarize(values)
    assert out["mean"] == pytest.approx(np.mean(values))
    assert out["std"] == pytest.approx(np.std(values, ddof=1))
    assert out["sem"] == pytest.approx(sps.sem(values))
    assert out["median"] == pytest.approx(np.median(values))
    assert (out["q25"], out["q75"]) == (
        pytest.approx(np.percentile(values, 25)),
        pytest.approx(np.percentile(values, 75)),
    )
    low, high = sps.t.interval(0.95, df=3, loc=np.mean(values), scale=sps.sem(values))
    assert (out["ci95_low"], out["ci95_high"]) == (
        pytest.approx(low),
        pytest.approx(high),
    )
    assert (out["min"], out["max"], out["n"]) == (0.91, 0.95, 4.0)
    assert out["values"] == values
    with pytest.raises(ValueError):
        stats.summarize([])


def test_summarize_single_value_has_no_dispersion():
    out = stats.summarize([1.0])
    assert out["mean"] == out["median"] == 1.0
    assert out["std"] is out["sem"] is out["ci95_low"] is out["ci95_high"] is None


def test_summarize_identical_values_have_degenerate_interval():
    out = stats.summarize([0.5, 0.5, 0.5])
    assert out["std"] == 0.0
    assert (out["ci95_low"], out["ci95_high"]) == (0.5, 0.5)


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
