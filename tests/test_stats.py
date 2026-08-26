import numpy as np
import pytest

from airas_eval.metrics import stats
from airas_eval.metrics.classification import accuracy


def test_mean_std_reports_n_and_unbiased_std():
    result = stats.mean_std([0.9, 0.92, 0.88])
    assert result["mean"] == pytest.approx(0.9)
    assert result["std"] == pytest.approx(np.std([0.9, 0.92, 0.88], ddof=1))
    assert result["n"] == 3


def test_mean_std_single_value():
    assert stats.mean_std([0.5])["std"] == 0.0


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, 1 - y_true)
    a = stats.bootstrap_ci(accuracy, y_pred, y_true, seed=1)
    b = stats.bootstrap_ci(accuracy, y_pred, y_true, seed=1)
    assert a == b
    assert a["low"] <= a["point"] <= a["high"]
    assert 0.6 < a["point"] < 0.95


def test_paired_permutation_detects_a_real_difference():
    rng = np.random.default_rng(2)
    base = rng.random(100)
    p_same = stats.paired_permutation_test(base, base + rng.normal(0, 0.01, 100))
    p_diff = stats.paired_permutation_test(base, base + 0.5)
    assert p_diff < 0.01
    assert p_same > 0.05


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci(accuracy, [1], [1, 0])
