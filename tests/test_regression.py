import numpy as np
import pytest
from scipy import stats
from sklearn import metrics as skm

from airas_eval.metrics import regression as r

rng = np.random.default_rng(7)
N = 300
Y_TRUE = rng.normal(10, 3, size=N)
Y_PRED = Y_TRUE + rng.normal(0, 1.5, size=N)


def test_mse_rmse_mae_match_sklearn():
    assert r.mse(Y_PRED, Y_TRUE) == pytest.approx(
        skm.mean_squared_error(Y_TRUE, Y_PRED)
    )
    assert r.rmse(Y_PRED, Y_TRUE) == pytest.approx(
        np.sqrt(skm.mean_squared_error(Y_TRUE, Y_PRED))
    )
    assert r.mae(Y_PRED, Y_TRUE) == pytest.approx(
        skm.mean_absolute_error(Y_TRUE, Y_PRED)
    )


def test_mape_matches_sklearn():
    assert r.mape(Y_PRED, Y_TRUE) == pytest.approx(
        skm.mean_absolute_percentage_error(Y_TRUE, Y_PRED)
    )


def test_smape_bounds_and_zero():
    assert r.smape(Y_TRUE, Y_TRUE) == pytest.approx(0.0)
    assert 0.0 <= r.smape(Y_PRED, Y_TRUE) <= 2.0


def test_r2_and_explained_variance_match_sklearn():
    assert r.r2_score(Y_PRED, Y_TRUE) == pytest.approx(skm.r2_score(Y_TRUE, Y_PRED))
    assert r.explained_variance(Y_PRED, Y_TRUE) == pytest.approx(
        skm.explained_variance_score(Y_TRUE, Y_PRED)
    )


def test_pearson_matches_scipy():
    assert r.pearson_r(Y_PRED, Y_TRUE) == pytest.approx(
        stats.pearsonr(Y_PRED, Y_TRUE).statistic
    )


def test_spearman_matches_scipy():
    assert r.spearman_rho(Y_PRED, Y_TRUE) == pytest.approx(
        stats.spearmanr(Y_PRED, Y_TRUE).statistic
    )


def test_spearman_with_ties_matches_scipy():
    pred_tied = np.round(Y_PRED)
    true_tied = np.round(Y_TRUE)
    assert r.spearman_rho(pred_tied, true_tied) == pytest.approx(
        stats.spearmanr(pred_tied, true_tied).statistic
    )


def test_kendall_matches_scipy():
    small_pred, small_true = Y_PRED[:80], Y_TRUE[:80]
    assert r.kendall_tau(small_pred, small_true) == pytest.approx(
        stats.kendalltau(small_pred, small_true).statistic
    )


def test_kendall_with_ties_matches_scipy():
    pred_tied = np.round(Y_PRED[:80])
    true_tied = np.round(Y_TRUE[:80])
    assert r.kendall_tau(pred_tied, true_tied) == pytest.approx(
        stats.kendalltau(pred_tied, true_tied).statistic
    )


def test_mape_zero_reference_raises():
    with pytest.raises(ValueError):
        r.mape([1.0], [0.0])


def test_constant_reference_r2_raises():
    with pytest.raises(ValueError):
        r.r2_score([1.0, 2.0], [3.0, 3.0])
