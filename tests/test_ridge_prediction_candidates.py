"""Forward-selection candidate fits independently of LOO loss validity."""
import numpy as np
import pytest
from regime_alloc.contracts import ResearchError
from regime_alloc.models.ridge import ridge_candidates, ridge_prediction_candidates


def augmented_oracle(x, y, query, lam):
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0., *([lam] * x.shape[1])])
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return np.r_[1., query] @ beta, beta[1:], beta[0]


def test_prediction_grid_matches_augmented_normal_equations_one_svd(monkeypatch):
    x = np.array([[-2., 1.], [-1., 2.], [0., -1.], [2., 0.], [3., 2.]])
    y = np.array([[1., 2.], [2., -1.], [4., 3.], [3., 4.], [7., -2.]])
    query = np.array([1.5, -.5])
    grid = np.logspace(-4, 4, 41)
    expected = [augmented_oracle(x, y, query, lam) for lam in grid]
    original = np.linalg.svd
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np.linalg, 'svd', counted)
    result = ridge_prediction_candidates(x, y, query, grid)
    assert len(calls) == 1 and result.valid.all() and not hasattr(result, 'loo_mse')
    np.testing.assert_allclose(result.predictions, [a[0] for a in expected], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.coefficients, [a[1] for a in expected], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.intercepts, [a[2] for a in expected], rtol=1e-12, atol=1e-12)
    assert result.reasons == ('',) * 41


def test_finite_prediction_remains_valid_when_loo_squared_loss_overflows():
    x = np.array([[-1.], [0.], [1.]])
    y = np.array([[1.], [1.], [4.]]) * 1e155
    query = np.array([2.])
    grid = [.1, 1., 10.]
    loo = ridge_candidates(x, y, query, grid)
    assert not loo.valid.any()
    result = ridge_prediction_candidates(x, y, query, grid)
    assert result.valid.all()
    expected = [augmented_oracle(x, y, query, lam)[0] for lam in grid]
    np.testing.assert_allclose(result.predictions, expected, rtol=1e-13)
    np.testing.assert_allclose(result.predictions[1], [4e155], rtol=1e-14)


def test_nonfinite_prediction_is_invalid_with_explicit_padding_reason():
    x = np.array([[-1.], [0.], [1.]])
    result = ridge_prediction_candidates(x, np.array([1., 1., 4.]) * 1e155, [1e308], [.1, 1.])
    assert not result.valid.any() and all(result.reasons)
    assert np.isfinite(result.predictions).all() and not result.predictions.any()


def test_single_row_fit_needs_no_loo_and_grid_errors_are_explicit():
    result = ridge_prediction_candidates([[2., 3.]], [[5., 7.]], [10., 20.], [.1, 10.])
    assert result.valid.all()
    np.testing.assert_array_equal(result.predictions, [[5., 7.], [5., 7.]])
    for grid in ([], [0.], [-1.], [1., 1.], [np.inf]):
        with pytest.raises(ResearchError): ridge_prediction_candidates([[1.]], [2.], [3.], grid)
