"""Independent unpenalized-intercept normal equations and explicit LOO."""
import json
from pathlib import Path
import numpy as np
import pytest
from regime_alloc.contracts import ResearchError
from regime_alloc.models import fit_ridge, ridge_candidates, aggregate_ridge


def fixture():
    path = Path(__file__).resolve().parents[1] / 'reports/paper_audit/evidence/forecasting/fixtures.json'
    return json.loads(path.read_text(encoding='utf-8'))['loocv_counterexample']['data']


def independent_fit(x, y, lam):
    z = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.] + [lam] * x.shape[1])
    return np.linalg.solve(z.T @ z + penalty, z.T @ y)


def test_historical_eight_row_counterexample_all_candidates_and_prediction():
    f = fixture()
    x, y, q = np.array(f['X']), np.array(f['y']), np.array(f['x_query'])
    errors = []
    predictions = []
    for lam in f['lambdas']:
        held = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            coef = independent_fit(x[keep], y[keep], lam)
            held.append((y[i] - np.r_[1., x[i]] @ coef) ** 2)
        errors.append(np.mean(held))
        predictions.append(np.r_[1., q] @ independent_fit(x, y, lam))
    candidates = ridge_candidates(x, y, q, f['lambdas'])
    np.testing.assert_allclose(candidates.loo_mse[:, 0], errors, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(candidates.predictions[:, 0], predictions, rtol=1e-10)
    fit = fit_ridge(x, y, q, lambdas=f['lambdas'])
    best = np.argmin(errors)
    assert fit.selected_lambda[0] == f['lambdas'][best]
    assert fit.prediction[0] == pytest.approx(predictions[best])
    expected = independent_fit(x, y, f['lambdas'][best])
    np.testing.assert_allclose(np.r_[fit.intercept[0], fit.coefficients[:, 0]], expected)


@pytest.mark.parametrize('constant', [4., .1])
def test_constant_target_unpenalized_intercept_and_larger_lambda_tie(constant):
    x = np.arange(18.).reshape(6, 3)
    fit = fit_ridge(x, np.full(6, constant), [30., 31., 32.], lambdas=[.1, 1., 10.])
    assert fit.selected_lambda.tolist() == [10.]
    assert fit.prediction.tolist() == [constant]
    np.testing.assert_array_equal(fit.coefficients, np.zeros((3, 1)))


def test_multi_target_fixed_lambdas_and_candidate_predictions():
    x = np.arange(12.).reshape(6, 2)
    y = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 0])])
    fit = fit_ridge(x, y, [4., 5.], fixed_lambda=[.1, 2.])
    for j, lam in enumerate([.1, 2.]):
        expected = independent_fit(x, y[:, j], lam)
        np.testing.assert_allclose(fit.coefficients[:, j], expected[1:])
        assert fit.intercept[j] == pytest.approx(expected[0])
    assert fit.candidates is None
    assert fit.selection_method == 'fixed_external'


def test_near_unit_hat_uses_explicit_refits():
    x = np.eye(4)
    y = np.array([1., 3., 7., 2.])
    result = ridge_candidates(x, y, np.zeros(4), [1e-14])
    assert result.methods == ('explicit_refit',)
    expected = np.mean([(y[i] - np.delete(y, i).mean()) ** 2 for i in range(4)])
    assert result.loo_mse[0, 0] == pytest.approx(expected, rel=1e-10)


def test_r0_probability_not_renormalized_and_no_crisis_model():
    conditional = np.arange(10.).reshape(5, 2)
    np.testing.assert_allclose(aggregate_ridge(conditional, [.5, .1, .1, .1, .1, .1]), conditional.sum(axis=0) * .1)
    np.testing.assert_array_equal(aggregate_ridge(conditional, [1., 0., 0., 0., 0., 0.]), np.zeros(2))
    with pytest.raises(ResearchError): aggregate_ridge(np.ones((6, 2)), np.ones(6) / 6)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_not_silently_zeroed(bad):
    with pytest.raises(ResearchError): fit_ridge([[0.], [1.]], [0., bad], [0.], lambdas=[1.])
    with pytest.raises(ResearchError): fit_ridge([[0.], [1.]], [0., 1.], [bad], lambdas=[1.])


def test_invalid_candidates_and_insufficient_loo_fail():
    for grid in ([], [0.], [-1.], [np.inf], [1., 1.]):
        with pytest.raises(ResearchError): ridge_candidates([[0.], [1.]], [0., 1.], [0.], grid)
    with pytest.raises(ResearchError): fit_ridge([[0.]], [1.], [0.], lambdas=[1.])


def test_all_candidates_numerically_invalid_are_explicit(monkeypatch):
    import regime_alloc.models.ridge as module
    def impossible(*args): raise np.linalg.LinAlgError('test injected solver failure')
    monkeypatch.setattr(module, '_fit', impossible)
    candidates = ridge_candidates([[0.], [1.]], [0., 1.], [0.], [1., 2.])
    assert not candidates.valid.any()
    assert all(row['mse'] is None and row['reason'] for row in candidates.records(['A']))
    with pytest.raises(ResearchError, match='all Ridge LOO candidates failed'):
        fit_ridge([[0.], [1.]], [0., 1.], [0.], lambdas=[1., 2.])
