from fractions import Fraction
import numpy as np
import pytest
from regime_alloc.contracts import ResearchError
from regime_alloc.models import regularized_moments, black_litterman_posterior, utility_scores


def test_one_asset_posterior_hand_calculation():
    # Precision weighted mean: prior precision=500, view precision=100.
    result = black_litterman_posterior([.02], [[.04]], [1.], [[1.]], [[.01]], tau=.05)
    assert result[0] == pytest.approx(float(Fraction(11, 60)))


def test_two_assets_full_covariance_equation19_independent_inverse():
    covariance = np.array([[.04, .01], [.01, .09]])
    mu, q = np.array([.02, .03]), np.array([.07, -.01])
    p, omega, tau = np.eye(2), np.diag([.01, .03]), .05
    precision = np.linalg.inv(tau * covariance)
    expected = np.linalg.inv(precision + np.linalg.inv(omega)) @ (precision @ mu + np.linalg.inv(omega) @ q)
    got = black_litterman_posterior(mu, covariance, q, p, omega, tau=tau)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(utility_scores(got, covariance, delta=3.), np.linalg.inv(3 * covariance) @ expected)
    prior = black_litterman_posterior(mu, covariance, [], np.empty((0, 2)), np.empty((0, 0)), tau=tau)
    np.testing.assert_array_equal(prior, mu)
    np.testing.assert_allclose(black_litterman_posterior(mu, covariance, mu, p, omega), mu)


def test_sample_covariance_and_shrinkage_units():
    returns = np.array([[.01, .04], [.03, .02], [.02, .09]])
    mu, sample, covariance = regularized_moments(returns, shrink=.25, eps=1e-8)
    centered = returns - returns.mean(axis=0)
    expected = centered.T @ centered / 2
    np.testing.assert_allclose(mu, returns.mean(axis=0))
    np.testing.assert_allclose(sample, expected)
    np.testing.assert_allclose(covariance, .75 * expected + .25 * np.diag(np.diag(expected)) + 1e-8 * np.eye(2))
    np.testing.assert_allclose(utility_scores(mu, covariance), np.linalg.solve(3 * covariance, mu))


@pytest.mark.parametrize('covariance', [[[1., 2.], [2., 1.]], [[1., 0.], [0., 0.]], [[1., 1.], [0., 1.]], [[np.nan, 0.], [0., 1.]]])
def test_non_positive_definite_or_nonfinite_covariance_rejected(covariance):
    with pytest.raises(ResearchError): utility_scores([.1, .2], covariance)
    with pytest.raises(ResearchError): black_litterman_posterior([.1, .2], covariance, [.2, .3], np.eye(2), np.eye(2))


def test_invalid_view_shapes_covariance_and_parameters_rejected():
    with pytest.raises(ResearchError): black_litterman_posterior([1.], [[1.]], [1.], [[1.]], [[0.]])
    with pytest.raises(ResearchError): utility_scores([1.], [[1.]], delta=0)
    with pytest.raises(ResearchError): regularized_moments([[1.]], shrink=.1)
    with pytest.raises(ResearchError): regularized_moments([[1.], [2.]], shrink=-.1)
