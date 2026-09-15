"""Monthly moments, Eq. 19 posterior means, and separate utility directions."""
import numpy as np
from ..contracts import ErrorCode, ResearchError
from ._validation import finite, pd_matrix, positive


def regularized_moments(returns, *, shrink=.1, eps=1e-8):
    x = finite(returns, 'returns', 2)
    shrink, eps = positive(shrink, 'shrink', zero=True), positive(eps, 'eps', zero=True)
    if len(x) < 2 or not x.shape[1]:
        raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'sample covariance requires two observations and assets')
    if shrink > 1:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'shrink must be at most one')
    with np.errstate(over='raise', invalid='raise'):
        try:
            mean = x.mean(axis=0)
            centered = x - mean
            sample = centered.T @ centered / (len(x) - 1)
            covariance = (1 - shrink) * sample + shrink * np.diag(np.diag(sample)) + eps * np.eye(x.shape[1])
        except FloatingPointError as exc:
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'sample moment overflow') from exc
    pd_matrix(covariance, 'regularized covariance')
    return mean, sample, covariance


def black_litterman_posterior(mean, covariance, q, p, omega, *, tau=.05):
    """Eq. 19 via equivalent Gaussian conditioning; accepts zero views.

    The posterior is an expected return vector. It is not the final position.
    Solving the view covariance system avoids forming explicit inverses.
    """
    mean = finite(mean, 'prior mean', 1)
    covariance = pd_matrix(covariance, 'covariance')
    q, p, omega = finite(q, 'views', 1), finite(p, 'view matrix', 2), finite(omega, 'view covariance', 2)
    tau = positive(tau, 'tau')
    n, k = len(mean), len(q)
    if covariance.shape != (n, n) or p.shape != (k, n) or omega.shape != (k, k):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'Black-Litterman shape mismatch')
    if not k: return mean.copy()
    pd_matrix(omega, 'view covariance')
    try:
        with np.errstate(over='raise', invalid='raise'):
            prior = tau * covariance
            posterior = mean + prior @ p.T @ np.linalg.solve(p @ prior @ p.T + omega, q - p @ mean)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'Black-Litterman linear solve failed') from exc
    return finite(posterior, 'posterior mean')


def utility_scores(mean, covariance, *, delta=3.):
    """Unnormalized solve(delta * covariance, mean); portfolio owns sizing."""
    mean, covariance = finite(mean, 'mean', 1), pd_matrix(covariance, 'covariance')
    delta = positive(delta, 'delta')
    if covariance.shape != (len(mean), len(mean)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'utility mean/covariance shape mismatch')
    try:
        with np.errstate(over='raise', invalid='raise'):
            scores = np.linalg.solve(delta * covariance, mean)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'utility linear solve failed') from exc
    return finite(scores, 'utility scores')
