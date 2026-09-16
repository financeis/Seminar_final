"""Small, side-effect-free numerical model contracts."""
import numpy as np
from ..contracts import ErrorCode, ResearchError


def finite(value, name, ndim=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'{name} must be numeric') from exc
    if not np.isfinite(result).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'{name} contains nonfinite values')
    if ndim is not None and result.ndim != ndim:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{name} must have {ndim} dimensions')
    return result


def positive(value, name, *, zero=False):
    result = finite(value, name, 0).item()
    if result < 0 or not zero and result == 0:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{name} must be positive')
    return result


def pd_matrix(value, name):
    result = finite(value, name, 2)
    if not len(result) or result.shape[0] != result.shape[1] or not np.allclose(result, result.T, rtol=1e-12, atol=0):
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'{name} must be symmetric positive definite')
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'{name} must be positive definite') from exc
    return result


def probabilities(value):
    result = finite(value, 'next regime probabilities', 1)
    if result.shape != (6,) or (result < 0).any() or not np.isclose(result.sum(), 1., rtol=0, atol=1e-12):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'probabilities require ordered R0..R5 and sum one')
    return result
