"""Equation 10: a conditional Sharpe score, not an expected return."""
from dataclasses import dataclass
import numpy as np
from ..contracts import ErrorCode, ResearchError
from ._validation import finite


@dataclass(frozen=True)
class ConditionalSharpe:
    values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    raw_n: int
    status: tuple[str, ...]
    reasons: tuple[str, ...]


def conditional_sharpe(returns):
    x = finite(returns, 'conditional returns', 2)
    if not x.shape[1]:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'conditional returns require assets')
    # Absent means/std are archive padding, diagnosed explicitly by n/status.
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            mean = x.mean(axis=0) if len(x) else np.zeros(x.shape[1])
            std = x.std(axis=0, ddof=1) if len(x) >= 2 else np.zeros(x.shape[1])
    except FloatingPointError as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'conditional moment overflow') from exc
    # Equal floating-point observations are exactly constant even if the
    # summation used to compute their mean leaves a tiny rounding residual.
    if len(x): std[np.all(x == x[0], axis=0)] = 0.
    valid = (std > 0) & (len(x) >= 2)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            value = np.divide(mean, std, out=np.zeros_like(mean), where=valid)
    except FloatingPointError as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'conditional Sharpe overflow') from exc
    finite(value, 'conditional Sharpe result')
    return ConditionalSharpe(value, mean, std, len(x),
        tuple('ok' if item else 'undefined' for item in valid),
        tuple('' if item else 'undefined_conditional_sharpe' for item in valid))
