"""Ex-ante volatility scaling using only the decision's known training returns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import month_string
from ..contracts import TICKERS, ErrorCode, ResearchError
from ._validation import finite_scalar, months, ticker_order, timestamp, vector


@dataclass(frozen=True)
class ScalingResult:
    target_weights: pd.Series
    cash_weight: float
    expected_volatility: float
    multiplier: float
    used_months: tuple[str, ...]
    vol_target: float
    gross_cap: float


def scale_to_volatility(base_weights, returns: pd.DataFrame, *, decision_month: str,
                        decision_at, return_end_at: pd.Series, lookback: int = 36,
                        vol_target: float = .10, gross_cap: float = 2.,
                        tickers: tuple[str, ...] = TICKERS) -> ScalingResult:
    """Scale by target / sqrt(12*w'Cov*w), limited by total absolute exposure.

    Require all 48 consecutive known R_s, s=m-49..m-2, plus their actual
    timezone-aware price-end timestamps. The entire supplied panel must be
    complete and known before the decision; future rows are never trimmed away.
    The covariance uses only its last lookback rows (default 36, sensitivity 12).
    No borrowing costs or realized evaluation-period volatility enter scaling.
    """
    names = ticker_order(tickers)
    w = vector(base_weights, names, 'base weights')
    gross = float(w.abs().sum())
    if gross > 1 + 1e-12:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'base gross exposure exceeds one')
    target = finite_scalar(vol_target, 'vol_target', positive=True)
    cap = finite_scalar(gross_cap, 'gross_cap', positive=True)
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)) or not 2 <= lookback <= 48:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'volatility lookback must be between 2 and 48')
    decision_month = month_string(decision_month)
    decision = timestamp(decision_at, 'decision_at')
    if decision.tz_convert('America/New_York').strftime('%Y-%m') != decision_month:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'decision timestamp does not match decision month')
    if not isinstance(returns, pd.DataFrame):
        raise ResearchError(ErrorCode.MISSING_DATA, 'training returns must be a month-indexed DataFrame')
    observed = months(returns.index)
    m = pd.Period(decision_month, 'M')
    expected = tuple(str(x) for x in pd.period_range(m-49, m-2, freq='M'))
    if observed != expected:
        raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'volatility requires exactly 48 training months m-49 through m-2')
    if returns.columns.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate training ticker')
    if set(returns.columns) != set(names):
        raise ResearchError(ErrorCode.MISSING_DATA, 'training return ticker panel mismatch')
    data = np.asarray([vector(returns.iloc[i], names, 'training returns').to_numpy() for i in range(48)])
    if (data < -1).any():
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'asset returns cannot be below -1')
    if not isinstance(return_end_at, pd.Series) or return_end_at.index.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY if isinstance(return_end_at, pd.Series) else ErrorCode.MISSING_DATA,
                            'return_end_at requires a unique month-indexed Series')
    if tuple(return_end_at.index) != observed:
        raise ResearchError(ErrorCode.MISSING_DATA, 'return-end evidence does not align with training months')
    prior_end = None
    for month, value in return_end_at.items():
        end = timestamp(value, f'return_end_at[{month}]')
        if end >= decision:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'return availability must precede the decision', details={'month': month})
        if end.tz_convert('America/New_York').strftime('%Y-%m') != str(pd.Period(month, 'M')+1):
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'return end does not match the next holding-month boundary')
        if prior_end is not None and end <= prior_end:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'return-end evidence must be increasing')
        prior_end = end
    sample = data[-lookback:]
    # Anchoring before centering makes exactly constant inputs exactly zero.
    centered = sample - sample[0]
    centered -= centered.mean(axis=0)
    with np.errstate(over='ignore', invalid='ignore'):
        covariance = centered.T @ centered / (lookback-1)
        variance = float(w.to_numpy() @ covariance @ w.to_numpy())
    if not np.isfinite(variance) or variance < -1e-15:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'invalid estimated portfolio variance')
    expected_vol = float(np.sqrt(max(variance, 0.))*np.sqrt(12))
    multiplier = min(target/expected_vol, cap/gross) if expected_vol > 0 and gross > 0 else 0.
    scaled = w * multiplier
    if not np.isfinite(scaled.to_numpy()).all() or not np.isfinite(scaled.abs().sum()):
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'scaled weights exceed finite numeric range')
    scaled.name = 'target_weight'
    return ScalingResult(scaled, float(1-scaled.sum()), expected_vol, float(multiplier),
                         observed[-lookback:], target, cap)
