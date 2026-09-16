"""Local numeric and alignment guards shared by pure portfolio calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import month_string
from ..contracts import ErrorCode, ResearchError


def ticker_order(tickers) -> tuple[str, ...]:
    names = tuple(tickers)
    if not names or any(not isinstance(x, str) or not x or x == 'CASH' for x in names):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'tickers must be nonempty ETF identifiers, excluding CASH')
    if len(set(names)) != len(names):
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate ticker')
    return names


def vector(values, names, label: str) -> pd.Series:
    """Validate the complete vector, including zero-weight/inactive assets."""
    names = tuple(names)
    if isinstance(values, dict):
        values = pd.Series(values)
    if isinstance(values, pd.Series):
        if values.index.has_duplicates:
            raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate {label} key')
        if set(values.index) != set(names):
            raise ResearchError(ErrorCode.MISSING_DATA, f'{label} keys do not match expected panel')
        values = values.reindex(names).to_numpy()
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'{label} must be numeric') from exc
    if array.shape != (len(names),):
        raise ResearchError(ErrorCode.MISSING_DATA, f'{label} has incorrect dimensions')
    if not np.isfinite(array).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'{label} contains NaN or infinity')
    return pd.Series(array.copy(), index=list(names), dtype=float, name=label)


def months(values) -> tuple[str, ...]:
    result = tuple(month_string(str(x) if isinstance(x, pd.Period) else x) for x in values)
    if len(set(result)) != len(result):
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate month')
    if any(pd.Period(b, 'M') != pd.Period(a, 'M')+1 for a, b in zip(result, result[1:])):
        raise ResearchError(ErrorCode.CALENDAR_GAP, 'months must be consecutive and increasing')
    return result


def timestamp(value, label: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'invalid {label} timestamp') from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} requires an explicit timezone offset')
    return result


def finite_scalar(value, label: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, (bool, str)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must be numeric')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must be finite') from exc
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} outside allowed finite range')
    return value
