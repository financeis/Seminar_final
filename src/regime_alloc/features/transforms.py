"""FRED t-codes on a continuous monthly calendar; no imputation or IO."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd

from ..config import month_string
from ..contracts import ErrorCode, ResearchError


def _months(months: Sequence[str]) -> list[str]:
    result = [month_string(str(m)) for m in months]
    if len(set(result)) != len(result):
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate feature base month')
    if not result:
        raise ResearchError(ErrorCode.MISSING_DATA, 'empty feature month list')
    return result


def _frame(values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame) or values.empty:
        raise ResearchError(ErrorCode.MISSING_DATA, 'features require a nonempty monthly DataFrame')
    if values.columns.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate feature column')
    if any(not isinstance(c, str) or not c for c in values.columns):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'feature column names must be nonempty strings')
    months = _months(values.index)
    try:
        result = values.astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ResearchError(ErrorCode.MISSING_DATA, 'feature values must be numeric') from exc
    result.index = pd.Index(months, name='base_month')
    result = result.sort_index()
    index = pd.period_range(result.index[0], result.index[-1], freq='M').astype(str)
    return result.reindex(pd.Index(index, name='base_month')).where(lambda x: np.isfinite(x))


def _codes(metadata: Mapping | pd.Series, columns, *, kind: str, allowed: range) -> dict[str, int]:
    if isinstance(metadata, pd.Series) and metadata.index.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate {kind} metadata')
    result = {}
    for column in columns:
        value = metadata.get(column)
        if isinstance(value, (bool, np.bool_)) or value not in allowed:
            raise ResearchError(ErrorCode.MISSING_DATA, f'missing or invalid {kind}: {column}')
        result[column] = int(value)
    return result


def transform_tcodes(raw: pd.DataFrame, tcodes: Mapping | pd.Series) -> pd.DataFrame:
    """Apply codes 1..7 after inserting absent months, retaining every row.

    Code 7 is the difference of simple growth rates, without pct_change's
    implicit fill. Nonpositive log inputs, zero denominators and all nonfinite
    inputs/results are missing. No statistics are fitted in this function.
    """
    values = _frame(raw)
    codes = _codes(tcodes, values.columns, kind='t-code', allowed=range(1, 8))
    result = {}
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for column, code in codes.items():
            x = values[column]
            if code <= 3:
                y = x
                for _ in range(code - 1):
                    y = y.diff()
            elif code <= 6:
                y = np.log(x.where(x > 0))
                for _ in range(code - 4):
                    y = y.diff()
            else:
                previous = x.shift(1)
                growth = x / previous.where(previous != 0) - 1
                y = growth.where(np.isfinite(growth)).diff()
            result[column] = y.where(np.isfinite(y))
    return pd.DataFrame(result, index=values.index)
