"""Calendar-adjacent transition counts and the paper's literal denominator."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..config import month_string
from ..contracts import ErrorCode, ResearchError
from .probabilities import finite_array


def integer_labels(labels, n_regimes, n_rows=None):
    a = finite_array(labels, ndim=1, name='labels')
    if (type(n_regimes) is not int or n_regimes < 1 or not len(a) or
        (n_rows is not None and len(a) != n_rows) or np.any(a != np.floor(a)) or
        np.any((a < 0) | (a >= n_regimes))):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid regime labels or count')
    return a.astype(np.int64)


def ordered_months(months, n_rows=None):
    values = list(months)
    for m in values: month_string(m)
    if len(set(values)) != len(values):
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'duplicate months in regime rows')
    if not values or values != sorted(values) or n_rows is not None and len(values) != n_rows:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'regime months must be ordered and aligned')
    return values


@dataclass(frozen=True)
class TransitionResult:
    matrix: np.ndarray
    literal_matrix: np.ndarray
    counts: np.ndarray
    occurrences: np.ndarray
    outgoing: np.ndarray
    no_outgoing: np.ndarray
    skipped_gaps: int


def transition_matrices(labels, months, n_regimes=6):
    labels = integer_labels(labels, n_regimes)
    months = ordered_months(months, len(labels))
    ordinal = pd.PeriodIndex(months, freq='M').asi8
    adjacent = np.diff(ordinal) == 1
    counts = np.zeros((n_regimes, n_regimes), dtype=np.int64)
    np.add.at(counts, (labels[:-1][adjacent], labels[1:][adjacent]), 1)
    occurrences = np.bincount(labels, minlength=n_regimes)
    outgoing = counts.sum(axis=1)
    matrix = np.divide(counts, outgoing[:,None], out=np.zeros_like(counts,dtype=float), where=outgoing[:,None]!=0)
    no_outgoing = outgoing == 0
    matrix[no_outgoing, np.flatnonzero(no_outgoing)] = 1.
    literal = np.divide(counts, occurrences[:,None], out=np.zeros_like(counts,dtype=float), where=occurrences[:,None]!=0)
    return TransitionResult(matrix, literal, counts, occurrences, outgoing, no_outgoing, int((~adjacent).sum()))
