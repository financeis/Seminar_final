"""Deterministic signed-score allocations; ETF weights exclude the CASH leg."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..contracts import TICKERS, ErrorCode, ResearchError
from ._validation import ticker_order, vector

MODELS = ('naive', 'ridge', 'bl', 'mvo')
SIZINGS = ('lo', 'lns', 'los', 'mx')
SELECTION_SIZES = (2, 3, 4)


def modal_regime(probabilities) -> str:
    """Return argmax in numeric R0, R1, ... order, preserving the R0 tie rule."""
    if isinstance(probabilities, dict):
        probabilities = pd.Series(probabilities)
    if isinstance(probabilities, pd.Series):
        labels = probabilities.index.tolist()
        if any(not isinstance(x, str) or not re.fullmatch(r'R(0|[1-9][0-9]*)', x) for x in labels):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'probabilities require R0, R1, ... labels')
        names = tuple(f'R{i}' for i in range(len(labels)))
    else:
        try:
            names = tuple(f'R{i}' for i in range(len(probabilities)))
        except TypeError as exc:
            raise ResearchError(ErrorCode.MISSING_DATA, 'mx requires next-regime probabilities') from exc
    if not names:
        raise ResearchError(ErrorCode.MISSING_DATA, 'empty next-regime probabilities')
    p = vector(probabilities, names, 'regime probabilities')
    if (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'regime probabilities must be nonnegative and sum to one')
    return names[int(np.argmax(p.to_numpy()))]


def allocate(scores, sizing: str, selection_size: int, *, next_probabilities=None,
             tickers: tuple[str, ...] = TICKERS) -> pd.Series:
    """Normalize selected signed scores by their absolute sum.

    lo selects positive top-l, lns positive top-l plus negative bottom-l,
    los absolute top-l, and mx uses los when R0 is most probable, else lo.
    Ties use the explicit ticker order (the research default is TICKERS).
    Zero selections are all cash. Cash itself is 1 - sum(ETF weights).
    """
    names = ticker_order(tickers)
    if (sizing not in SIZINGS or isinstance(selection_size, bool)
            or not isinstance(selection_size, (int, np.integer)) or selection_size not in SELECTION_SIZES):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'unknown sizing or selection size (requires 2, 3, or 4)')
    values = vector(scores, names, 'scores').to_numpy()
    method = ('los' if modal_regime(next_probabilities) == 'R0' else 'lo') if sizing == 'mx' else sizing
    indices = range(len(values))
    if method == 'los':
        selected = sorted((i for i in indices if values[i] != 0), key=lambda i: (-abs(values[i]), i))[:selection_size]
    else:
        selected = sorted((i for i in indices if values[i] > 0), key=lambda i: (-values[i], i))[:selection_size]
        if method == 'lns':
            selected += sorted((i for i in indices if values[i] < 0), key=lambda i: (values[i], i))[:selection_size]
    weights = np.zeros(len(names))
    if selected:
        # Rescale first so finite scores near float max cannot overflow the sum.
        chosen = values[selected] / np.max(np.abs(values[selected]))
        weights[selected] = chosen / np.sum(np.abs(chosen))
    return pd.Series(weights, index=list(names), name='base_weight')


def benchmark_weights(name: str, *, tickers: tuple[str, ...] = TICKERS) -> pd.Series:
    """Monthly SPY 100% or equal-weight target across the complete panel."""
    names = ticker_order(tickers)
    if name == 'spy' and 'SPY' in names:
        weights = [float(t == 'SPY') for t in names]
    elif name == 'ew':
        weights = [1 / len(names)] * len(names)
    else:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'unknown benchmark or missing SPY')
    return pd.Series(weights, index=list(names), name='base_weight')


def strategy_ids() -> tuple[str, ...]:
    return tuple(f'{model}_{sizing}_{size}' for model in MODELS for sizing in SIZINGS for size in SELECTION_SIZES) + ('spy', 'ew')


def build_base_weights(model_scores: dict, *, next_probabilities, tickers: tuple[str, ...] = TICKERS) -> dict[str, pd.Series]:
    """Build exactly the 48 research strategies and the two monthly benchmarks."""
    if set(model_scores) != set(MODELS):
        raise ResearchError(ErrorCode.MISSING_DATA, f'scores require exactly {MODELS}')
    result = {f'{model}_{sizing}_{size}': allocate(model_scores[model], sizing, size,
        next_probabilities=next_probabilities, tickers=tickers)
        for model in MODELS for sizing in SIZINGS for size in SELECTION_SIZES}
    result.update({name: benchmark_weights(name, tickers=tickers) for name in ('spy', 'ew')})
    return result
