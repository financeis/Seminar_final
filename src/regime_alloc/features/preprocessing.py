"""Fit-once feature selection, causal imputation, scaling and SVD PCA.

The pure fitting function accepts arbitrary fit rows (including inner folds).
Only the rolling window adapter imposes the research's 48-row contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd

from ..contracts import (ErrorCode, ResearchError, canonical_id, load_npz,
                         save_npz, write_json)
from ..data.io import read_json
from .transforms import _codes, _frame, _months


def _finite(values, reason):
    if not np.isfinite(values).all():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, reason)


def _digest(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    specs = {name: {'shape': list(a.shape), 'dtype': a.dtype.str,
                   'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
             for name, a in arrays.items()}
    return canonical_id({'metadata': metadata, 'arrays': specs})


def _impute(source: pd.DataFrame, months, medians: np.ndarray, limit: int):
    # source already includes a continuous calendar, so limits count months.
    original = source.reindex(months)
    observed = original.notna()
    forward = source.ffill(limit=limit) if limit else source
    forward = forward.reindex(months)
    ffill = ~observed & forward.notna()
    median = forward.isna()
    imputed = forward.fillna(pd.Series(medians, index=source.columns))
    _finite(imputed.to_numpy(), 'imputation did not produce finite feature values')
    return imputed, observed, ffill, median


@dataclass(frozen=True)
class FeatureResult:
    """Rows stay indexed by their actual macro base months, never target months."""
    imputed: pd.DataFrame
    standardized: pd.DataFrame
    scores: pd.DataFrame
    observed_mask: pd.DataFrame
    ffill_mask: pd.DataFrame
    median_mask: pd.DataFrame
    feature_hash: str
    transform_hash: str
    query_hash: str

    @property
    def counts(self) -> dict[str, int]:
        return {name: int(mask.to_numpy().sum()) for name, mask in
                [('observed', self.observed_mask), ('ffill', self.ffill_mask), ('median', self.median_mask)]}

    @property
    def counts_by_column(self) -> dict[str, dict[str, int]]:
        return {column: {name: int(mask[column].sum()) for name, mask in
                         [('observed', self.observed_mask), ('ffill', self.ffill_mask), ('median', self.median_mask)]}
                for column in self.imputed.columns}


@dataclass(frozen=True)
class FittedPreprocessor:
    """Reusable fitted state. transform never changes selection/statistics/PCA.

    Stored original observations provide causal history for isolated new rows.
    Passing more transformed history supports earlier predictors and gaps after
    fit. Fitted months replay their frozen imputation (also for blocked/LOO
    folds); held-out rows inside the fit span cannot alter training imputation.
    inverse_transform reconstructs selected, t-code-transformed variables in
    their original units, not the raw levels before differencing/log transforms.
    Every calculation, scope check and save verifies the complete fitted-state
    digest. Public metadata/array inspection cannot silently modify computation
    under an existing ID; changing a policy requires fitting a new object.
    """
    metadata: dict
    _arrays: dict[str, np.ndarray]
    feature_hash: str
    transform_hash: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.metadata['columns'])

    @property
    def fit_months(self) -> tuple[str, ...]:
        return tuple(self.metadata['fit_months'])

    @property
    def selection(self) -> list[dict]:
        return self.metadata['selection']

    @property
    def medians(self) -> np.ndarray:
        return self._arrays['medians']

    @property
    def mean(self) -> np.ndarray:
        return self._arrays['mean']

    @property
    def scale(self) -> np.ndarray:
        return self._arrays['scale']

    @property
    def components(self) -> np.ndarray:
        return self._arrays['components']

    @property
    def pca_mean(self) -> np.ndarray:
        return self._arrays['pca_mean']

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Ratios for every numerically nonzero component, in descending order."""
        return self._arrays['explained_variance_ratio']

    @property
    def numerical_rank(self) -> int:
        return self.metadata['numerical_rank']

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def _verify_integrity(self) -> None:
        try:
            valid = (self.metadata.get('feature_hash') == self.feature_hash and
                     self.transform_hash == _digest(self.metadata, self._arrays))
        except (ResearchError, TypeError, ValueError, AttributeError) as exc:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'fitted state changed after fitting; fit a new object') from exc
        if not valid:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'fitted state changed after fitting; fit a new object')

    def require_scope(self, scope: str) -> None:
        self._verify_integrity()
        if self.metadata['scope'] != scope:
            raise ResearchError(ErrorCode.PARTITION_MISMATCH,
                                f"preprocessor scope is {self.metadata['scope']}, expected {scope}")

    def _result(self, imputed, observed, ffill, median) -> FeatureResult:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            z = (imputed.to_numpy() - self.mean) / self.scale
            scores = (z - self.pca_mean) @ self.components.T
        _finite(z, 'standardization overflow or invalid scale')
        _finite(scores, 'nonfinite PCA scores')
        query_hash = _digest({'months': list(imputed.index), 'transform_hash': self.transform_hash},
                             {'imputed': imputed.to_numpy(), 'observed': observed.to_numpy(),
                              'ffill': ffill.to_numpy(), 'median': median.to_numpy()})
        return FeatureResult(imputed, pd.DataFrame(z, index=imputed.index, columns=self.columns),
                             pd.DataFrame(scores, index=imputed.index,
                                          columns=[f'PC{i+1}' for i in range(self.n_components)]),
                             observed, ffill, median, self.feature_hash, self.transform_hash, query_hash)

    @property
    def fit_result(self) -> FeatureResult:
        self._verify_integrity()
        index = pd.Index(self.fit_months, name='base_month')
        frames = [pd.DataFrame(self._arrays[name].copy(), index=index, columns=self.columns)
                  for name in ('fit_imputed', 'fit_observed', 'fit_ffill', 'fit_median')]
        return self._result(*frames)

    def transform(self, transformed: pd.DataFrame, months: Sequence[str] | None = None,
                  *, expected_scope: str | None = None) -> FeatureResult:
        self._verify_integrity()
        if expected_scope is not None:
            self.require_scope(expected_scope)
        values = _frame(transformed)
        requested = _months(transformed.index if months is None else months)
        requested = sorted(requested)
        source = pd.DataFrame(self._arrays['history_values'].copy(),
                              index=self.metadata['history_months'], columns=self.columns)
        source = source.where(self._arrays['history_observed'])
        # New values can supply extra history, but cannot rewrite fitted rows.
        updates = values.reindex(columns=self.columns).drop(index=list(self.fit_months), errors='ignore')
        source = pd.concat([source.drop(index=updates.index, errors='ignore'), updates]).sort_index()
        bounds = [*source.index, *requested]
        source = source.reindex(pd.Index(pd.period_range(min(bounds), max(bounds), freq='M').astype(str), name='base_month'))
        parts = _impute(source, requested, self.medians, self.metadata['ffill_limit'])
        frozen = self.fit_result
        overlap = sorted(set(requested) & set(self.fit_months))
        for result, saved in zip(parts, (frozen.imputed, frozen.observed_mask, frozen.ffill_mask, frozen.median_mask)):
            result.loc[overlap] = saved.loc[overlap]
        return self._result(*parts)

    def inverse_transform(self, scores: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        self._verify_integrity()
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_components:
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'PCA inverse input has an incompatible component dimension')
        _finite(values, 'nonfinite PCA inverse input')
        with np.errstate(over='ignore', invalid='ignore'):
            restored = (values @ self.components + self.pca_mean) * self.scale + self.mean
        _finite(restored, 'PCA inverse overflow')
        return pd.DataFrame(restored, index=scores.index if isinstance(scores, pd.DataFrame) else None,
                            columns=self.columns)

    def save(self, directory: str | Path) -> dict:
        """Write UTF-8 JSON plus finite numeric NPZ, never pickle or overwrite."""
        self._verify_integrity()
        directory = Path(directory)
        if any((directory / name).exists() for name in ('preprocessor.json', 'model.npz')):
            raise ResearchError(ErrorCode.RUN_CONFLICT, f'preprocessor output already exists: {directory}')
        archive = save_npz(directory / 'model.npz', self._arrays)
        manifest = {'metadata': self.metadata, 'feature_hash': self.feature_hash,
                    'transform_hash': self.transform_hash, 'archive': archive}
        write_json(directory / 'preprocessor.json', manifest)
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> FittedPreprocessor:
        directory = Path(directory)
        record = read_json(directory / 'preprocessor.json')
        arrays = load_npz(directory / 'model.npz', record['archive'])
        metadata = record['metadata']
        if metadata.get('version') != 1 or record['transform_hash'] != _digest(metadata, arrays):
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'preprocessor metadata/state hash mismatch')
        if record['feature_hash'] != metadata['feature_hash']:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'feature selection hash mismatch')
        for array in arrays.values():
            array.setflags(write=False)
        return cls(metadata, arrays, record['feature_hash'], record['transform_hash'])


def fit_preprocessor(transformed: pd.DataFrame, fit_months: Sequence[str], groups: Mapping | pd.Series,
                     *, missing_rate: float = .20, ffill_limit: int = 2, pca_variance: float = .95,
                     scope: str = 'rolling', provenance: dict | None = None) -> FittedPreprocessor:
    """Fit on named rows only, with up to ffill_limit earlier context months.

    Medians use original finite fit observations; means and population standard
    deviations use imputed fit rows. Missingness is measured before imputation.
    Fit row order is chronological and retained. Non-fit rows inside the span
    are unavailable to fit, supporting independent inner-fold preprocessing.
    """
    if (not np.isfinite(missing_rate) or not 0 <= missing_rate < 1 or
        not np.isfinite(pca_variance) or not 0 < pca_variance <= 1 or
        type(ffill_limit) is not int or not 0 <= ffill_limit <= 2 or scope not in ('rolling', 'full_sample')):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid preprocessing policy or scope')
    values = _frame(transformed)
    months = sorted(_months(fit_months))
    if len(months) < 2:
        raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'PCA requires at least two fit rows')
    if months[0] < values.index[0] or months[-1] > values.index[-1]:
        raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'fit months exceed supplied history')
    group_codes = _codes(groups, values.columns, kind='group', allowed=range(1, 9))
    fit = values.reindex(months)
    selection = []
    columns = []
    for column in values:
        observed = fit[column].dropna()
        rate = float(fit[column].isna().mean())
        reasons = []
        if group_codes[column] == 6: reasons.append('excluded_group_6')
        if observed.empty: reasons.append('no_finite_observations')
        elif observed.nunique() == 1: reasons.append('constant_observed_values')
        if rate > missing_rate: reasons.append('missing_rate_exceeded')
        selection.append({'column': column, 'group': group_codes[column], 'missing_rate': rate,
                          'observed_count': len(observed), 'selected': not reasons, 'reasons': reasons})
        if not reasons: columns.append(column)
    if not columns:
        raise ResearchError(ErrorCode.MISSING_DATA, 'no valid feature columns', details={'selection': selection})
    # Never use later rows or held-out rows within the fit span as fit context.
    start = str(pd.Period(months[0], freq='M') - ffill_limit)
    history_months = pd.period_range(start, months[-1], freq='M').astype(str).tolist()
    history = values.reindex(index=history_months, columns=columns)
    history.loc[[m for m in history.index if m >= months[0] and m not in months]] = np.nan
    history.index.name = 'base_month'
    medians = fit[columns].median().to_numpy()
    imputed, observed, ffill, median = _impute(history, months, medians, ffill_limit)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mean = imputed.to_numpy().mean(axis=0)
        scale = imputed.to_numpy().std(axis=0, ddof=0)
        _finite(mean, 'fit mean is nonfinite')
        _finite(scale, 'fit scale is nonfinite')
        if np.any(scale <= 0):
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'imputed fit columns have zero scale')
        standardized = (imputed.to_numpy() - mean) / scale
        pca_mean = standardized.mean(axis=0)
        centered = standardized - pca_mean
    _finite(centered, 'nonfinite centered fit matrix')
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'PCA SVD did not converge') from exc
    tolerance = np.finfo(float).eps * max(centered.shape) * singular[0]
    rank = min(int((singular > tolerance).sum()), len(months)-1)
    if rank < 1:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'PCA fit has zero numerical rank')
    energy = np.square(singular[:rank])
    ratios = energy / energy.sum()
    count = min(int(np.searchsorted(np.cumsum(ratios), pca_variance, side='left')) + 1, rank)
    components = vt[:count].copy()
    # Canonicalize each component's sign by its largest absolute loading.
    anchors = np.argmax(np.abs(components), axis=1)
    components *= np.where(components[np.arange(count), anchors] < 0, -1., 1.)[:, None]
    arrays = {'medians': medians, 'mean': mean, 'scale': scale, 'pca_mean': pca_mean,
              'components': components, 'explained_variance_ratio': ratios,
              'singular_values': singular[:rank], 'fit_imputed': imputed.to_numpy(),
              'fit_observed': observed.to_numpy(), 'fit_ffill': ffill.to_numpy(),
              'fit_median': median.to_numpy(), 'history_values': history.fillna(0.).to_numpy(),
              'history_observed': history.notna().to_numpy()}
    metadata = {'version': 1, 'scope': scope, 'columns': columns, 'fit_months': months,
                'history_months': history_months, 'selection': selection, 'missing_rate': float(missing_rate),
                'ffill_limit': ffill_limit, 'pca_variance': float(pca_variance), 'scale_ddof': 0,
                'median_policy': 'original_finite_fit_observations', 'numerical_rank': rank,
                'history_missing_encoding': 'zero_placeholder_with_history_observed_mask',
                'fit_imputation_counts': {column: {'observed': int(observed[column].sum()),
                    'ffill': int(ffill[column].sum()), 'median': int(median[column].sum())} for column in columns},
                'provenance': deepcopy(provenance or {})}
    feature_hash = _digest(metadata, {k: arrays[k] for k in ('history_values', 'history_observed')})
    metadata['feature_hash'] = feature_hash
    transform_hash = _digest(metadata, arrays)
    for array in arrays.values():
        array.setflags(write=False)
    return FittedPreprocessor(metadata, arrays, feature_hash, transform_hash)
