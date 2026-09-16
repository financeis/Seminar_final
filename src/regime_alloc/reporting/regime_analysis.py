"""Ex-post full-sample diagnostics, isolated from every trading window.

The paper's Figure 1--6 / Table 1 methods are re-executed on an explicit public
snapshot. Neither NBER labels nor the post-hoc indicators enter model fitting.
Optional dependencies are imported only when their computation is requested.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import (SCHEMA_VERSION, ErrorCode, ResearchError, canonical_id,
                         save_npz, load_npz, write_json)
from ..data.calendar import month_range, add_months
from ..data.io import read_json
from ..features import FittedPreprocessor, fit_preprocessor, transform_tcodes
from ..regimes import (fit_partition, spherical_kmeans, regime_probabilities,
                       transition_matrices)
from ..regimes.transitions import integer_labels, ordered_months


PAPER_INDICATORS = ('RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL', 'S&P 500')
INDICATOR_NAMES = {
    'RPI': 'Real personal income / 실질 개인소득',
    'UNRATE': 'Unemployment rate / 실업률',
    'UMCSENTx': 'Consumer sentiment / 소비자 심리',
    'FEDFUNDS': 'Federal funds rate / 연방기금금리',
    'CPIAUCSL': 'Consumer price index / 소비자물가지수',
    'S&P 500': 'S&P 500 index / 주가지수',
}
TCODE_UNITS = {1: 'source level', 2: 'first difference of source level',
               3: 'second difference of source level', 4: 'natural log of source level',
               5: 'first difference of natural log', 6: 'second difference of natural log',
               7: 'first difference of fractional growth rate'}


@dataclass(frozen=True)
class AnalysisSettings:
    start_month: str = '1959-12'
    end_month: str = '2023-01'
    missing_rate: float = .20
    ffill_limit: int = 2
    pca_variance: float = .95
    seed: int = 0
    n_init: int = 20
    max_iter: int = 300
    tol: float = 1e-6
    gmm_reg_covar: float = 1e-6
    gmm_n_init: int = 20
    gmm_max_iter: int = 300
    gmm_tol: float = 1e-3
    indicators: tuple[str, ...] = PAPER_INDICATORS

    def validate(self):
        months = month_range(self.start_month, self.end_month)
        if len(months) < 12:
            raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'full-sample analysis requires at least 12 months')
        if (any(type(v) is not int or v < 1 for v in
                (self.n_init, self.max_iter, self.gmm_n_init, self.gmm_max_iter)) or
                type(self.seed) is not int or self.seed < 0 or
                any(not np.isfinite(v) or v <= 0 for v in
                    (self.tol, self.gmm_reg_covar, self.gmm_tol)) or
                not self.indicators or len(set(self.indicators)) != len(self.indicators) or
                any(not isinstance(x, str) or not x for x in self.indicators)):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid full-sample analysis settings')
        if (not np.isfinite(self.missing_rate) or not 0 <= self.missing_rate < 1 or
                type(self.ffill_limit) is not int or not 0 <= self.ffill_limit <= 2 or
                not np.isfinite(self.pca_variance) or not 0 < self.pca_variance <= 1):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid full-sample preprocessing settings')


def _array_spec(arrays):
    return {name: {'shape': list(a.shape), 'dtype': a.dtype.str,
                   'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
            for name, a in arrays.items()}


def _digest(metadata, arrays):
    return canonical_id({'metadata': metadata, 'arrays': _array_spec(arrays)})


def conditional_departures(matrix):
    """Return P(next=j | next != current=i), and no-departure row flags."""
    e = np.asarray(matrix, dtype=float)
    if (e.ndim != 2 or e.shape[0] != e.shape[1] or not len(e) or
            not np.isfinite(e).all() or np.any(e < 0) or np.any(e > 1) or
            not np.allclose(e.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'transition matrix must be finite and row-stochastic')
    remaining = 1-np.diag(e)
    no_departure = remaining == 0
    result = e.copy()
    np.fill_diagonal(result, 0)
    result = np.divide(result, remaining[:, None], out=np.zeros_like(result),
                       where=~no_departure[:, None])
    if not np.allclose(result[~no_departure].sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'conditional departure probabilities do not sum to one')
    # A single possible destination can round to 1 + machine epsilon.
    return np.clip(result, 0, 1), no_departure


def row_minmax(values):
    """Scale each indicator across regimes. NaN means absent, not a zero."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or np.isinf(a).any():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, 'indicator statistics must be a finite-or-missing matrix')
    result = np.full_like(a, np.nan)
    constant = np.zeros(len(a), dtype=bool)
    for i, row in enumerate(a):
        observed = np.isfinite(row)
        if not observed.any():
            continue
        low, high = row[observed].min(), row[observed].max()
        constant[i] = high == low
        result[i, observed] = 0 if constant[i] else (row[observed]-low)/(high-low)
    return result, constant


def summarize_indicators(frame, labels, *, indicators=PAPER_INDICATORS, n_regimes=6):
    """Observed-value means only; no post-hoc interpolation or invented series."""
    labels = integer_labels(labels, n_regimes, len(frame))
    means = np.full((len(indicators), n_regimes), np.nan)
    counts = np.zeros(means.shape, dtype=np.int64)
    for i, name in enumerate(indicators):
        if name not in frame:
            continue
        x = frame[name].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise ResearchError(ErrorCode.NONFINITE_VALUE, f'infinite indicator: {name}')
        for j in range(n_regimes):
            observed = x[(labels == j) & np.isfinite(x)]
            counts[i, j] = len(observed)
            if len(observed):
                means[i, j] = observed.mean()
    normalized, constant = row_minmax(means)
    records = []
    for i, name in enumerate(indicators):
        available = bool(counts[i].sum())
        reason = '' if available else 'series_absent' if name not in frame else 'no_observed_values'
        cells = [{'regime': f'R{j}', 'observed_count': int(counts[i, j]),
                  'value': float(means[i, j]) if counts[i, j] else None,
                  'normalized': float(normalized[i, j]) if counts[i, j] else None,
                  'reason': '' if counts[i, j] else 'no_observed_values'} for j in range(n_regimes)]
        records.append({'series': name, 'status': 'ok' if available else 'unavailable',
                        'reason': reason, 'constant': bool(constant[i]), 'cells': cells})
    return {'means': means.tolist(), 'counts': counts.tolist(), 'normalized': normalized.tolist(),
            'constant': constant.tolist(), 'records': records}


def fit_gmm_comparison(scores, labels, *, settings=None):
    """Fit on exactly the same PCA rows; align labels by empirical centroid distance.

    The assignment uses no recession labels. The component displayed as R0 is
    an alignment to the layered method's outlier group, not a recession truth.
    """
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import linear_sum_assignment
    from threadpoolctl import threadpool_limits
    import warnings
    settings = settings or AnalysisSettings()
    settings.validate()
    x = np.asarray(scores, dtype=float)
    labels = integer_labels(labels, 6, len(x))
    if (x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all() or
            len(np.unique(labels)) != 6 or len(np.unique(x, axis=0)) < 6):
        raise ResearchError(ErrorCode.DEGENERATE_PARTITION, 'GMM comparison needs six populated reference regimes')
    model = GaussianMixture(n_components=6, covariance_type='full', reg_covar=settings.gmm_reg_covar,
                            n_init=settings.gmm_n_init, max_iter=settings.gmm_max_iter,
                            tol=settings.gmm_tol, random_state=settings.seed, init_params='kmeans')
    try:
        with threadpool_limits(1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x)
        if not model.converged_:
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'GMM did not converge; no comparison is published')
        probability = model.predict_proba(x)
        if (not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1) or
                not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0) or
                not np.isfinite(model.lower_bound_) or np.any(model.weights_ <= 0)):
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'invalid GMM probability or fitted state')
        for cov in model.covariances_:
            np.linalg.cholesky(cov)
    except (ValueError, np.linalg.LinAlgError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'GMM fit failed: {exc}') from exc
    # Empirical centroids, not unit spherical centers, share the GMM mean units.
    reference_means = np.vstack([x[labels == i].mean(axis=0) for i in range(6)])
    cost = np.linalg.norm(reference_means[:, None]-model.means_[None, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    order = cols[np.argsort(rows)]
    aligned = probability[:, order]
    arrays = {'gmm_probabilities': aligned, 'gmm_labels': aligned.argmax(axis=1).astype(np.int64),
              'gmm_weights': model.weights_[order], 'gmm_means': model.means_[order],
              'gmm_covariances': model.covariances_[order],
              'gmm_precisions_cholesky': model.precisions_cholesky_[order],
              'gmm_reference_means': reference_means, 'gmm_alignment_cost': cost,
              'gmm_component_order': order.astype(np.int64)}
    metadata = {'n_components': 6, 'covariance_type': 'full', 'reg_covar': settings.gmm_reg_covar,
                'seed': settings.seed, 'n_init': settings.gmm_n_init, 'max_iter': settings.gmm_max_iter,
                'tol': settings.gmm_tol, 'init_params': 'kmeans', 'converged': True,
                'iterations': int(model.n_iter_), 'lower_bound': float(model.lower_bound_),
                'alignment': 'Hungarian Euclidean distance: empirical PCA regime means to GMM means',
                'alignment_ties': 'scipy linear_sum_assignment deterministic input order',
                'nber_used_in_fit_or_alignment': False,
                'r0_meaning': 'GMM component aligned to layered outlier group; not a known crisis label',
                'warnings': [str(w.message) for w in caught]}
    return arrays, metadata


def _nber_rows(nber, months):
    if not isinstance(nber, pd.DataFrame) or not {'month', 'usrec'} <= set(nber):
        raise ResearchError(ErrorCode.MISSING_DATA, 'NBER requires month/usrec columns')
    ordered_months(nber['month'].tolist())
    if not nber.usrec.isin([0, 1]).all():
        raise ResearchError(ErrorCode.MISSING_DATA, 'NBER indicator must be binary')
    selected = nber.set_index('month').reindex(months)['usrec']
    if selected.isna().any():
        raise ResearchError(ErrorCode.MISSING_DATA, 'NBER does not cover all analysis months')
    return selected.to_numpy(dtype=np.int64)


def _indicator_metadata(raw, transformed, labels, settings, codes, groups, series_metadata):
    metadata, arrays = {}, {}
    for units, frame in [('raw', raw), ('tcode', transformed)]:
        summary = summarize_indicators(frame, labels, indicators=settings.indicators)
        for field in ('means', 'normalized'):
            values = np.asarray(summary[field], dtype=float)
            arrays[f'indicator_{units}_{field}'] = np.nan_to_num(values, nan=0)
            arrays[f'indicator_{units}_{field}_observed'] = np.isfinite(values)
        arrays[f'indicator_{units}_counts'] = np.asarray(summary['counts'], dtype=np.int64)
        arrays[f'indicator_{units}_constant'] = np.asarray(summary['constant'], dtype=bool)
        for row in summary['records']:
            name = row['series']
            row['paper_figure_label'] = name
            row['description'] = INDICATOR_NAMES.get(name, name)
            row['source_metadata'] = deepcopy(series_metadata.get(name, {}))
            row['role'] = 'ex_post_interpretation_only'
            row['group6_excluded_from_model'] = bool(name in groups and groups[name] == 6)
            row['unit'] = ('original source units; base year/scale not inferred' if units == 'raw'
                           else TCODE_UNITS[int(codes[name])] if name in codes else 'series_absent')
            if name in codes:
                row['tcode'] = int(codes[name])
        metadata[units] = {'records': summary['records'], 'aggregation': 'mean of observed values within each regime; no imputation',
                           'normalization': 'row min-max across regime means; constant observed rows display zero',
                           'missing_cells': 'null plus reason in JSON; zero storage placeholder plus observed mask in NPZ'}
    metadata['figure4_units'] = 'raw'
    metadata['limitations'] = ['Paper Figure 4 does not unambiguously specify raw versus transformed units.',
                               'Both are preserved; raw levels can reflect long-term price/income trends.',
                               'FEDFUNDS and other group 6 series are post-hoc indicators only, never model inputs.',
                               'Indicator names are matched exactly; absent series are never substituted.']
    return metadata, arrays


def _table1(indicators, counts):
    records = []
    for regime in range(6):
        ordered = []
        for item in indicators['raw']['records']:
            cell = item['cells'][regime]
            if cell['normalized'] is not None and not item['constant']:
                ordered.append((cell['normalized'], item['series']))
        ordered.sort()
        low = ordered[0][1] if ordered else 'available indicator absent'
        high = ordered[-1][1] if ordered else 'available indicator absent'
        records.append({'regime': f'R{regime}', 'sample_count': int(counts[regime]),
                        'description': f'Within this sample: relatively high {high}; relatively low {low}.',
                        'basis': 'extremes of row min-max raw indicator means; compare numeric Table 1 evidence',
                        'economic_label_status': 'descriptive_only_not_ground_truth',
                        'group_definition': ('smaller first-stage L2 cluster; not NBER recession ground truth'
                                             if regime == 0 else 'spherical cluster within first-stage normal sample'),
                        'sample_limitation': ('single observed month; no evidence of a persistent economic state'
                                              if counts[regime] == 1 else 'sample-specific descriptive association'),
                        'outlier_group': regime == 0})
    return records


def _outlier_diagnostics(fit, labels):
    features = fit.fit_result
    results = []
    for row in np.flatnonzero(labels == 0):
        z = features.standardized.iloc[row]
        order = sorted(range(len(z)), key=lambda i: (-abs(float(z.iloc[i])), str(z.index[i])))[:10]
        extremes = []
        for i in order:
            extremes.append({'series': str(z.index[i]), 'standardized_value': float(z.iloc[i]),
                              'transformed_imputed_value': float(features.imputed.iloc[row, i]),
                              'was_observed': bool(features.observed_mask.iloc[row, i]),
                              'was_forward_filled': bool(features.ffill_mask.iloc[row, i]),
                              'was_median_filled': bool(features.median_mask.iloc[row, i])})
        results.append({'month': str(z.name), 'standardized_extremes': extremes,
                        'interpretation': 'largest absolute fitted z-scores, not causal explanations or tuned labels'})
    return results


@dataclass(frozen=True)
class FullSampleAnalysis:
    """Reloadable full-sample state. Public views are copies and IDs are checked."""
    _metadata: dict
    _arrays: dict[str, np.ndarray]
    preprocessor: FittedPreprocessor
    analysis_id: str

    def _verify(self):
        self.preprocessor.require_scope('full_sample')
        if (self._metadata.get('scope') != 'full_sample' or
                self._metadata.get('transform_hash') != self.preprocessor.transform_hash or
                self.analysis_id != _digest(self._metadata, self._arrays)):
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'full-sample analysis state or scope changed')

    @property
    def metadata(self):
        self._verify()
        return deepcopy(self._metadata)

    @property
    def arrays(self):
        self._verify()
        return {key: value.copy() for key, value in self._arrays.items()}

    @property
    def model_id(self):
        self._verify()
        return self._metadata['model_id']

    def require_scope(self, scope):
        self._verify()
        if scope != 'full_sample':
            raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'full_sample analysis scope cannot be used for rolling investment')

    def save(self, directory):
        self._verify()
        directory = Path(directory)
        if directory.exists():
            raise ResearchError(ErrorCode.RUN_CONFLICT, f'analysis output already exists: {directory}')
        directory.mkdir(parents=True)
        self.preprocessor.save(directory/'preprocessor')
        archive = save_npz(directory/'analysis.npz', self._arrays)
        record = {'analysis_id': self.analysis_id, 'metadata': self._metadata, 'archive': archive}
        write_json(directory/'analysis.json', record)
        return record

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = read_json(directory/'analysis.json')
        arrays = load_npz(directory/'analysis.npz', record['archive'])
        result = cls(record['metadata'], arrays, FittedPreprocessor.load(directory/'preprocessor'), record['analysis_id'])
        result._verify()
        return result


def fit_full_sample(raw, tcodes, groups, nber, *, settings=None, provenance=None, series_metadata=None):
    """Pure calculation from supplied snapshot data. No file access or download.

    Defaults cover 1959-12 through 2023-01. Earlier raw rows provide difference
    context. Settings permit a named smaller diagnostic sample for tests.
    """
    from threadpoolctl import threadpool_limits
    from importlib.metadata import version
    settings = settings or AnalysisSettings()
    settings.validate()
    months = month_range(settings.start_month, settings.end_month)
    ordered_months(raw.index.tolist())
    context_months = month_range(add_months(months[0], -2), months[-1])
    if not set(context_months) <= set(raw.index):
        raise ResearchError(ErrorCode.CALENDAR_GAP, 'full sample needs every month and two preceding raw context months')
    # Validate ex-post coverage before expensive fitting. Values are not passed below.
    nber_values = _nber_rows(nber, months)
    raw = raw.loc[:settings.end_month].copy()
    transformed = transform_tcodes(raw, tcodes)
    fit = fit_preprocessor(transformed, months, groups, missing_rate=settings.missing_rate,
                           ffill_limit=settings.ffill_limit, pca_variance=settings.pca_variance,
                           scope='full_sample', provenance=provenance or {'source': 'caller_supplied'})
    scores = fit.fit_result.scores.to_numpy()
    algorithm = dict(seed=settings.seed, n_init=settings.n_init,
                     max_iter=settings.max_iter, tol=settings.tol)
    with threadpool_limits(1):
        partition = fit_partition(scores, k_normal=5, **algorithm)
        normal = scores[partition.labels != 0]
        elbow_results = [spherical_kmeans(normal, k, **algorithm) for k in range(1, 11)]
        probabilities = regime_probabilities(scores, partition.stage1_centers, partition.centers[1:])
        gmm_arrays, gmm_meta = fit_gmm_comparison(scores, partition.labels, settings=settings)
    transitions = transition_matrices(partition.labels, months)
    conditional, no_departure = conditional_departures(transitions.matrix)
    arrays = {'scores': scores, 'labels': partition.labels, 'centers': partition.centers,
              'stage1_centers': partition.stage1_centers, 'probabilities': probabilities,
              'transition_matrix': transitions.matrix, 'literal_transition_matrix': transitions.literal_matrix,
              'transition_counts': transitions.counts, 'occurrences': transitions.occurrences,
              'outgoing': transitions.outgoing, 'no_outgoing': transitions.no_outgoing,
              'conditional_departures': conditional, 'no_departure': no_departure,
              'explained_variance_ratio': fit.explained_variance_ratio, **gmm_arrays}
    elbow = []
    for k, result in enumerate(elbow_results, start=1):
        arrays[f'elbow_k{k}_labels'] = result.labels
        arrays[f'elbow_k{k}_centers'] = result.centers
        elbow.append({'k_normal': k, 'cosine_inertia': result.inertia,
                       'n_observations': len(normal), 'iterations': result.iterations,
                       'diagnostics': result.diagnostics})
    input_hash = _digest({'months': months, 'scope': 'full_sample', 'transform_hash': fit.transform_hash}, {'scores': scores})
    gmm_meta['input_hash'] = input_hash
    partition_meta = {**partition.diagnostics, 'input_hash': input_hash}
    model_meta = {'scope': 'full_sample', 'feature_hash': fit.feature_hash, 'transform_hash': fit.transform_hash,
                  'partition': partition_meta, 'gmm': gmm_meta, 'months': months,
                  'library_versions': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}}
    model_id = _digest(model_meta, arrays)
    indicators, indicator_arrays = _indicator_metadata(raw.loc[months], transformed.loc[months], partition.labels,
                                                       settings, tcodes, groups, series_metadata or {})
    arrays.update(indicator_arrays)
    arrays['nber_usrec'] = nber_values
    metadata = {**model_meta, 'schema_version': SCHEMA_VERSION, 'model_id': model_id,
                'settings': asdict(settings), 'regime_ids': [f'R{i}' for i in range(6)],
                'selected_k_normal': 5, 'pca_components': fit.n_components,
                'sample_count': len(months), 'selected_features': list(fit.columns),
                'provenance': provenance or {'source': 'caller_supplied'},
                'elbow': elbow, 'indicators': indicators, 'table1': _table1(indicators, transitions.occurrences),
                'outlier_diagnostics': _outlier_diagnostics(fit, partition.labels),
                'nber_purpose': 'ex_post_shading_only; excluded from features, fit, tuning and alignment',
                'transition_orientation': 'row=origin; column=destination',
                'literal_transition_denominator': 'all regime occurrences including terminal month',
                'primary_transition_denominator': 'observed outgoing transitions',
                'no_departure_policy': 'zero row with no_departure flag; not a conditional distribution',
                'skipped_calendar_gaps': transitions.skipped_gaps,
                'reproduction_kind': 'method_reexecution; no assertion of matching paper numerical results',
                'paper_reference': {'pdf_sha256': '1dfd2574208a52effd3fa195e6005e33122b54f19220b4fb974cae603fc1aaab',
                                    'pdf_pages': [8, 9, 10, 11, 12, 13]},
                'paper_links': {'Figure 1': 'explained_variance_ratio', 'Figure 2': 'labels,gmm_labels,nber_usrec',
                               'Figure 3': 'probabilities,gmm_probabilities,nber_usrec',
                               'Figure 4': 'indicators.raw', 'Figure 5': 'transition_matrix,conditional_departures',
                               'Figure 6': 'conditional_departures', 'Table 1': 'table1'},
                'limitations': ['Full-sample fits use later observations and are descriptive, not tradable forecasts.',
                                'Regime numbers and descriptions are sample-dependent, not fixed economic truths.',
                                'Public snapshot variable count/PCA rank can differ from the paper.',
                                'Figure 6 uses a fixed circular layout without forcing the paper path graph.']}
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ResearchError(ErrorCode.NONFINITE_VALUE, 'nonfinite full-sample model state')
        array.setflags(write=False)
    return FullSampleAnalysis(metadata, arrays, fit, _digest(metadata, arrays))


def analyze_fixed_snapshot(config, *, settings=None):
    """Offline convenience adapter, resolving only explicitly pinned snapshots."""
    from ..data import load_macro_vintage, load_nber
    data = config.data
    if data.fixed_vintage != '2023-02' or not data.fred_dataset_id or not data.nber_dataset_id:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'full sample requires pinned FRED/NBER IDs and fixed 2023-02 vintage')
    vintage = load_macro_vintage(data.root, data.fixed_vintage, dataset_id=data.fred_dataset_id)
    nber = load_nber(data.root, dataset_id=data.nber_dataset_id)
    result = fit_full_sample(vintage.values, vintage.tcodes, vintage.groups, nber,
                             settings=settings, series_metadata=vintage.metadata,
                             provenance={'fred_dataset_id': vintage.dataset_id, 'vintage_month': vintage.vintage_month})
    metadata = result.metadata
    metadata['nber_dataset_id'] = data.nber_dataset_id
    return FullSampleAnalysis(metadata, result._arrays, result.preprocessor, _digest(metadata, result._arrays))
