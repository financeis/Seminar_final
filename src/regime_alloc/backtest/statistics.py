"""Paired exploratory comparisons; shared circular blocks preserve co-movement.

The resampling units are months from one market history, not independent data
sets. Paired t is greater-sided, standard two-method Nemenyi is two-sided.
Its infinite-df studentized-range tail has the exact stable form 2*Normal.sf(z).
Main uncertainty is conditional on the retained membership seeds; it does not
combine seed variability with independent time-series sampling uncertainty.
"""
from __future__ import annotations
import math
import numpy as np
from ..contracts import SCHEMA_VERSION, ErrorCode, ResearchError
from ..data.calendar import month_range

METRICS = ('sharpe', 'sortino', 'maxdd', 'positive_ratio')
PAIR_NAMES = {'naive_vs_random': ('naive_lo_2', 'random_naive_lo_2'),
              'ridge_vs_random': ('ridge_lo_2', 'random_ridge_lo_2'),
              'bl_vs_mvo': ('bl_lo_2', 'mvo_lo_2')}


def holm(pvalues):
    """Keep the planned family size: null inputs occupy p=1 positions."""
    raw = np.array([1. if p is None else p for p in pvalues], dtype=float)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'p values must be probabilities')
    order = np.argsort(raw, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(raw[order]*np.arange(len(raw), 0, -1)))
    restored = np.empty(len(raw)); restored[order] = adjusted
    return [None if p is None else float(restored[i]) for i, p in enumerate(pvalues)]


def paired_tests(treatment, control):
    # SciPy's import may inspect the OS with a subprocess on Windows. Keep
    # that initialization inside explicit statistical execution, never import.
    from scipy.stats import t as student_t, norm
    if len(treatment) != len(control):
        raise ResearchError(ErrorCode.MISSING_DATA, 'paired metric lengths differ')
    valid, excluded = [], []
    for i, (a, b) in enumerate(zip(treatment, control)):
        if a is None or b is None:
            excluded.append({'pair_index': i, 'reason': 'undefined_treatment_or_control_metric'})
        elif not math.isfinite(a) or not math.isfinite(b):
            raise ResearchError(ErrorCode.NONFINITE_VALUE, 'paired metric contains nonfinite value')
        else:
            valid.append((a, b))
    n = len(valid)
    reason = 'fewer_than_two_pairs' if n < 2 else ''
    if not reason:
        values = np.asarray(valid); difference = values[:, 0]-values[:, 1]
        if np.ptp(difference) == 0:
            reason = 'constant_metric_differences'
    result = {'n_pairs': n, 'excluded_pairs': excluded}
    if reason:
        result.update(paired_t={'statistic': None, 'p': None, 'reason': reason, 'alternative': 'greater'},
                      nemenyi={'statistic': None, 'p': None, 'reason': reason, 'alternative': 'two-sided'})
        return result
    statistic = float(difference.mean()/(difference.std(ddof=1)/math.sqrt(n)))
    ranks_t = np.where(difference > 0, 1., np.where(difference < 0, 2., 1.5))
    ranks_c = 3-ranks_t
    rank_difference = float(ranks_c.mean()-ranks_t.mean())
    z = rank_difference*math.sqrt(n)  # sqrt(k*(k+1)/(6*N)), k=2.
    result.update(paired_t={'statistic': statistic, 'p': float(student_t.sf(statistic, n-1)),
                            'df': n-1, 'alternative': 'greater', 'reason': ''},
                  nemenyi={'statistic': abs(z)*math.sqrt(2), 'z': z, 'p': float(2*norm.sf(abs(z))),
                           'treatment_mean_rank': float(ranks_t.mean()), 'control_mean_rank': float(ranks_c.mean()),
                           'rank_difference_control_minus_treatment': rank_difference,
                           'methods': 2, 'df': 'infinite', 'alternative': 'two-sided', 'reason': ''})
    return result


def circular_blocks(n_months, block_months, repetitions, *, seed):
    if any(type(x) is not int or x <= 0 for x in (n_months, block_months, repetitions)) or type(seed) is not int or seed < 0:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'positive block dimensions and nonnegative integer seed required')
    starts = np.random.default_rng(seed).integers(0, n_months, size=(repetitions, math.ceil(n_months/block_months)))
    return ((starts[..., None]+np.arange(block_months)) % n_months).reshape(repetitions, -1)[:, :n_months]


def _metrics(panel):
    """Vectorized common metric conventions over the final (monthly) axis.

    Bootstrap indices are unordered calendar samples, so compute_metrics's
    consecutive-calendar input check does not apply. Sortino uses all-month
    downside RMS; MaxDD starts at NAV1; Sharpe uses sample standard deviation.
    """
    x = np.asarray(panel, dtype=float)
    if not np.isfinite(x).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, 'return panel contains nonfinite values')
    if (x <= -1).any():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'nonpositive monthly growth')
    n = x.shape[-1]
    mean = x.mean(axis=-1)
    std = x.std(axis=-1, ddof=1) if n >= 2 else np.zeros_like(mean)
    constant = np.ptp(x, axis=-1) == 0
    downside = np.sqrt(np.mean(np.minimum(x, 0)**2, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        sharpe = np.where((std > 0) & ~constant & (n >= 2), mean/std*math.sqrt(12), np.nan)
        sortino = np.where((downside > 0) & (n >= 2), mean/downside*math.sqrt(12), np.nan)
        wealth = np.cumprod(1+x, axis=-1)
    if not np.isfinite(wealth).all() or (wealth <= 0).any():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'nonfinite or nonpositive cumulative wealth')
    peaks = np.maximum.accumulate(np.concatenate((np.ones((*x.shape[:-1], 1)), wealth), axis=-1), axis=-1)[..., 1:]
    return {'sharpe': sharpe, 'sortino': sortino, 'maxdd': (wealth/peaks-1).min(axis=-1),
            'positive_ratio': (x > 0).mean(axis=-1)}


def _value(value, reason='undefined_metric_denominator_or_insufficient_months'):
    return {'value': float(value), 'reason': ''} if np.isfinite(value) else {'value': None, 'reason': reason}


def _interval(draws):
    valid = [x['value'] for x in draws if x['value'] is not None]
    reason = '' if valid else 'no_defined_resampled_differences'
    bounds = np.quantile(valid, [.025, .975]) if valid else [None, None]
    return {'lower': float(bounds[0]) if valid else None, 'upper': float(bounds[1]) if valid else None,
            'confidence': .95, 'valid_draws': len(valid), 'excluded_draws': len(draws)-len(valid),
            'reason': reason, 'draws': draws}


def compare_profile(profile, months, pairs, *, seeds, repetitions=1000,
                    block_lengths=(12, 6, 24), seed=0, auxiliary_seed=1):
    """Return strict-JSON records plus original series and shared index manifests.

    pairs maps the three fixed pair IDs to (monthly treatment, seed x month
    control). BL/MVO has one control row; auxiliary index i selects seed i for
    the random pairs and the same MVO row for BL/MVO.
    """
    if set(pairs) != set(PAIR_NAMES) or list(seeds) != list(range(len(seeds))) or not seeds:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'all three pairs and contiguous seeds starting at zero required')
    if not months or list(months) != month_range(months[0], months[-1]):
        raise ResearchError(ErrorCode.CALENDAR_GAP, 'statistics require consecutive monthly returns')
    if len(set(block_lengths)) != len(block_lengths):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'duplicate bootstrap block lengths')
    indices = {str(b): circular_blocks(len(months), b, repetitions, seed=seed) for b in block_lengths}
    auxiliary = circular_blocks(len(months), block_lengths[0], len(seeds), seed=auxiliary_seed)
    result = {'schema_version': SCHEMA_VERSION, 'profile': profile,
              'bootstrap_indices': {k: v.tolist() for k, v in indices.items()},
              'auxiliary_indices': auxiliary.tolist(), 'comparisons': [],
              'paired_monthly_returns': {}, 'original_series': {},
              'seed_manifest': {'control_seeds': list(seeds), 'bootstrap_seed': seed, 'auxiliary_seed': auxiliary_seed,
                                'block_lengths': list(block_lengths), 'repetitions': repetitions,
                                'auxiliary_repetitions': len(seeds), 'index_policy': 'same_month_indices_jointly_all_pairs_and_all_seeds'}}
    for pair_id, (treatment, control) in pairs.items():
        t, c = np.asarray(treatment, dtype=float), np.asarray(control, dtype=float)
        required_controls = 1 if pair_id == 'bl_vs_mvo' else len(seeds)
        if t.shape != (len(months),) or c.shape != (required_controls, len(months)):
            raise ResearchError(ErrorCode.MISSING_DATA, 'comparison panel dimensions do not cover planned seeds/months')
        original_t, original_c = _metrics(t), _metrics(c)
        result['original_series'][pair_id] = {'treatment': t.tolist(), 'controls': c.tolist(), 'months': list(months)}
        result['paired_monthly_returns'][pair_id] = (t-c.mean(axis=0)).tolist()
        sampled = {}
        for length, index in indices.items():
            # Bound work to one draw at a time: 100 x 239, not 1000 x 100 x 239.
            values = {m: [] for m in METRICS}
            for draw_index in index:
                tm, cm = _metrics(t[draw_index]), _metrics(c[:, draw_index])
                for metric in METRICS:
                    valid = np.isfinite(cm[metric]); valid_n = int(valid.sum())
                    diff = tm[metric]-cm[metric][valid].mean() if valid_n else np.nan
                    values[metric].append({**_value(diff), 'valid_controls': valid_n,
                        'excluded_controls': [{'seed': seeds[i], 'reason': 'undefined_control_metric_denominator'}
                                              for i in np.flatnonzero(~valid)]})
            sampled[length] = values
        aux_t, aux_c = {m: [] for m in METRICS}, {m: [] for m in METRICS}
        for i, index in enumerate(auxiliary):
            tm, cm = _metrics(t[index]), _metrics(c[0 if required_controls == 1 else i, index])
            for metric in METRICS:
                aux_t[metric].append(_value(tm[metric])); aux_c[metric].append(_value(cm[metric]))
        for metric in METRICS:
            tests = paired_tests([x['value'] for x in aux_t[metric]], [x['value'] for x in aux_c[metric]])
            row = {'schema_version': SCHEMA_VERSION, 'profile': profile, 'pair_id': pair_id, 'metric': metric,
                   'treatment': PAIR_NAMES[pair_id][0], 'control': PAIR_NAMES[pair_id][1],
                   'direction': 'higher_is_better_including_less_negative_maxdd', **tests,
                   'bootstrap_difference_ci': {length: _interval(values[metric]) for length, values in sampled.items()},
                   'seed_distribution': [{'seed': seeds[i] if pair_id != 'bl_vs_mvo' else 'not_random', **_value(value)} for i, value in enumerate(original_c[metric])],
                   'treatment_original_metric': _value(original_t[metric]),
                   'auxiliary_pairs': [{'pair_index': i, 'control_seed': seeds[i] if pair_id != 'bl_vs_mvo' else 'not_random',
                                        'treatment': aux_t[metric][i], 'control': aux_c[metric][i]} for i in range(len(seeds))],
                   'assumptions': ['Exploratory p-values: dependent resamples are not independent datasets.',
                                   'Paired t: one-sided greater. Nemenyi: standard two-method, two-sided, infinite df.',
                                   'Time CI conditions on retained seeds; seed distribution is separate membership uncertainty.',
                                   'Difference is treatment metric minus mean seed-specific control metrics.',
                                   'Undefined seed metrics excluded with counts; execution failures are fatal.'],
                   'seed_manifest': result['seed_manifest']}
            result['comparisons'].append(row)
    for method in ('paired_t', 'nemenyi'):
        adjusted = holm([row[method]['p'] for row in result['comparisons']])
        for row, value in zip(result['comparisons'], adjusted):
            row.setdefault('holm_p', {})[method] = {'value': value, 'reason': row[method]['reason'],
                'family': f'{profile}:{method}:3_pairs_x_4_metrics', 'family_size': 12, 'undefined_policy': 'pad_p_1_then_restore_null'}
    return result
