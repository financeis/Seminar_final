"""Monthly prequential tuning of a common per-ETF Ridge penalty.

The last 24 targets of outer48 are evaluated in four reporting groups of six.
Every month v gets a fresh fit using targets A..v-2 (23..46 observations),
including earlier validation targets once their returns become available.
Groups do not freeze a six-month fit. Loss is on the final ETF forecast,
after the normal-regime probability mixture, not on conditional LOO errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..contracts import ErrorCode, ResearchError, TICKERS
from ..data.calendar import add_months, decision_ledger, month_range
from ..features.preprocessing import fit_preprocessor
from ..features.transforms import transform_tcodes
from ..features.windows import FeatureWindow
from ..regimes.state import build_inner_window_state
from ._validation import finite, probabilities
from .ridge import ridge_prediction_candidates


def forward_ledgers(month, vintages, settings, *, fixed_vintage='2023-02'):
    """Return all 24 as-of ledgers, anchored to the outer first target A."""
    if settings.train_months != 48 or settings.validation != 'blocked':
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'forward validation requires outer48 and validation=blocked')
    return [decision_ledger(v, vintages, profile=settings.profile,
                            lag_months=settings.lag_months, train_months=23 + i,
                            fixed_vintage=fixed_vintage)
            for i, v in enumerate(month_range(add_months(month, -25), add_months(month, -2)))]


def _inner_state(vintage, ledger, settings):
    """Fit every preprocessing step on exactly the inner condition rows."""
    if (vintage.vintage_month != ledger['vintage_month'] or
        vintage.assumed_available_at != ledger['assumed_available_at']):
        raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'inner vintage differs from as-of ledger')
    raw = vintage.values.loc[vintage.values.index <= ledger['macro_cutoff_base_month']]
    transformed = transform_tcodes(raw, vintage.tcodes)
    rows = pd.DataFrame(ledger['training_rows'])
    provenance = {key: ledger[key] for key in ('decision_month', 'decision_at', 'profile', 'lag_months',
                  'vintage_month', 'assumed_available_at', 'macro_cutoff_base_month')}
    provenance.update(dataset_id=vintage.dataset_id, tcodes={c: int(vintage.tcodes[c]) for c in transformed},
                      validation='monthly_prequential', window_kind='inner_fold')
    prep = fit_preprocessor(transformed, rows.condition_base_month.tolist(), vintage.groups,
                            missing_rate=settings.missing_rate, ffill_limit=settings.ffill_limit,
                            pca_variance=settings.pca_variance, scope='rolling', provenance=provenance)
    features = FeatureWindow(prep, prep.fit_result,
                             prep.transform(transformed, rows.predictor_base_month.tolist(), expected_scope='rolling'),
                             prep.transform(transformed, [ledger['current_base_month']], expected_scope='rolling'),
                             rows, dict(ledger))
    return build_inner_window_state(features, settings)


def candidate_forecasts(state, training, settings):
    """Candidate-by-ETF final forecasts and validity; one SVD per design.

    No LOO loss is computed. The same small-regime fallback and R0 cash mass
    as forecast_window apply; only the final forward forecasts are scored.
    """
    if state.scope != 'rolling' or state.metadata.get('window_kind') != 'inner_fold':
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'forward forecasts require rolling inner_fold state')
    labels_series = state.labels
    if (not isinstance(training, pd.DataFrame) or training.index.has_duplicates or
        not training.index.equals(labels_series.index)):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'inner target month order differs from state labels')
    x = finite(state.predictor_scores, 'inner predictors', 2)
    y = finite(training, 'inner returns', 2)
    q = finite(state.current_scores, 'inner query', 2)[0]
    labels = labels_series.to_numpy()
    p = probabilities(state.next_probability)
    if (y.shape != (len(x), len(TICKERS)) or len(labels) != len(x) or
        tuple(training.columns) != TICKERS or p.shape != (6,)):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'inner forecast input dimensions or tickers differ')
    if settings.fallback not in ('pooled', 'regime_mean'):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'unknown inner Ridge fallback')
    grid = np.asarray(settings.lambdas)
    result = np.zeros((len(grid), len(TICKERS)))
    valid = np.ones(len(grid), dtype=bool)
    diagnostics, pooled = {}, None
    for regime in range(1, 6):
        mask = labels == regime
        raw_n = int(mask.sum())
        short = raw_n < settings.min_regime_samples
        fallback = 'none'
        if short and settings.fallback == 'regime_mean':
            fallback = 'regime_mean' if raw_n else 'pooled_mean'
            effective_n = raw_n if raw_n else len(y)
            predictions = np.broadcast_to((y[mask] if raw_n else y).mean(axis=0), result.shape)
            reasons = [''] * len(grid)
        else:
            if short:
                fallback = 'pooled'
                if pooled is None: pooled = ridge_prediction_candidates(x, y, q, grid)
                candidates = pooled
                effective_n = len(y)
            else:
                candidates = ridge_prediction_candidates(x[mask], y[mask], q, grid)
                effective_n = raw_n
            predictions = candidates.predictions
            valid &= candidates.valid
            reasons = list(candidates.reasons)
        result += p[regime] * predictions
        diagnostics[f'R{regime}'] = {'raw_n': raw_n, 'effective_n': effective_n, 'fallback': fallback,
                                    'candidate_invalid_reasons': reasons}
    valid &= np.isfinite(result).all(axis=1)
    return result, valid, diagnostics


def select_from_predictions(predictions, observed, lambdas, valid):
    """Score identical 24-month panels; invalid candidates never receive a loss."""
    predictions = np.asarray(predictions, dtype=float)
    observed = finite(observed, 'forward observed returns', 2)
    grid = finite(lambdas, 'forward lambda grid', 1)
    valid = np.asarray(valid, dtype=bool)
    if (observed.shape != (24, len(TICKERS)) or predictions.shape != (24, len(grid), len(TICKERS))
        or valid.shape != (24, len(grid)) or not len(grid) or (grid <= 0).any() or len(set(grid)) != len(grid)):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'forward loss requires 24 months and distinct positive candidates')
    usable = valid[:, :, None] & np.isfinite(predictions)
    counts = usable.sum(axis=0)
    eligible = counts == 24
    with np.errstate(over='ignore', invalid='ignore'):
        losses = np.mean((predictions - observed[:, None, :]) ** 2, axis=0)
    eligible &= np.isfinite(losses)
    records = [{'ticker': ticker, 'lambda': float(lam), 'mse': float(losses[i, j]) if eligible[i, j] else None,
                'valid_months': int(counts[i, j]), 'required_months': 24,
                'status': 'ok' if eligible[i, j] else 'undefined',
                'reason': '' if eligible[i, j] else 'requires_24_valid_months_and_finite_final_forecast_mse'}
               for i, lam in enumerate(grid) for j, ticker in enumerate(TICKERS)]
    selected = []
    for j, ticker in enumerate(TICKERS):
        if not eligible[:, j].any():
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'no valid forward candidate for {ticker}',
                                details={'candidate_losses': records})
        best = np.min(losses[eligible[:, j], j])
        ties = eligible[:, j] & np.isclose(losses[:, j], best, rtol=1e-12, atol=0)
        selected.append(float(np.max(grid[ties])))
    return pd.Series(selected, index=TICKERS, name='lambda'), records


def _prediction_records(predictions, valid, lambdas):
    return [{'lambda': float(lam),
             'values': row.tolist() if valid[i] and np.isfinite(row).all() else None,
             'reason': '' if valid[i] and np.isfinite(row).all() else 'invalid_candidate_see_regime_diagnostics'}
            for i, (lam, row) in enumerate(zip(lambdas, predictions))]


def select_forward_lambdas(config, context, month):
    """Return engine.LambdaSelection for fixed_lambdas={month: selection}.

    Outer fitting stays with ExecutionContext.forecast/run_research. The
    evidence preserves every inner fit's dates, IDs, forecasts and losses.
    No stored run cache or external network is consulted.
    """
    # Local import avoids engine -> models -> engine initialization cycles.
    from ..backtest.engine import LambdaSelection
    context.require_config(config)
    settings = config.research
    ledgers = forward_ledgers(month, context._vintages, settings, fixed_vintage=config.data.fixed_vintage)
    start, end = add_months(month, -49), add_months(month, -2)
    returns = context.returns(start, end)
    forecasts, validity, monthly = [], [], []
    with threadpool_limits(1):
        for i, ledger in enumerate(ledgers):
            v = ledger['decision_month']
            key = ledger['vintage_month']
            try:
                vintage = context.vintage(config, v)
                state = _inner_state(vintage, ledger, settings)
                train = returns.loc[[x['target_month'] for x in ledger['training_rows']]]
                predictions, valid, diagnostics = candidate_forecasts(state, train, settings)
            except ResearchError as exc:
                raise ResearchError(exc.code, f'forward validation failed at {v}: {exc}',
                                    details={'decision_month': month, 'validation_month': v,
                                             'completed_monthly_folds': monthly, 'cause': exc.details}) from exc
            forecasts.append(predictions)
            validity.append(valid)
            monthly.append({'validation_month': v, 'fold': i // 6 + 1,
                            'training_n': len(train), 'train_start': train.index[0], 'train_end': train.index[-1],
                            'training_target_end_at': ledger['training_rows'][-1]['target_end_at'],
                            'decision_at': ledger['decision_at'], 'validation_target_end_at': ledger['target_return_end_at'],
                            'vintage_month': key, 'assumed_available_at': ledger['assumed_available_at'],
                            'macro_cutoff_base_month': ledger['macro_cutoff_base_month'],
                            'condition_months': state.metadata['condition_months'],
                            'predictor_months': state.metadata['predictor_months'],
                            'window_kind': state.metadata['window_kind'], 'scope': state.scope,
                            'partition_id': state.partition_id, 'feature_hash': state.feature_hash,
                            'transform_hash': state.transform_hash, 'regime_diagnostics': diagnostics,
                            'candidate_valid': valid.tolist(),
                            'candidate_predictions': _prediction_records(predictions, valid, settings.lambdas),
                            'observed_returns': returns.loc[v].to_dict()})
    try:
        values, losses = select_from_predictions(forecasts, returns.loc[[x['decision_month'] for x in ledgers]],
                                                settings.lambdas, validity)
    except ResearchError as exc:
        raise ResearchError(exc.code, str(exc), details={**exc.details, 'monthly_folds': monthly}) from exc
    folds = [{'fold': i + 1, 'validation_months': [x['validation_month'] for x in monthly[i * 6:(i + 1) * 6]],
              'training_n': [x['training_n'] for x in monthly[i * 6:(i + 1) * 6]],
              'candidate_valid_months': np.asarray(validity[i * 6:(i + 1) * 6]).sum(axis=0).tolist()}
             for i in range(4)]
    evidence = {'source': 'monthly_prequential_forward_validation', 'decision_month': month,
                'selection_end_at': ledgers[-1]['target_return_end_at'], 'tickers': list(TICKERS),
                'outer_train_start': start, 'outer_train_end': end, 'outer_training_n': 48,
                'validation_method': 'monthly_refits_reported_in_four_six_month_groups',
                'training_policy': 'outer_first_target_A_through_validation_month_v_minus_2',
                'loss': 'mean_squared_error_of_final_probability_aggregated_ETF_forecast',
                'tuning_unit': 'one_lambda_per_ETF_shared_by_all_normal_regimes',
                'baseline_tuning_unit': 'LOO_lambda_per_regime_and_ETF',
                'tie_policy': 'relative_tolerance_1e-12_absolute_0_then_largest_lambda',
                'required_validation_months': 24, 'invalid_policy': 'all_24_months_required_per_candidate_and_ETF',
                'lambdas': list(settings.lambdas), 'candidate_losses': losses,
                'monthly_folds': monthly, 'folds': folds,
                'profile': settings.profile,
                'availability_caveat': 'ex_post_fixed_snapshot' if settings.profile == 'fixed_snapshot' else 'assumed_vintage_release_lag'}
    selection = LambdaSelection(values, evidence)
    from ..data.calendar import decision_at
    selection.record(month, decision_at(month).isoformat())
    return selection
