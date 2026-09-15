"""Use a shared, keyed WindowState to forecast one decision month.

Conditional table semantics are model-specific: Ridge emits R1..R5 predictions,
Naive emits the selected regime Sharpe, BL emits the selected regime view q,
and MVO emits no conditional rows. BL posterior means and utility scores are
separate arrays. No model fits macro preprocessing or changes regime labels.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import ResearchSettings
from ..contracts import ErrorCode, ResearchError, TICKERS, canonical_id, save_npz, load_npz, write_json
from ..data.calendar import decision_ledger
from ..data.io import read_json
from ..regimes import WindowState
from ._validation import finite, probabilities
from .ridge import fit_ridge, aggregate_ridge
from .naive import conditional_sharpe
from .black_litterman import regularized_moments, black_litterman_posterior, utility_scores


def _digest(metadata, arrays):
    return canonical_id({'metadata': metadata, 'arrays': {
        name: {'shape': list(a.shape), 'dtype': a.dtype.str,
               'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
        for name, a in arrays.items()}})


@dataclass(frozen=True)
class ModelForecasts:
    _metadata: dict
    _arrays: dict[str, np.ndarray]
    model_id: str

    def _verify(self):
        if self.model_id != _digest(self._metadata, self._arrays):
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'model state changed under existing identity')

    @property
    def metadata(self): self._verify(); return deepcopy(self._metadata)
    @property
    def arrays(self): self._verify(); return {k: v.copy() for k, v in self._arrays.items()}
    @property
    def partition_id(self): self._verify(); return self._metadata['partition_id']
    @property
    def decision_month(self): self._verify(); return self._metadata['decision_month']
    @property
    def scores(self):
        self._verify()
        return {m: pd.Series(self._arrays[m + '_scores'].copy(), index=self._metadata['tickers'], name=m)
                for m in self._metadata['models']}

    def forecast_records(self, run_id):
        self._verify()
        return [{'run_id': run_id, 'decision_month': self._metadata['decision_month'], 'partition_id': self._metadata['partition_id'],
                 **deepcopy(row)} for row in self._metadata['forecasts']]

    def conditional_records(self, run_id):
        self._verify()
        return [{'run_id': run_id, 'decision_month': self._metadata['decision_month'], 'partition_id': self._metadata['partition_id'],
                 **deepcopy(row)} for row in self._metadata['conditional_forecasts']]

    def save(self, directory):
        self._verify()
        directory = Path(directory)
        if any((directory / name).exists() for name in ('models.json', 'models.npz')):
            raise ResearchError(ErrorCode.RUN_CONFLICT, f'model output exists: {directory}')
        archive = save_npz(directory / 'models.npz', self._arrays)
        record = {'model_id': self.model_id, 'metadata': self._metadata, 'archive': archive}
        write_json(directory / 'models.json', record)
        return record

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = read_json(directory / 'models.json')
        arrays = load_npz(directory / 'models.npz', record['archive'])
        state = cls(record['metadata'], arrays, record['model_id'])
        state._verify()
        if state._metadata.get('version') != 1:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'unknown model state version')
        for a in arrays.values(): a.setflags(write=False)
        return state


def _inputs(state, returns, partition_id, decision_month, settings, allow_inner):
    if not isinstance(state, WindowState):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model input requires a WindowState')
    state.require_identity(partition_id, decision_month=decision_month)
    state.require_scope('rolling')
    metadata, ledger, rows = state.metadata, state.ledger, state.training_rows
    inner = metadata['window_kind'] == 'inner_fold'
    if metadata['window_kind'] not in ('outer', 'inner_fold') or inner and not allow_inner or not inner and len(rows) != 48:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'outer requires 48 rows; inner_fold requires allow_inner=True')
    if not isinstance(returns, pd.DataFrame) or returns.index.has_duplicates or returns.columns.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY, 'model returns require unique row and ticker keys')
    expected = decision_ledger(decision_month, [ledger['vintage_month']], profile=ledger['profile'], lag_months=ledger['lag_months'], train_months=len(rows))
    if any(ledger.get(k) != v for k, v in expected.items()):
        raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'model timing ledger differs from past-only calendar')
    if rows.drop(columns='regime_id').to_dict(orient='records') != expected['training_rows']:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model training roles differ from ledger')
    if settings.profile != ledger['profile'] or settings.lag_months != ledger['lag_months'] or settings.train_months != 48:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model settings differ from state timing')
    if list(returns.index) != rows.target_month.tolist() or tuple(returns.columns) != TICKERS:
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'target month/ticker order must exactly match the training window')
    labels, predictors, query = state.labels, state.predictor_scores, state.current_scores
    if (state.regime_ids != tuple(f'R{i}' for i in range(6)) or list(labels.index) != list(returns.index)
        or rows.regime_id.tolist() != [f'R{i}' for i in labels]
        or predictors.index.tolist() != rows.predictor_base_month.tolist()
        or query.index.tolist() != [ledger['current_base_month']]
        or not predictors.columns.equals(query.columns)):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model predictor, query, target or regime ordering differs')
    if labels.dtype.kind not in 'iu' or (labels < 0).any() or (labels > 5).any():
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model regime labels must be integers R0..R5')
    x, y, q = finite(predictors, 'model predictors', 2), finite(returns, 'model returns', 2), finite(query, 'model query', 2)[0]
    if len(x) != len(y):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'model predictor/target length differs')
    return x, y, q, labels.to_numpy(), probabilities(state.next_probability), ledger, rows, metadata


def forecast_window(state, returns, *, partition_id, decision_month, settings=None, fixed_lambdas=None, allow_inner=False):
    """Outer adapter is 48 months / 10 fixed ETFs; inner use must be explicit.

    `fixed_lambdas` is a Series indexed by TICKERS in that order. For blocked
    validation it is mandatory: the caller owns past-only forward selection.
    Supplied returns contain only the exact training rows, never future data.
    """
    settings = settings or ResearchSettings(profile=state.ledger['profile'], lag_months=state.ledger['lag_months'])
    x, y, query, labels, p, ledger, rows, state_metadata = _inputs(state, returns, partition_id, decision_month, settings, allow_inner)
    if settings.validation not in ('loo', 'blocked') or settings.fallback not in ('pooled', 'regime_mean') or settings.min_regime_samples < 2:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid model validation or fallback settings')
    models = tuple(settings.models)
    if not models or len(set(models)) != len(models) or not set(models) <= {'ridge', 'naive', 'bl', 'mvo'}:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'unknown or duplicate models')
    if 'ridge' in models and settings.validation == 'blocked' and fixed_lambdas is None:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'blocked validation requires past-only fixed_lambdas from forward validation')
    fixed = None
    if fixed_lambdas is not None:
        if not isinstance(fixed_lambdas, pd.Series) or tuple(fixed_lambdas.index) != TICKERS:
            raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'fixed_lambdas require exact ticker order')
        fixed = finite(fixed_lambdas, 'fixed_lambdas', 1)
        if (fixed <= 0).any(): raise ResearchError(ErrorCode.INVALID_CONFIG, 'fixed_lambdas must be positive')
    arrays = {'predictor_scores': x.copy(), 'training_returns': y.copy(), 'current_scores': query.copy(),
              'labels': labels.copy(), 'next_probability': p.copy()}
    metadata = {'version': 1, 'partition_id': partition_id, 'decision_month': decision_month,
                'window_kind': state_metadata['window_kind'], 'training_n': len(y), 'tickers': list(TICKERS),
                'training_rows': rows.to_dict(orient='records'), 'settings': asdict(settings), 'models': list(models),
                'feature_hash': state.feature_hash, 'transform_hash': state.transform_hash,
                'conditional_semantics': {'ridge': 'normal_regime_expected_return', 'naive': 'selected_regime_conditional_sharpe',
                    'bl': 'selected_regime_view_q_expected_return', 'mvo': 'unconditional_no_rows'},
                'ridge_selection': 'fixed_external' if fixed is not None else 'analytic_loo_with_explicit_guard',
                'ridge_loo_tie_policy': 'relative_tolerance_1e-12_absolute_0_then_largest_lambda',
                'bl_interpretation': 'Eq19 is posterior expected return; solve(delta*Sigma_reg,posterior) is a separately declared utility direction; portfolio sizing follows',
                'mvo_regime_use': 'none in forecast; mx portfolio sizing can use common crisis signal',
                'model_diagnostics': {}, 'forecasts': [], 'conditional_forecasts': []}

    def forecast(model, value, kind, statuses=None, reasons=None):
        arrays[model + '_scores'] = finite(value, model + ' output', 1).copy()
        for j, ticker in enumerate(TICKERS):
            metadata['forecasts'].append({'model': model, 'ticker': ticker, 'value': float(value[j]), 'score_kind': kind,
                'status': statuses[j] if statuses is not None else 'ok', 'reason': reasons[j] if reasons is not None else ''})

    def conditional(model, regime, value, kind, raw_n, effective_n, fallback='none', statuses=None, reasons=None, selected=None):
        for j, ticker in enumerate(TICKERS):
            reason = reasons[j] if reasons is not None else ''
            if selected is None:
                reason = (reason + '; ' if reason else '') + 'selected_lambda_not_applicable'
            metadata['conditional_forecasts'].append({'model': model, 'ticker': ticker, 'regime_id': f'R{regime}',
                'value': float(value[j]), 'score_kind': kind, 'raw_n': int(raw_n), 'effective_n': int(effective_n),
                'selected_lambda': float(selected[j]) if selected is not None else None, 'fallback': fallback,
                'status': statuses[j] if statuses is not None else 'fallback' if fallback != 'none' else 'ok', 'reason': reason})

    modal = int(np.argmax(p))  # Ordered R0..R5 makes probability ties deterministic.
    selected_returns = y[labels == modal]
    n_regime = len(selected_returns)
    metadata['selected_regime'] = f'R{modal}'
    if 'naive' in models:
        naive = conditional_sharpe(selected_returns)
        arrays.update(naive_mean=naive.mean, naive_std=naive.std)
        metadata['model_diagnostics']['naive'] = {'raw_n': n_regime, 'effective_n': n_regime, 'fallback': 'none',
            'mean_defined': n_regime > 0, 'std_defined': n_regime >= 2,
            'padding_policy': 'undefined summary components are finite zero padding identified by sample count; score status remains undefined'}
        conditional('naive', modal, naive.values, 'conditional_sharpe', n_regime, n_regime, statuses=naive.status, reasons=naive.reasons)
        forecast('naive', naive.values, 'conditional_sharpe', naive.status, naive.reasons)

    if 'ridge' in models:
        pooled = None
        conditional_values, fallback_used = [], []
        ridge_diagnostics = {}
        fit_options = {'fixed_lambda': fixed} if fixed is not None else {'lambdas': settings.lambdas}
        for regime in range(1, 6):
            mask = labels == regime
            raw_n = int(mask.sum())
            short = raw_n < settings.min_regime_samples
            fallback = 'none'
            if short and settings.fallback == 'regime_mean':
                fallback = 'regime_mean' if raw_n else 'pooled_mean'
                effective_n = raw_n if raw_n else len(y)
                prediction = (y[mask] if raw_n else y).mean(axis=0)
                arrays[f'ridge_R{regime}_mean'] = prediction.copy()
                selected_lambda = None
                diag = {'method': fallback, 'selected_lambda_reason': 'mean fallback has no Ridge penalty'}
            else:
                if short:
                    fallback = 'pooled'
                    if pooled is None: pooled = fit_ridge(x, y, query, **fit_options)
                    fit = pooled
                    effective_n = len(y)
                else:
                    fit = fit_ridge(x[mask], y[mask], query, **fit_options)
                    effective_n = raw_n
                prediction, selected_lambda = fit.prediction, fit.selected_lambda
                prefix = f'ridge_R{regime}_'
                arrays.update({prefix + 'coefficients': fit.coefficients.copy(), prefix + 'intercept': fit.intercept.copy(), prefix + 'selected_lambda': fit.selected_lambda.copy()})
                diag = {'method': fit.selection_method}
                if fit.candidates is not None:
                    candidates = fit.candidates
                    for name in ('lambdas', 'loo_mse', 'predictions', 'valid'):
                        arrays[prefix + 'candidate_' + name] = getattr(candidates, name).copy()
                    diag['candidate_methods'] = list(candidates.methods)
                    diag['candidate_reasons'] = list(candidates.reasons)
                    diag['candidate_axes'] = ['lambda', 'ticker']
                    diag['invalid_candidate_policy'] = 'finite zero padding with candidate_valid=false; never eligible for selection'
                else:
                    diag['candidate_mse_reason'] = 'external forward selection; candidate loss belongs to caller validation artifact'
            diag.update(raw_n=raw_n, effective_n=effective_n, fallback=fallback)
            ridge_diagnostics[f'R{regime}'] = diag
            reason = 'small_regime_' + fallback if fallback != 'none' else ''
            conditional('ridge', regime, prediction, 'expected_return', raw_n, effective_n, fallback,
                        reasons=[reason] * len(TICKERS), selected=selected_lambda)
            conditional_values.append(prediction)
            fallback_used.append(fallback != 'none')
        arrays['ridge_conditional_predictions'] = np.asarray(conditional_values)
        value = aggregate_ridge(conditional_values, p)
        used = any(flag and probability > 0 for flag, probability in zip(fallback_used, p[1:]))
        reason = 'normal_regime_probability_zero_cash' if p[1:].sum() == 0 else 'contributing_regime_fallback' if used else ''
        forecast('ridge', value, 'expected_return', ['fallback' if used else 'ok'] * len(TICKERS), [reason] * len(TICKERS))
        metadata['model_diagnostics']['ridge'] = ridge_diagnostics

    if 'bl' in models or 'mvo' in models:
        mean, sample, covariance = regularized_moments(y, shrink=settings.cov_shrink, eps=settings.cov_eps)
        arrays.update(prior_mean=mean, sample_covariance=sample, regularized_covariance=covariance)
        if 'mvo' in models:
            forecast('mvo', utility_scores(mean, covariance, delta=settings.delta), 'utility_weight')
            metadata['model_diagnostics']['mvo'] = {'raw_n': len(y), 'effective_n': len(y), 'fallback': 'none'}
        if 'bl' in models:
            fallback = 'prior_mean' if n_regime < 2 else 'none'
            q = mean.copy() if n_regime < 2 else selected_returns.mean(axis=0)
            omega_scale = finite(settings.omega_scale, 'omega_scale', 0).item()
            if omega_scale <= 0: raise ResearchError(ErrorCode.INVALID_CONFIG, 'omega_scale must be positive')
            view_matrix = np.eye(len(TICKERS))
            omega = omega_scale * np.diag(np.diag(covariance)) / max(n_regime, 1)
            posterior = black_litterman_posterior(mean, covariance, q, view_matrix, omega, tau=settings.tau)
            arrays.update(bl_q=q, bl_P=view_matrix, bl_omega=omega, bl_posterior_mean=posterior)
            reason = 'insufficient_regime_sample_prior_mean' if fallback != 'none' else ''
            conditional('bl', modal, q, 'expected_return', n_regime, len(y) if fallback != 'none' else n_regime,
                        fallback, reasons=[reason] * len(TICKERS))
            forecast('bl', utility_scores(posterior, covariance, delta=settings.delta), 'utility_weight',
                     ['fallback' if fallback != 'none' else 'ok'] * len(TICKERS), [reason] * len(TICKERS))
            metadata['model_diagnostics']['bl'] = {'raw_n': n_regime, 'effective_n': len(y) if fallback != 'none' else n_regime,
                'prior_n': len(y), 'fallback': fallback, 'tau': settings.tau, 'omega_scale': omega_scale, 'delta': settings.delta}
    # Public order follows configured model order, independently of computation.
    metadata['forecasts'].sort(key=lambda r: (models.index(r['model']), TICKERS.index(r['ticker'])))
    for array in arrays.values():
        finite(array, 'model artifact')
        array.setflags(write=False)
    return ModelForecasts(metadata, arrays, _digest(metadata, arrays))
