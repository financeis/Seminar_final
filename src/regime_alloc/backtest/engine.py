"""Offline monthly orchestration with shared, content-checked research state.

No computation cache is read from previous run directories. A caller may share
one ExecutionContext within a suite; its cache records immutable source IDs,
fit settings, actual inputs and immutable model/partition IDs. Experiment
generation and forward selection belong to the experiments layer.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import re
import subprocess
import uuid
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..config import ResearchConfig, validate_config
from ..contracts import SCHEMA_VERSION, TICKERS, ErrorCode, ResearchError, canonical_id, canonical_json, sha256_file
from ..data import available_vintages, load_macro_vintage, load_prices
from ..data.calendar import add_months, decision_ledger, first_session, month_range, monthly_returns
from ..features.windows import build_feature_window
from ..models import forecast_window
from ..portfolio.sizing import allocate, benchmark_weights
from ..portfolio.volatility import scale_to_volatility
from ..regimes import build_window_state
from .accounting import account_month, ACCOUNTING_POLICY
from .artifacts import RunWriter, WINDOW_COLUMNS, COLUMNS
from .metrics import compute_metrics

FEATURE_SETTINGS = ('profile', 'train_months', 'lag_months', 'missing_rate', 'ffill_limit', 'pca_variance',
                    'k_normal', 'n_init', 'max_iter', 'tol', 'seed')


def _frame_id(frame):
    return canonical_id({'index': list(frame.index), 'columns': list(frame.columns),
                         'dtypes': [str(x) for x in frame.dtypes],
                         'values': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()})


def _dataset_ids(config):
    result = {name: getattr(config.data, name + '_dataset_id') for name in ('yahoo', 'fred', 'nber')}
    if not all(result.values()):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'runs require explicitly pinned yahoo, fred and nber dataset IDs')
    return result


def _code_snapshot():
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    paths = sorted(package.rglob('*.py')) + [p for p in (repo / 'pyproject.toml', repo / 'requirements-research.lock') if p.exists()]
    hashes = {p.relative_to(repo).as_posix(): sha256_file(p) for p in paths}
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
    revision = result.stdout.strip() if result.returncode == 0 else 'unavailable_source_tree'
    return revision, hashes


def _environment():
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get('Name')}
    normalized = {re.sub(r'[-_.]+', '-', name).lower(): version for name, version in packages.items()}
    lock = Path(__file__).resolve().parents[3] / 'requirements-research.lock'
    expected = {}
    for line in lock.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line:
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'research lock must contain exact package versions')
        name, version = line.split('==', 1)
        expected[name] = version
        if normalized.get(re.sub(r'[-_.]+', '-', name).lower()) != version:
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'installed version differs from research lock: {name}=={version}')
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
              'locked_packages': expected, 'lock_sha256': sha256_file(lock), 'lock_validation': 'pass'}
    result['content_hash'] = canonical_id(result)
    return result


def selected_months(config, smoke=False):
    months = month_range(config.research.start_month, config.research.end_month)
    if not smoke:
        return months
    if len(months) < 3 or '2020-04' not in months or months[0] == '2020-04' or months[-1] == '2020-04':
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'smoke requires first, 2020-04 shock and last distinct months')
    return [months[0], '2020-04', months[-1]]


def _strategies(config, selection):
    r = config.research
    configured = tuple(f'{m}_{s}_{n}' for m in r.models for s in r.strategies for n in r.selection_sizes) + ('spy', 'ew')
    chosen = configured if selection is None else tuple(selection)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(configured):
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'strategy selection must be unique and contained in configured strategies')
    return chosen


@dataclass(frozen=True)
class LambdaSelection:
    """A fixed per-ETF selection plus serializable as-of evidence from T08.

    evidence requires source, decision_month, selection_end_at, and tickers.
    selection_end_at is the latest target price used in validation and must be
    before the outer decision. Detailed fold/loss records may be added here.
    Values are a Series in the canonical TICKERS order.
    """
    values: pd.Series
    evidence: dict

    def record(self, month, decision_at):
        evidence = deepcopy(self.evidence)
        if (not isinstance(self.values, pd.Series) or tuple(self.values.index) != TICKERS
            or not np.isfinite(self.values.to_numpy(dtype=float)).all() or (self.values <= 0).any()):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'lambda selection requires finite positive values in ETF order')
        if (not evidence.get('source') or evidence.get('decision_month') != month
            or tuple(evidence.get('tickers', [])) != TICKERS or not evidence.get('selection_end_at')):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'lambda selection lacks source, decision month, tickers or timing evidence')
        try:
            end = pd.Timestamp(evidence['selection_end_at'])
            if end.tzinfo is None or end >= pd.Timestamp(decision_at):
                raise ValueError('selection end must precede decision')
        except (TypeError, ValueError) as exc:
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'lambda selection uses unavailable returns') from exc
        record = {'values': self.values.to_dict(), 'evidence': evidence}
        canonical_json(record)
        return record


class ExecutionContext:
    """Owned raw input and checked memory caches shared only within this process.

    Returned return panels are copies; states and models enforce their own
    content digests. A cache key is also bound to its initial value ID, so
    replacing an entry with a different valid object cannot alias that key.
    No external mutable callback is accepted.
    """
    def __init__(self, config):
        validate_config(config)
        self.dataset_ids = _dataset_ids(config)
        self.data_root = Path(config.data.root).resolve()
        self._identity = canonical_id({'ids': self.dataset_ids, 'root': self.data_root})
        self._prices = load_prices(self.data_root, self.dataset_ids['yahoo'])
        self._prices_id = _frame_id(self._prices)
        self._vintages = tuple(available_vintages(self.data_root, self.dataset_ids['fred']))
        self._vintages_id = canonical_id(self._vintages)
        self._code_identity = _code_snapshot()
        self._raw, self._returns, self._states, self._models = {}, {}, {}, {}
        self._raw_ids, self._return_ids, self._state_ids, self._model_ids = {}, {}, {}, {}
        self.counts = dict(raw_reads=0, return_panels_computed=0, states_computed=0, states_shared=0,
                           models_computed=0, models_shared=0, prior_run_cache_reads=0, state_evictions=0, model_evictions=0)

    def _require_identity(self, config):
        expected = canonical_id({'ids': _dataset_ids(config), 'root': Path(config.data.root).resolve()})
        if expected != self._identity or canonical_id({'ids': self.dataset_ids, 'root': self.data_root}) != self._identity:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'execution context dataset identity changed')

    def require_config(self, config):
        self._require_identity(config)
        if _frame_id(self._prices) != self._prices_id or canonical_id(self._vintages) != self._vintages_id:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'execution context raw inputs changed')
        if _code_snapshot() != self._code_identity:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'execution context was built by different code')
        for key, value in self._raw.items():
            if self._vintage_id(value) != self._raw_ids[key]:
                raise ResearchError(ErrorCode.HASH_MISMATCH, 'macro cache ID collision or mutation')

    def feasible_start(self, config):
        """Listing warmup only; never step past a gap after the complete panel starts."""
        first = self._prices.groupby('ticker').session.min()
        if set(first.index) != set(TICKERS):
            raise ResearchError(ErrorCode.MISSING_DATA, 'price panel lacks a required ETF')
        latest_listing = max(first)
        first_month = latest_listing[:7]
        if first_session(first_month) < latest_listing:
            first_month = add_months(first_month, 1)
        first_decision = add_months(first_month, 49)
        if config.research.profile == 'vintage_lagged':
            first_decision = max(first_decision, add_months(self._vintages[0], config.research.lag_months))
        result = max(config.research.start_month, first_decision)
        if result > config.research.end_month:
            raise ResearchError(ErrorCode.INSUFFICIENT_HISTORY, 'requested period has no complete 48-month training context',
                                details={'first_feasible_month': first_decision})
        return result

    def returns(self, start, end):
        if _frame_id(self._prices) != self._prices_id:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'raw price input changed')
        key = canonical_id({'dataset': self.dataset_ids['yahoo'], 'start': start, 'end': end})
        if key not in self._returns:
            frame = monthly_returns(self._prices, start, end)
            self._returns[key], self._return_ids[key] = frame, _frame_id(frame)
            self.counts['return_panels_computed'] += 1
        result = self._returns[key]
        if _frame_id(result) != self._return_ids[key]:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'return cache ID collision or mutation')
        return result.copy(deep=True)

    def _decision_ledger(self, config, month):
        validate_config(config)
        self._require_identity(config)
        if canonical_id(self._vintages) != self._vintages_id:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'execution context vintage catalog changed')
        settings = config.research
        return decision_ledger(month, self._vintages, profile=settings.profile,
                               lag_months=settings.lag_months, train_months=48, fixed_vintage=config.data.fixed_vintage)

    def _cached_vintage(self, ledger):
        """Load once, then check the owned value against its original content ID."""
        raw_key = (ledger['vintage_month'], ledger['lag_months'])
        if raw_key not in self._raw:
            vintage = load_macro_vintage(self.data_root, raw_key[0], lag_months=raw_key[1], dataset_id=self.dataset_ids['fred'])
            if (vintage.vintage_month != raw_key[0] or vintage.dataset_id != self.dataset_ids['fred']
                or vintage.assumed_available_at != ledger['assumed_available_at']):
                raise ResearchError(ErrorCode.HASH_MISMATCH, 'loaded macro vintage differs from selected source identity')
            self._raw[raw_key], self._raw_ids[raw_key] = vintage, self._vintage_id(vintage)
            self.counts['raw_reads'] += 1
        vintage = self._raw[raw_key]
        if self._vintage_id(vintage) != self._raw_ids[raw_key]:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'macro cache ID collision or mutation')
        return vintage

    def vintage(self, config, decision_month):
        """Return a deep copy of the release selected for this decision month.

        Uses the same profile/lag ledger as an outer window, but does not fit
        features or require 48 observed target returns. Inner forward folds can
        therefore reuse raw releases without requesting a future release or
        accessing the private cache. A fixed snapshot keeps its actual assumed
        availability, even when it is later than the decision.
        """
        vintage = self._cached_vintage(self._decision_ledger(config, decision_month))
        return replace(vintage, values=vintage.values.copy(deep=True), tcodes=vintage.tcodes.copy(deep=True),
                       groups=vintage.groups.copy(deep=True), metadata=deepcopy(vintage.metadata))

    def state(self, config, month):
        ledger = self._decision_ledger(config, month)
        settings = config.research
        key = canonical_id({'dataset': self.dataset_ids['fred'], 'ledger': ledger,
                            'settings': {k: getattr(settings, k) for k in FEATURE_SETTINGS}})
        vintage = self._cached_vintage(ledger)
        if key not in self._states:
            features = build_feature_window(vintage, ledger, settings)
            state = build_window_state(features, settings)
            self._states[key], self._state_ids[key] = state, state.partition_id
            self.counts['states_computed'] += 1
            # Bound memory across many sensitivity/control runs. Eviction only
            # causes an identical recomputation; it never reads an old run.
            if len(self._states) > 512:
                oldest = next(iter(self._states)); del self._states[oldest]; del self._state_ids[oldest]
                self.counts['state_evictions'] += 1
        else:
            self.counts['states_shared'] += 1
        state = self._states[key]
        state.require_identity(self._state_ids[key], decision_month=month)
        state.require_scope('rolling')
        if state.ledger != ledger:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'cached state ledger changed')
        return state

    @staticmethod
    def _vintage_id(vintage):
        return canonical_id({'values': _frame_id(vintage.values), 'tcodes': vintage.tcodes.to_dict(),
                             'groups': vintage.groups.to_dict(), 'month': vintage.vintage_month,
                             'available': vintage.assumed_available_at, 'dataset': vintage.dataset_id,
                             'metadata': vintage.metadata})

    def forecast(self, state, training, settings, selection=None):
        record = selection.record(state.decision_month, state.ledger['decision_at']) if selection is not None else {}
        key = canonical_id({'partition_id': state.partition_id, 'training': _frame_id(training),
                            'settings': asdict(settings), 'lambda_selection': record})
        if key not in self._models:
            model = forecast_window(state, training, partition_id=state.partition_id, decision_month=state.decision_month,
                                    settings=settings, fixed_lambdas=selection.values.copy() if selection else None)
            self._models[key], self._model_ids[key] = model, model.model_id
            self.counts['models_computed'] += 1
            if len(self._models) > 512:
                oldest = next(iter(self._models)); del self._models[oldest]; del self._model_ids[oldest]
                self.counts['model_evictions'] += 1
        else:
            self.counts['models_shared'] += 1
        model = self._models[key]
        state.require_identity(state.partition_id, decision_month=state.decision_month)
        if model.model_id != self._model_ids[key] or model.partition_id != state.partition_id or model.decision_month != state.decision_month:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'model cache ID collision or mutation')
        return model


def _override_state(base, supplied):
    supplied.require_scope('rolling')
    meta = supplied.metadata
    if (meta.get('parent_partition_id') != base.partition_id or supplied.ledger != base.ledger
        or meta.get('window_kind') != 'outer' or supplied.feature_hash != base.feature_hash
        or supplied.transform_hash != base.transform_hash):
        raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'override must be a membership-only child of this original window')
    rebuilt = base.with_labels(supplied.labels.to_numpy(), provenance=meta['membership_provenance'])
    supplied.require_identity(rebuilt.partition_id, decision_month=base.decision_month)
    return supplied


def _window_record(run_id, state, model):
    meta, ledger = state.metadata, state.ledger
    prep = meta['preprocessor_metadata']
    labels = state.labels.to_numpy()
    raw_n = {f'R{i}': int(np.count_nonzero(labels == i)) for i in range(6)}
    effective = {m: model.metadata['model_diagnostics'][m] for m in model.metadata['models']}
    arrays = state._arrays  # Verified by metadata above; only read numeric fit diagnostics.
    n_pc = len(meta['pc_columns'])
    warnings = [x['reason'] for x in ledger['checks'] if x['status'] == 'not_applicable']
    return dict(run_id=run_id, window_id=canonical_id({'partition_id': state.partition_id, 'model_id': model.model_id}),
                decision_month=state.decision_month, decision_at=ledger['decision_at'], execution_at=ledger['execution_at'],
                macro_vintage=ledger['vintage_month'], macro_cutoff=ledger['macro_cutoff_base_month'],
                train_months=state.labels.index.tolist(), train_target_end_max=max(x['target_end_at'] for x in ledger['training_rows']),
                feature_names=meta['feature_columns'], feature_hash=state.feature_hash, transform_hash=state.transform_hash,
                partition_id=state.partition_id, regime_order=list(state.regime_ids), raw_n=raw_n, effective_n=effective,
                pca_components=n_pc, explained_variance=float(arrays['preprocessor_explained_variance_ratio'][:n_pc].sum()),
                imputation_counts=prep['fit_imputation_counts'], exclusions=[x for x in prep['selection'] if not x.get('selected', False)], warnings=warnings)


def _metric_rows(run_id, strategy_ids, accounts, months, smoke):
    rows, reasons = [], {}
    for strategy in strategy_ids:
        history = accounts[strategy]
        if smoke:
            values = {k: None for k in COLUMNS['metrics.csv'][3:]}
            values.update(n_months=len(months), start_month=months[0], end_month=months[-1])
            undefined = {k: 'smoke_independent_months_not_a_continuous_performance_series' for k, v in values.items() if v is None}
            rows.append({'run_id': run_id, 'strategy_id': strategy, **values})
            reasons[strategy] = undefined
        else:
            result = compute_metrics(months, [a.net_return for a in history],
                **{key: [getattr(a, key) for a in history] for key in ('turnover', 'transaction_cost', 'borrow_cost', 'financing_cost', 'gross_exposure')})
            rows.append(result.to_record(run_id, strategy)); reasons[strategy] = result.reasons
    return rows, reasons


def _check_forecast_panel(records, conditional, state, models, run_id):
    decision_month, partition_id = state.decision_month, state.partition_id
    regime_ids, next_probability = state.regime_ids, state.next_probability
    expected = {(model, ticker) for model in models for ticker in TICKERS}
    found = {(r['model'], r['ticker']) for r in records}
    modal = regime_ids[int(np.argmax(next_probability))]
    expected_conditional = {(model, ticker, regime) for model in models for ticker in TICKERS
                            for regime in (regime_ids[1:] if model == 'ridge' else [modal] if model in ('naive', 'bl') else [])}
    found_conditional = {(r['model'], r['ticker'], r['regime_id']) for r in conditional}
    if found != expected or len(records) != len(expected) or found_conditional != expected_conditional or len(conditional) != len(expected_conditional):
        raise ResearchError(ErrorCode.MISSING_DATA, 'forecast output has missing, duplicated or unexpected model/ticker/regime rows')
    for row in records + conditional:
        if row['run_id'] != run_id or row['decision_month'] != decision_month or row['partition_id'] != partition_id:
            raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'forecast output key differs from shared window')


def run_research(config: ResearchConfig, output, *, smoke=False, strategy_selection=None,
                 context=None, state_overrides=None, fixed_lambdas=None, experiment=None, use_prior_cache=False):
    """Execute every selected month/strategy into one new independent directory.

    smoke independently initializes each of the three evaluation months at NAV
    one; each retains its normal 48-row training history. No aggregate statistic
    is claimed. Full runs never drop a failed month. `state_overrides` and
    `fixed_lambdas`, when provided, must cover exactly every evaluated month.
    All effective experiment inputs are saved before any model is fitted.
    """
    validate_config(config)
    if use_prior_cache:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'prior run disk cache is unsupported; only explicitly shared in-process context is available')
    if config.portfolio.main_table_scaled:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'main tables must be unscaled; scaled results have a separate directory')
    months, strategies = selected_months(config, smoke), _strategies(config, strategy_selection)
    datasets = _dataset_ids(config)
    for label, mapping in [('state_overrides', state_overrides), ('fixed_lambdas', fixed_lambdas)]:
        if mapping is not None and not isinstance(mapping, dict):
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must be a month-keyed mapping')
    if fixed_lambdas is not None:
        if any(not isinstance(s, LambdaSelection) for s in fixed_lambdas.values()):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'fixed_lambdas requires LambdaSelection records')
        fixed_lambdas = {m: LambdaSelection(s.values.copy(deep=True), deepcopy(s.evidence)) for m, s in fixed_lambdas.items()}
    if state_overrides is not None:
        state_overrides = dict(state_overrides)
    if (state_overrides is not None or fixed_lambdas is not None) and not experiment:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'injected states or lambda selections require recorded experiment provenance')
    if config.research.validation == 'blocked' and 'ridge' in config.research.models and fixed_lambdas is None:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'blocked validation requires fixed_lambdas for every month')
    effective = {'research': asdict(config.research), 'portfolio': asdict(config.portfolio), 'dataset_ids': datasets,
                 'strategy_ids': list(strategies), 'evaluated_months': months, 'smoke': smoke,
                 'experiment': deepcopy(experiment or {}), 'prior_run_cache': False,
                 'state_overrides': {m: s.partition_id for m, s in (state_overrides or {}).items()},
                 'lambda_selections': {m: s.record(m, decision_ledger(m, [add_months(m, -config.research.lag_months), '2023-02'],
                    profile=config.research.profile, lag_months=config.research.lag_months)['decision_at']) for m, s in (fixed_lambdas or {}).items()}}
    config_hash = canonical_id(effective)
    revision, code_hashes = _code_snapshot()
    environment = _environment()
    run_id = uuid.uuid4().hex
    writer = RunWriter(output, dict(schema_version=SCHEMA_VERSION, run_id=run_id, profile=config.research.profile,
        code_revision=revision, code_hashes=code_hashes, config=effective, config_hash=config_hash, dataset_ids=datasets,
        seed=config.research.seed, environment=environment,
        requested_period={'start_month': config.research.start_month, 'end_month': config.research.end_month},
        actual_period={'start_month': months[0], 'end_month': months[-1], 'evaluated_months': months, 'n_months': len(months),
                       'first_requested_month': config.research.start_month, 'first_actual_month': months[0],
                       'start_reason': 'requested start has complete 48-month context; no month skipped',
                       'mode': 'smoke_independent_months' if smoke else 'continuous'}, strategy_ids=list(strategies),
        warnings=['smoke months reset NAV and pretrade holdings independently; aggregate performance undefined'] if smoke else []))
    try:
        ctx = context if context is not None else ExecutionContext(config)
        ctx.require_config(config)
        actual_start = ctx.feasible_start(config)
        months = selected_months(replace(config, research=replace(config.research, start_month=actual_start)), smoke)
        for label, mapping in [('state_overrides', state_overrides), ('fixed_lambdas', fixed_lambdas)]:
            if mapping is not None and set(mapping) != set(months):
                raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must match actual months after listing warmup')
        effective['evaluated_months'] = months
        writer.manifest['config_hash'] = canonical_id(effective)
        writer.manifest['actual_period'].update(start_month=months[0], first_actual_month=months[0], evaluated_months=months, n_months=len(months),
            start_reason='listing/public-vintage warmup: retain 48 complete target months' if actual_start != config.research.start_month
                         else 'requested start has complete 48-month context; no month skipped')
        before = dict(ctx.counts)
        writer.json('effective_config.json', effective)
        writer.json('accounting_policy.json', ACCOUNTING_POLICY)
        windows, probabilities, conditional, forecasts, ledgers, scaling_rows = [], [], [], [], [], []
        tables = {prefix: {'weights': [], 'returns': [], 'accounts': {s: [] for s in strategies}} for prefix in ('', 'scaled/')}
        # This panel is never passed wholesale into model/scaling functions.
        all_returns = ctx.returns(add_months(months[0], -49), months[-1])
        writer.jsonl('asset_returns.jsonl', [{'holding_month': m, 'returns': row.to_dict()} for m, row in all_returns.iterrows()])
        with threadpool_limits(1):
            for month in months:
                state = ctx.state(config, month)
                if state_overrides is not None:
                    state = _override_state(state, state_overrides[month])
                ledger = state.ledger
                training = all_returns.loc[state.labels.index, list(TICKERS)].copy()
                selection = fixed_lambdas[month] if fixed_lambdas is not None else None
                model = ctx.forecast(state, training, config.research, selection)
                state.require_identity(model.partition_id, decision_month=model.decision_month)
                windows.append(_window_record(run_id, state, model))
                probabilities.extend(state.probability_records(run_id))
                current_conditional, current_forecasts = model.conditional_records(run_id), model.forecast_records(run_id)
                _check_forecast_panel(current_forecasts, current_conditional, state, config.research.models, run_id)
                conditional.extend(current_conditional); forecasts.extend(current_forecasts)
                ledgers.append({'run_id': run_id, 'partition_id': state.partition_id, **ledger})
                state.save(writer.path / 'states' / month)
                model.save(writer.path / 'models' / month)
                scores = model.scores
                next_probability = state.next_probability
                end_dates = pd.Series([r['target_end_at'] for r in ledger['training_rows']], index=training.index)
                realized = all_returns.loc[month, list(TICKERS)]
                for strategy in strategies:
                    if strategy in ('spy', 'ew'):
                        base = benchmark_weights(strategy)
                    else:
                        name, sizing, size = strategy.split('_')
                        base = allocate(scores[name], sizing, int(size), next_probabilities=next_probability)
                    scaled = scale_to_volatility(base, training, decision_month=month, decision_at=ledger['decision_at'],
                        return_end_at=end_dates, lookback=config.portfolio.vol_lookback, vol_target=config.portfolio.vol_target,
                        gross_cap=config.portfolio.gross_cap)
                    scaling_rows.append({'run_id': run_id, 'decision_month': month, 'strategy_id': strategy,
                                         'expected_volatility': scaled.expected_volatility, 'multiplier': scaled.multiplier,
                                         'used_months': list(scaled.used_months), 'vol_target': scaled.vol_target, 'gross_cap': scaled.gross_cap})
                    for prefix, target in [('', base), ('scaled/', scaled.target_weights)]:
                        table = tables[prefix]; history = table['accounts'][strategy]
                        account = account_month(target, realized, holding_month=month, decision_at=ledger['decision_at'],
                            execution_at=ledger['execution_at'], target_end=ledger['target_return_end_at'],
                            previous=history[-1] if history and not smoke else None, settings=config.portfolio, base_weights=base)
                        history.append(account)
                        table['weights'].extend(account.weight_records(run_id, strategy))
                        table['returns'].append(account.to_record(run_id, strategy))
        writer.jsonl('windows.jsonl', windows, WINDOW_COLUMNS)
        writer.jsonl('timing_ledgers.jsonl', ledgers)
        writer.jsonl('scaled/scaling.jsonl', scaling_rows)
        for name, records in [('probabilities', probabilities), ('conditional_forecasts', conditional), ('forecasts', forecasts)]:
            writer.csv(name + '.csv', records)
        for prefix, table in tables.items():
            if len(table['returns']) != len(months) * len(strategies) or len(table['weights']) != 11 * len(months) * len(strategies):
                raise ResearchError(ErrorCode.MISSING_DATA, 'incomplete month/strategy output panel')
            writer.csv(prefix + 'weights.csv', table['weights']); writer.csv(prefix + 'returns.csv', table['returns'])
            metrics, reasons = _metric_rows(run_id, strategies, table['accounts'], months, smoke)
            writer.csv(prefix + 'metrics.csv', metrics); writer.json(prefix + 'metrics_reasons.json', reasons)
        ctx.require_config(config)
        if _code_snapshot() != (revision, code_hashes):
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'code changed during the run')
        writer.json('execution_context.json', {'shared_context_supplied': context is not None,
            'cache_policy': 'process_memory_only; no prior run files read', 'before': before, 'after': dict(ctx.counts),
            'this_run': {k: ctx.counts[k] - before[k] for k in before}})
        return writer.succeed()
    except BaseException as exc:
        writer.fail(exc)
        raise
