"""Keyed model inputs, past-only targets, fallback and artifact replay."""
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from regime_alloc.config import ResearchSettings, load_config
from regime_alloc.contracts import TICKERS, ResearchError
from regime_alloc.data import available_vintages, load_macro_vintage, load_prices
from regime_alloc.data.calendar import decision_ledger, monthly_returns
from regime_alloc.features import fit_preprocessor
from regime_alloc.features.windows import FeatureWindow, build_feature_window
from regime_alloc.regimes import build_window_state, build_inner_window_state
from regime_alloc.models import forecast_window, ModelForecasts


def synthetic_state(n=48, scope='rolling'):
    ledger = decision_ledger('2003-02', ['2002-12'], train_months=n)
    rows = pd.DataFrame(ledger['training_rows'])
    months = pd.period_range(rows.predictor_base_month.iloc[0], '2002-11', freq='M').astype(str)
    rng = np.random.default_rng(417)
    x = rng.normal(size=(n + 2, 4))
    x[-4:-1] += 20
    values = pd.DataFrame(x, index=months, columns=list('abcd'))
    prep = fit_preprocessor(values, rows.condition_base_month, dict.fromkeys(values, 1), pca_variance=1., scope=scope)
    features = FeatureWindow(prep, prep.fit_result, prep.transform(values, rows.predictor_base_month), prep.transform(values, [ledger['current_base_month']]), rows, ledger)
    with threadpool_limits(1):
        state = build_window_state(features, scope=scope) if n == 48 else build_inner_window_state(features)
    returns = pd.DataFrame(rng.normal(.01, .03, (n, 10)), index=rows.target_month.tolist(), columns=TICKERS)
    return state, returns


def run(state, returns, **kwargs):
    return forecast_window(state, returns, partition_id=state.partition_id, decision_month=state.decision_month, **kwargs)


def test_all_models_same_keys_and_conditional_contract(tmp_path):
    state, returns = synthetic_state()
    result = run(state, returns)
    assert list(result.scores) == ['naive', 'ridge', 'bl', 'mvo']
    assert all(list(v.index) == list(TICKERS) for v in result.scores.values())
    rows = result.conditional_records('run')
    assert {model: sum(r['model'] == model for r in rows) for model in result.scores} == {'naive': 10, 'ridge': 50, 'bl': 10, 'mvo': 0}
    assert all(r['partition_id'] == state.partition_id for r in rows)
    assert all(r['regime_id'] != 'R0' for r in rows if r['model'] == 'ridge')
    bl = [r for r in rows if r['model'] == 'bl']
    modal = int(np.argmax(state.next_probability))
    selected = returns.loc[state.labels.eq(modal)]
    expected_q = selected.mean().to_numpy() if len(selected) >= 2 else returns.mean().to_numpy()
    np.testing.assert_allclose([r['value'] for r in bl], expected_q)
    assert result.metadata['conditional_semantics']['bl'] == 'selected_regime_view_q_expected_return'
    assert result.metadata['conditional_semantics']['mvo'] == 'unconditional_no_rows'
    np.testing.assert_allclose(result.arrays['bl_q'], expected_q)
    result.save(tmp_path / 'model')
    loaded = ModelForecasts.load(tmp_path / 'model')
    assert result.model_id == loaded.model_id
    assert result.forecast_records('run') == loaded.forecast_records('run')
    for name, value in result.arrays.items(): np.testing.assert_array_equal(value, loaded.arrays[name])
    with pytest.raises(ResearchError): result.save(tmp_path / 'model')


def test_pooled_fallback_raw_and_effective_counts_and_mean_variant():
    state, returns = synthetic_state()
    settings = replace(ResearchSettings(), min_regime_samples=49)
    result = run(state, returns, settings=settings)
    for row in result.conditional_records('run'):
        if row['model'] == 'ridge':
            assert row['fallback'] == 'pooled' and row['effective_n'] == 48
            assert row['raw_n'] == int(state.labels.eq(int(row['regime_id'][1:])).sum())
    means = run(state, returns, settings=replace(settings, fallback='regime_mean'))
    for row in means.conditional_records('run'):
        if row['model'] == 'ridge':
            selected = returns.loc[state.labels.eq(int(row['regime_id'][1:]))]
            assert row['value'] == pytest.approx(selected[row['ticker']].mean())
            assert row['selected_lambda'] is None and row['reason']


def test_fixed_external_lambda_shared_across_regimes_and_blocked_requires_it():
    state, returns = synthetic_state()
    settings = replace(ResearchSettings(), validation='blocked')
    with pytest.raises(ResearchError, match='fixed_lambdas'): run(state, returns, settings=settings)
    fixed = pd.Series(np.linspace(.1, 1., 10), index=TICKERS)
    result = run(state, returns, settings=settings, fixed_lambdas=fixed)
    for row in result.conditional_records('run'):
        if row['model'] == 'ridge': assert row['selected_lambda'] == fixed[row['ticker']]
    with pytest.raises(ResearchError): run(state, returns, fixed_lambdas=fixed.iloc[::-1])


def test_mvo_scores_independent_of_membership_partition():
    state, returns = synthetic_state()
    other = state.with_labels(np.roll(state.labels.to_numpy(), 8), provenance={'test': 'permutation'})
    original = run(state, returns)
    control = run(other, returns)
    np.testing.assert_array_equal(original.scores['mvo'], control.scores['mvo'])
    assert original.partition_id != control.partition_id


def test_reordered_future_duplicate_nonfinite_and_partition_inputs_rejected():
    state, returns = synthetic_state()
    cases = [returns.iloc[::-1], returns.iloc[:, ::-1], returns.iloc[:-1], pd.concat([returns, returns.iloc[:1]])]
    future = returns.copy(); future.index = [*returns.index[:-1], '2003-01']; cases.append(future)
    bad = returns.copy(); bad.iloc[0, 0] = np.nan; cases.append(bad)
    for candidate in cases:
        with pytest.raises(ResearchError): run(state, candidate)
    with pytest.raises(ResearchError): forecast_window(state, returns, partition_id='wrong', decision_month=state.decision_month)
    with pytest.raises(ResearchError): forecast_window(state, returns, partition_id=state.partition_id, decision_month='2003-03')


def test_inner_fold_requires_explicit_adapter_and_full_sample_always_rejected():
    state, returns = synthetic_state(23)
    with pytest.raises(ResearchError): run(state, returns)
    result = run(state, returns, allow_inner=True)
    assert result.metadata['training_n'] == 23
    full, returns = synthetic_state(scope='full_sample')
    with pytest.raises(ResearchError): run(full, returns, allow_inner=True)


def model_boundary_state(state, *, labels=None, probability=None, change_end=False):
    # Synthetic consumer-boundary fixture: this deliberately does not claim
    # that Algorithm 1 or transition estimation produced these edge cases.
    from regime_alloc.regimes.state import _freeze
    metadata = state.metadata
    arrays = {key: value.copy() for key, value in state._arrays.items()}
    metadata['membership_provenance'] = {'method': 'synthetic_model_boundary_fixture'}
    if labels is not None: arrays['labels'] = np.asarray(labels)
    if probability is not None: arrays['next_probability'] = np.asarray(probability, dtype=float)
    if change_end:
        metadata['ledger']['training_rows'][-1]['target_end_at'] = metadata['ledger']['execution_at']
        metadata['training_rows'][-1]['target_end_at'] = metadata['ledger']['execution_at']
    return _freeze(metadata, arrays)


def test_r0_cash_and_sparse_selected_regime_fallbacks_in_adapter():
    state, returns = synthetic_state()
    crisis = model_boundary_state(state, probability=[1., 0., 0., 0., 0., 0.])
    result = run(crisis, returns)
    np.testing.assert_array_equal(result.scores['ridge'], np.zeros(10))
    assert all(row['reason'] == 'normal_regime_probability_zero_cash' for row in result.forecast_records('x') if row['model'] == 'ridge')
    # A one-observation modal view must return the prior, while Naive remains
    # explicitly undefined; an absent normal regime uses the whole-window mean.
    labels = state.labels.to_numpy()
    labels[labels == 1] = 0
    labels[0] = 1
    single = model_boundary_state(state, labels=labels, probability=[0., 1., 0., 0., 0., 0.])
    result = run(single, returns)
    np.testing.assert_allclose(result.arrays['bl_posterior_mean'], returns.mean())
    assert all(row['fallback'] == 'prior_mean' and row['raw_n'] == 1 for row in result.conditional_records('x') if row['model'] == 'bl')
    assert all(row['status'] == 'undefined' for row in result.forecast_records('x') if row['model'] == 'naive')
    labels[labels == 1] = 0
    empty = model_boundary_state(state, labels=labels, probability=[0., 1., 0., 0., 0., 0.])
    result = run(empty, returns, settings=replace(ResearchSettings(), fallback='regime_mean'))
    for row in result.conditional_records('x'):
        if row['model'] == 'ridge' and row['regime_id'] == 'R1':
            assert row['fallback'] == 'pooled_mean' and row['raw_n'] == 0 and row['effective_n'] == 48
            assert row['value'] == pytest.approx(returns[row['ticker']].mean())


def test_future_target_end_rejected_even_with_consistent_new_state_hash():
    state, returns = synthetic_state()
    future = model_boundary_state(state, change_end=True)
    with pytest.raises(ResearchError, match='timing ledger'): run(future, returns)


def test_saved_state_tamper_and_public_copy_safety(tmp_path):
    state, returns = synthetic_state()
    result = run(state, returns)
    original = result.scores['ridge'].copy()
    public_scores = result.scores
    public_scores['ridge'].iloc[0] = 999.
    result.metadata['training_n'] = 0
    np.testing.assert_array_equal(result.scores['ridge'], original)
    result.save(tmp_path / 'good')
    path = tmp_path / 'good' / 'models.json'
    content = json.loads(path.read_text(encoding='utf-8'))
    content['metadata']['training_n'] = 0
    path.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ResearchError): ModelForecasts.load(tmp_path / 'good')


@pytest.mark.real_data
@pytest.mark.parametrize('profile', ['vintage_lagged', 'fixed_snapshot'])
def test_both_actual_first_windows_all_models(profile):
    data = os.environ.get('REGIME_DATA_ROOT')
    if not data: pytest.skip('immutable market snapshot not configured')
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/data.toml')
    settings = replace(cfg.research, profile=profile)
    ledger = decision_ledger('2003-02', available_vintages(data, cfg.data.fred_dataset_id), profile=profile)
    vintage = load_macro_vintage(data, ledger['vintage_month'], dataset_id=cfg.data.fred_dataset_id)
    with threadpool_limits(1):
        state = build_window_state(build_feature_window(vintage, ledger, settings), settings)
        prices = load_prices(data, cfg.data.yahoo_dataset_id)
        returns = monthly_returns(prices, '1999-01', '2002-12')
        result = run(state, returns, settings=settings)
    assert len(result.forecast_records('real')) == 40
    assert all(np.isfinite(v).all() for v in result.scores.values())
