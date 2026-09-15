"""Forward-validation dates and losses checked independently of fitting code."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import socket
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from regime_alloc.config import ResearchSettings, load_config
from regime_alloc.contracts import ResearchError, TICKERS
from regime_alloc.models.forward_validation import (
    forward_ledgers, candidate_forecasts, select_from_predictions, select_forward_lambdas,
)


def test_forward_monthly_timing_first_23_last_46_independent_calendar():
    settings = ResearchSettings(validation='blocked')
    vintages = [str(x) for x in pd.period_range('1998-01', '2003-01', freq='M')]
    ledgers = forward_ledgers('2003-02', vintages, settings)
    assert len(ledgers) == 24
    assert [len(x['training_rows']) for x in ledgers] == list(range(23, 47))
    for j, ledger in enumerate(ledgers):
        v = pd.Period('2001-01', freq='M') + j
        assert ledger['decision_month'] == str(v)
        assert ledger['vintage_month'] == str(v - 2)
        assert ledger['training_rows'][0]['target_month'] == '1999-01'
        assert ledger['training_rows'][-1]['target_month'] == str(v - 2)
        assert ledger['current_base_month'] == str(v - 3)
        decision = pd.Timestamp(ledger['decision_at'])
        assert decision.hour == 9
        assert pd.Timestamp(ledger['assumed_available_at']) < decision
        for row in ledger['training_rows']:
            target = pd.Period(row['target_month'], freq='M')
            assert row['condition_base_month'] == str(target - 2)
            assert row['predictor_base_month'] == str(target - 3)
            assert pd.Timestamp(row['target_end_at']) < decision
            assert pd.Timestamp(row['target_end_at']).month == (target + 1).month


def test_forward_final_etf_predictions_match_augmented_normal_equations():
    rng = np.random.default_rng(141)
    x = rng.normal(size=(35, 3)); y = rng.normal(size=(35, 10)) / 20
    labels = np.repeat(np.arange(6), [5, 8, 8, 8, 4, 2])
    q = np.array([.4, -.8, .2]); p = np.array([.5, .05, .1, .15, .1, .1])
    state = SimpleNamespace(predictor_scores=pd.DataFrame(x), current_scores=pd.DataFrame([q]),
                            labels=pd.Series(labels), next_probability=p, scope='rolling', metadata={'window_kind': 'inner_fold'})
    settings = ResearchSettings(lambda_min=.1, lambda_max=10., lambda_count=3)
    got, valid, diagnostics = candidate_forecasts(state, pd.DataFrame(y, columns=TICKERS), settings)
    expected = np.zeros((3, 10))
    for i, lam in enumerate([.1, 1., 10.]):
        for regime in range(1, 6):
            mask = labels == regime
            if mask.sum() < 6: mask = np.ones(len(x), dtype=bool)
            z = np.column_stack([np.ones(mask.sum()), x[mask]])
            beta = np.linalg.solve(z.T @ z + np.diag([0., lam, lam, lam]), z.T @ y[mask])
            expected[i] += p[regime] * (np.r_[1., q] @ beta)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)
    assert valid.all() and diagnostics['R4']['fallback'] == 'pooled'
    assert diagnostics['R5']['effective_n'] == 35
    with pytest.raises(ResearchError, match='target month order'):
        candidate_forecasts(state, pd.DataFrame(y, columns=TICKERS).iloc[::-1], settings)
    state.scope = 'full_sample'
    with pytest.raises(ResearchError, match='rolling inner_fold'):
        candidate_forecasts(state, pd.DataFrame(y, columns=TICKERS), settings)
    state.scope = 'rolling'; state.metadata = {'window_kind': 'outer'}
    with pytest.raises(ResearchError, match='rolling inner_fold'):
        candidate_forecasts(state, pd.DataFrame(y, columns=TICKERS), settings)


def test_forward_mse_selects_per_etf_on_24_final_predictions_and_ties():
    observed = np.arange(24 * 10, dtype=float).reshape(24, 10) / 100
    predictions = np.repeat(observed[:, None, :], 3, axis=1)
    # ETF0: lambda .1 wins; ETF1: lambda 1 wins; remaining ETFs tie -> 10.
    predictions[:, 1:, 0] += .2
    predictions[:, [0, 2], 1] += .3
    predictions[:, :, 2:] += .1
    chosen, records = select_from_predictions(predictions, observed, [.1, 1., 10.], np.ones((24, 3), dtype=bool))
    assert chosen.tolist() == [.1, 1.] + [10.] * 8
    for row in records:
        j = TICKERS.index(row['ticker']); i = [.1, 1., 10.].index(row['lambda'])
        expected = sum((float(predictions[t, i, j]) - float(observed[t, j])) ** 2 for t in range(24)) / 24
        assert row['mse'] == pytest.approx(expected)
        assert row['valid_months'] == 24


def test_forward_incomplete_candidate_excluded_and_no_valid_candidate_fails():
    prediction = np.zeros((24, 2, 10))
    valid = np.ones((24, 2), dtype=bool)
    valid[0, 1] = False
    selected, records = select_from_predictions(prediction, np.zeros((24, 10)), [1., 10.], valid)
    assert selected.tolist() == [1.] * 10
    assert all(row['mse'] is None and row['valid_months'] == 23 for row in records if row['lambda'] == 10.)
    valid[-1, 0] = False
    with pytest.raises(ResearchError, match='no valid forward candidate') as exc:
        select_from_predictions(prediction, np.zeros((24, 10)), [1., 10.], valid)
    assert len(exc.value.details['candidate_losses']) == 20
    assert all(row['mse'] is None for row in exc.value.details['candidate_losses'])


def test_forward_invalid_prediction_serializes_null_with_reason(tmp_path):
    from regime_alloc.contracts import write_json
    from regime_alloc.models.forward_validation import _prediction_records
    values = np.ones((2, 10)); values[1, 4] = np.nan
    records = _prediction_records(values, [True, False], [1., 10.])
    write_json(tmp_path / 'predictions.json', records)
    reloaded = json.loads((tmp_path / 'predictions.json').read_text())
    assert reloaded[0]['values'] == [1.] * 10
    assert reloaded[1]['values'] is None and reloaded[1]['reason']


def test_forward_fit_ignores_future_macro_values_and_query_not_in_training():
    from regime_alloc.data import MacroVintage
    from regime_alloc.models.forward_validation import _inner_state
    settings = ResearchSettings(validation='blocked')
    ledger = forward_ledgers('2003-02', ['2000-11'], settings)[0]
    rng = np.random.default_rng(171)
    months = pd.period_range('1998-01', '2002-12', freq='M').astype(str)
    values = pd.DataFrame(rng.normal(size=(len(months), 8)), index=months, columns=list('abcdefgh'))
    vintage = MacroVintage('2000-11', values, pd.Series(1, index=values.columns),
                            pd.Series(1, index=values.columns), {}, ledger['assumed_available_at'], 'synthetic_future_test')
    perturbed = values.copy()
    perturbed.loc[perturbed.index > ledger['macro_cutoff_base_month']] = 1e9
    with threadpool_limits(1):
        first = _inner_state(vintage, ledger, settings)
        second = _inner_state(replace(vintage, values=perturbed), ledger, settings)
    assert first.partition_id == second.partition_id
    assert first.transform_hash == second.transform_hash
    assert first.metadata['condition_months'] == [str(x) for x in pd.period_range('1998-11', '2000-09', freq='M')]
    assert first.current_scores.index.tolist() == ['2000-10']
    assert '2000-10' not in first.metadata['condition_months']


@pytest.mark.real_data
def test_real_forward_24_refits_past_only_evidence_and_outer_refit(tmp_path, monkeypatch):
    root = os.environ.get('REGIME_DATA_ROOT')
    if not root: pytest.skip('immutable market snapshot not configured')
    from regime_alloc.backtest.engine import ExecutionContext
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/data.toml')
    cfg = replace(cfg, data=replace(cfg.data, root=Path(root)), research=replace(cfg.research, validation='blocked', models=('ridge',)))
    def offline(*args, **kwargs): raise AssertionError('forward validation must stay offline')
    monkeypatch.setattr(socket.socket, 'connect', offline)
    context = ExecutionContext(cfg)
    with threadpool_limits(1):
        selection = select_forward_lambdas(cfg, context, '2003-02')
        outer = context.state(cfg, '2003-02')
        training = context.returns('1999-01', '2002-12')
        fitted = context.forecast(outer, training, cfg.research, selection)
    evidence = selection.record('2003-02', outer.ledger['decision_at'])['evidence']
    assert len(evidence['monthly_folds']) == 24 and len(evidence['folds']) == 4
    assert len({x['partition_id'] for x in evidence['monthly_folds']}) == 24
    assert [x['training_n'] for x in evidence['monthly_folds']] == list(range(23, 47))
    assert evidence['selection_end_at'] == '2003-01-02T16:00:00-05:00'
    assert all(x['valid_months'] == 24 for x in evidence['candidate_losses'])
    assert all(x['window_kind'] == 'inner_fold' for x in evidence['monthly_folds'])
    assert len(evidence['candidate_losses']) == 410
    # Independent scalar loss calculation over serialized real ETF forecasts.
    for row in evidence['candidate_losses']:
        candidate = evidence['lambdas'].index(row['lambda'])
        ticker = TICKERS.index(row['ticker'])
        errors = [(fold['candidate_predictions'][candidate]['values'][ticker] - fold['observed_returns'][row['ticker']]) ** 2
                  for fold in evidence['monthly_folds']]
        assert row['mse'] == pytest.approx(sum(errors) / 24, rel=1e-13)
    for fold in evidence['monthly_folds']:
        v = pd.Period(fold['validation_month'], freq='M')
        assert fold['train_start'] == '1999-01' and fold['train_end'] == str(v - 2)
        assert fold['vintage_month'] <= str(v - 2)
        assert pd.Timestamp(fold['training_target_end_at']) < pd.Timestamp(fold['decision_at'])
        assert pd.Timestamp(fold['assumed_available_at']) < pd.Timestamp(fold['decision_at'])
    assert fitted.metadata['training_n'] == 48
    assert tuple(selection.values.index) == TICKERS
    assert all(x['selected_lambda'] == selection.values[x['ticker']] for x in fitted.metadata['conditional_forecasts'])
    (tmp_path / 'forward-real-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
