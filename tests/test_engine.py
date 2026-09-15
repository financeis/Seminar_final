from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import numpy as np
import pandas as pd
import pytest
from regime_alloc.config import load_config
from regime_alloc.contracts import ResearchError
from regime_alloc.backtest.engine import ExecutionContext, LambdaSelection, run_research, selected_months
from regime_alloc.data.calendar import decision_ledger
from regime_alloc.contracts import TICKERS
from threadpoolctl import threadpool_limits


def test_smoke_selection_and_config_rejection(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/data.toml')
    assert selected_months(cfg, True) == ['2003-02', '2020-04', '2022-12']
    with pytest.raises(ResearchError, match='strategy'):
        run_research(cfg, tmp_path / 'bad', strategy_selection=['unknown'])
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(ResearchError, match='prior'):
        run_research(cfg, tmp_path / 'bad', use_prior_cache=True)


@pytest.fixture(scope='module')
def actual_config():
    root = os.environ.get('REGIME_DATA_ROOT')
    if not root: pytest.skip('immutable market snapshot not configured')
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/data.toml')
    return replace(cfg, data=replace(cfg.data, root=Path(root)))


@pytest.mark.real_data
@pytest.mark.parametrize('profile', ['fixed_snapshot', 'vintage_lagged'])
def test_actual_smoke_complete_accounting_offline(actual_config, profile, tmp_path, monkeypatch):
    def offline(*a, **k): raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', offline)
    cfg = replace(actual_config, research=replace(actual_config.research, profile=profile))
    result = run_research(cfg, tmp_path / profile, smoke=True)
    assert result['status'] == 'succeeded' and len(result['strategy_ids']) == 50
    path = tmp_path / profile
    windows = [json.loads(x) for x in (path / 'windows.jsonl').read_text().splitlines()]
    assert len(windows) == 3 and all(len(x['train_months']) == 48 for x in windows)
    probabilities = pd.read_csv(path / 'probabilities.csv')
    assert len(probabilities) == 18
    np.testing.assert_allclose(probabilities.groupby('decision_month').next_probability.sum(), 1.)
    assert len(pd.read_csv(path / 'forecasts.csv')) == 120
    assert len(pd.read_csv(path / 'conditional_forecasts.csv')) == 210
    for prefix in ['', 'scaled/']:
        r = pd.read_csv(path / (prefix + 'returns.csv'))
        w = pd.read_csv(path / (prefix + 'weights.csv'))
        assert len(r) == 150 and len(w) == 1650
        np.testing.assert_allclose(r.net_return, r.gross_return + r.cash_return - r.transaction_cost - r.borrow_cost - r.financing_cost)
        np.testing.assert_allclose(w.groupby(['decision_month', 'strategy_id']).target_weight.sum(), 1.)
        assert pd.read_csv(path / (prefix + 'metrics.csv')).cagr.isna().all()
    with pytest.raises(ResearchError, match='run_conflict'):
        run_research(cfg, path, smoke=True)


@pytest.mark.real_data
def test_continuous_accounting_independent_oracle_and_warmup(actual_config, tmp_path):
    cfg = replace(actual_config, research=replace(actual_config.research, start_month='2000-01', end_month='2003-04'),
                  portfolio=replace(actual_config.portfolio, transaction_cost_bps=10, cash_rate=.015, borrow_rate=.02, financing_rate=.02, vol_target=.6))
    # Explicit stress configuration: target 60% forces capped leverage, while
    # signed Ridge selects shorts. This is not a paper/main-performance run.
    result = run_research(cfg, tmp_path / 'warmup', strategy_selection=['ridge_lns_2', 'ew'])
    assert result['actual_period']['start_month'] == '2003-02'
    assert result['actual_period']['n_months'] == 3 and 'warmup' in result['actual_period']['start_reason']
    path = tmp_path / 'warmup'
    assets = {x['holding_month']: x['returns'] for x in map(json.loads, (path / 'asset_returns.jsonl').read_text().splitlines())}
    for prefix in ['', 'scaled/']:
        rows = pd.read_csv(path / (prefix + 'returns.csv'))
        assert rows.borrow_cost.sum() > 0
        if prefix: assert rows.financing_cost.sum() > 0
        weights = pd.read_csv(path / (prefix + 'weights.csv'))
        for strategy, history in rows.groupby('strategy_id', sort=False):
            previous = pd.Series([0.] * 10 + [1.], index=[*TICKERS, 'CASH'])
            nav = peak = 1.
            for row in history.itertuples():
                block = weights.loc[(weights.strategy_id == strategy) & (weights.decision_month == row.holding_month)].set_index('ticker')
                w = block.loc[list(TICKERS), 'target_weight']; cash = block.loc['CASH', 'target_weight']
                np.testing.assert_allclose(block.pretrade_weight.reindex(previous.index), previous, atol=1e-14)
                turnover = (w - previous.loc[list(TICKERS)]).abs().sum()
                days = (pd.Timestamp(row.target_end).date() - pd.Timestamp(row.execution_at).date()).days
                cost = turnover * .001
                loan = -w.clip(upper=0).sum() * .02 * days / 365
                finance = max(-cash, 0) * .02 * days / 365
                interest = max(cash, 0) * .015 * days / 365
                asset = pd.Series(assets[row.holding_month]).reindex(TICKERS)
                net = w @ asset + interest - cost - loan - finance
                nav *= 1 + net; peak = max(peak, nav)
                assert row.net_return == pytest.approx(net, abs=1e-14)
                assert row.turnover == pytest.approx(turnover, abs=1e-14)
                assert row.wealth == pytest.approx(nav, abs=1e-14)
                assert row.drawdown == pytest.approx(nav / peak - 1, abs=1e-14)
                previous = w * (1 + asset)
                previous.loc['CASH'] = cash + interest - cost - loan - finance
                previous /= 1 + net


@pytest.mark.real_data
def test_missing_forecast_or_weight_rows_fail(actual_config, tmp_path, monkeypatch):
    from regime_alloc.models import ModelForecasts
    from regime_alloc.backtest.accounting import MonthlyAccount
    cfg = replace(actual_config, research=replace(actual_config.research, end_month='2003-02'))
    original = ModelForecasts.forecast_records
    monkeypatch.setattr(ModelForecasts, 'forecast_records', lambda self, run_id: original(self, run_id)[1:])
    with pytest.raises(ResearchError, match='forecast'):
        run_research(cfg, tmp_path / 'missing-forecast', strategy_selection=['ew'])
    monkeypatch.setattr(ModelForecasts, 'forecast_records', original)
    weights = MonthlyAccount.weight_records
    monkeypatch.setattr(MonthlyAccount, 'weight_records', lambda self, *a: weights(self, *a)[1:])
    with pytest.raises(ResearchError, match='incomplete'):
        run_research(cfg, tmp_path / 'missing-weight', strategy_selection=['ew'])


@pytest.mark.real_data
def test_injected_membership_and_external_lambda_evidence(actual_config, tmp_path):
    cfg = replace(actual_config, research=replace(actual_config.research, start_month='2000-01', end_month='2003-02', models=('naive', 'ridge'), validation='blocked'))
    context = ExecutionContext(cfg)
    with threadpool_limits(1):
        base = context.state(cfg, '2003-02')
        control = base.with_labels(np.roll(base.labels.to_numpy(), 3), provenance={'method': 'test_only_permutation', 'seed': 3})
    value = pd.Series(np.arange(1, 11) / 10, index=TICKERS)
    evidence = {'source': 'synthetic_boundary_fixed_values_not_forward_validation', 'decision_month': '2003-02',
                'selection_end_at': base.training_rows.target_end_at.max(), 'tickers': list(TICKERS)}
    selection = LambdaSelection(value, evidence)
    result = run_research(cfg, tmp_path / 'injected', context=context, strategy_selection=['naive_lo_2', 'ridge_lo_2'],
                         state_overrides={'2003-02': control}, fixed_lambdas={'2003-02': selection}, experiment={'test': 'injection_contract'})
    cond = pd.read_csv(tmp_path / 'injected' / 'conditional_forecasts.csv')
    assert len(cond) == 60 and len(result['strategy_ids']) == 2
    assert set(cond.partition_id) == {control.partition_id}
    for row in cond.loc[cond.model == 'ridge'].itertuples(): assert row.selected_lambda == pytest.approx(value[row.ticker])
    wrong = replace(selection, evidence={**evidence, 'selection_end_at': base.ledger['execution_at']})
    with pytest.raises(ResearchError, match='unavailable'):
        run_research(cfg, tmp_path / 'future-lambda', fixed_lambdas={'2003-02': wrong}, experiment={'test': True})
    with pytest.raises(ResearchError, match='child'):
        run_research(cfg, tmp_path / 'bad-parent', context=context, state_overrides={'2003-02': base},
                     fixed_lambdas={'2003-02': selection}, experiment={'test': True})
    assert json.loads((tmp_path / 'bad-parent' / 'run_manifest.json').read_text())['status'] == 'failed'


@pytest.mark.real_data
def test_cache_value_collision_private_mutation_and_public_copy(actual_config, tmp_path):
    cfg = replace(actual_config, research=replace(actual_config.research, end_month='2003-03'))
    context = ExecutionContext(cfg)
    with threadpool_limits(1):
        first = context.state(cfg, '2003-02'); second = context.state(cfg, '2003-03')
    key = next(k for k, v in context._states.items() if v.partition_id == first.partition_id)
    context._states[key] = second
    with pytest.raises(ResearchError): context.state(cfg, '2003-02')
    context._states[key] = first
    public = context.returns('1999-01', '2002-12'); public.iloc[0, 0] = 123.
    assert context.returns('1999-01', '2002-12').iloc[0, 0] != 123.
    return_key = next(iter(context._returns)); context._returns[return_key].iloc[0, 0] = 123.
    with pytest.raises(ResearchError, match='cache'): context.returns('1999-01', '2002-12')
    vintage = next(iter(context._raw.values())); vintage.values.iloc[0, 0] += 1.
    with pytest.raises(ResearchError, match='macro cache'): context.require_config(cfg)


@pytest.mark.real_data
def test_future_only_inputs_do_not_change_past_forecasts(actual_config, tmp_path, monkeypatch):
    from regime_alloc.backtest import engine
    cfg = replace(actual_config, research=replace(actual_config.research, end_month='2003-02'))
    original = run_research(cfg, tmp_path / 'original', strategy_selection=['ridge_lo_2'])
    price_loader, macro_loader = engine.load_prices, engine.load_macro_vintage
    def future_prices(*a, **kw):
        frame = price_loader(*a, **kw)
        frame.loc[frame.session >= '2003-02-01', 'adj_close'] *= 2
        return frame
    def future_macro(*a, **kw):
        vintage = macro_loader(*a, **kw)
        # Synthetic perturbation of post-cutoff rows in otherwise actual input.
        vintage.values.loc[vintage.values.index > '2002-11'] *= 3
        return vintage
    monkeypatch.setattr(engine, 'load_prices', future_prices)
    monkeypatch.setattr(engine, 'load_macro_vintage', future_macro)
    run_research(cfg, tmp_path / 'perturbed', strategy_selection=['ridge_lo_2'])
    for filename in ['forecasts.csv', 'weights.csv', 'scaled/weights.csv']:
        left = pd.read_csv(tmp_path / 'original' / filename).drop(columns='run_id')
        right = pd.read_csv(tmp_path / 'perturbed' / filename).drop(columns='run_id')
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.real_data
def test_missing_price_and_midwrite_failure_remain_failed(actual_config, tmp_path, monkeypatch):
    from regime_alloc.backtest import engine
    from regime_alloc.data.calendar import first_session
    cfg = replace(actual_config, research=replace(actual_config.research, end_month='2003-03'))
    loader = engine.load_prices
    def missing(*a, **kw):
        frame = loader(*a, **kw)
        return frame.loc[~((frame.session == first_session('2003-03')) & (frame.ticker == 'XLK'))].copy()
    monkeypatch.setattr(engine, 'load_prices', missing)
    with pytest.raises(ResearchError, match='missing'):
        run_research(cfg, tmp_path / 'missing', strategy_selection=['spy'])
    assert json.loads((tmp_path / 'missing' / 'run_manifest.json').read_text())['status'] == 'failed'
    monkeypatch.setattr(engine, 'load_prices', loader)
    def fail_write(self, name, records): raise OSError('injected middle-of-write failure')
    monkeypatch.setattr(engine.RunWriter, 'csv', fail_write)
    with pytest.raises(OSError, match='injected'):
        run_research(cfg, tmp_path / 'partial', strategy_selection=['spy'])
    manifest = json.loads((tmp_path / 'partial' / 'run_manifest.json').read_text())
    assert manifest['status'] == 'failed' and (tmp_path / 'partial' / 'states' / '2003-02' / 'state.npz').is_file()
    assert manifest['errors'][0]['error_code'] == 'missing_data'
    assert any(a['path'] == 'states/2003-02/state.npz' for a in manifest['artifacts'])
    with pytest.raises(ResearchError, match='run_conflict'):
        run_research(cfg, tmp_path / 'partial', strategy_selection=['spy'])
