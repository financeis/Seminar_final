import json

import numpy as np
import pandas as pd
import pytest

from regime_alloc.backtest.metrics import compute_metrics
from regime_alloc.contracts import ResearchError, write_json


def test_drawdown_includes_initial_one_and_both_averages():
    x = compute_metrics(['2023-01', '2023-02'], [-.10, .05], turnover=[1, .2],
        transaction_cost=[.01, .002], borrow_cost=[0, 0], financing_cost=[0, 0], gross_exposure=[1, 0])
    assert x.wealth.tolist() == pytest.approx([.9, .945])
    assert x.drawdown.tolist() == pytest.approx([-.10, -.055])
    v = x.values
    assert v['maxdd'] == pytest.approx(-.1)
    assert v['avgdd'] == pytest.approx(-.0775)
    assert v['avgdd_all_months'] == pytest.approx(-.0775)
    assert v['positive_ratio'] == .5 and v['n_months'] == 2
    assert v['cagr'] == pytest.approx(.945**6-1)
    assert v['ann_vol'] == pytest.approx(np.sqrt(.01125)*np.sqrt(12))
    assert v['sharpe'] == pytest.approx(-.025/np.sqrt(.01125)*np.sqrt(12))
    assert v['sortino'] == pytest.approx(-.025/np.sqrt(.01/2)*np.sqrt(12))
    assert v['mean_turnover'] == .6 and v['total_cost'] == pytest.approx(.012)
    assert v['cash_ratio'] == .5 and not x.reasons


def test_avgdd_excludes_recovered_month_and_zero_returns_are_not_positive():
    x = compute_metrics(['2023-01', '2023-02', '2023-03'], [-.2, .25, 0])
    assert x.values['maxdd'] == pytest.approx(-.2)
    assert x.values['avgdd'] == pytest.approx(-.2)
    assert x.values['avgdd_all_months'] == pytest.approx(-.2/3)
    assert x.values['positive_ratio'] == 1/3


def test_sortino_uses_all_month_downside_rms_and_excess_return():
    x = compute_metrics(['2023-01', '2023-02', '2023-03'], [-.09, .11, .11], risk_free_monthly=.01)
    assert x.values['sortino'] == pytest.approx((.1/3)/np.sqrt(.01/3)*np.sqrt(12))


@pytest.mark.parametrize('returns', [[.02], [.02, .02], [0, 0]])
def test_short_or_constant_samples_have_null_with_reasons(returns):
    months = pd.period_range('2023-01', periods=len(returns), freq='M').astype(str)
    x = compute_metrics(months, returns)
    assert x.values['sharpe'] is None and x.reasons['sharpe']
    assert x.values['sortino'] is None and x.reasons['sortino']
    if len(returns) == 1:
        assert x.values['ann_vol'] is None and x.reasons['ann_vol']
    assert x.values['maxdd'] == x.values['avgdd'] == 0
    json.dumps({'values': x.values, 'reasons': x.reasons}, allow_nan=False)


def test_constant_loss_has_valid_sortino_but_undefined_sharpe():
    x = compute_metrics(['2023-01', '2023-02'], [-.1, -.1])
    assert x.values['sharpe'] is None
    assert x.values['sortino'] == pytest.approx(-np.sqrt(12))


@pytest.mark.parametrize('months,returns', [
    (['2023-01', '2023-03'], [.1, .1]), (['2023-01', '2023-01'], [.1, .1]),
    (['2023-02', '2023-01'], [.1, .1]), (['2023-01'], [np.nan]),
    (['2023-01'], [-1.]), (['2023-01'], [-1.1]), (['2023-01'], [.1, .2]),
])
def test_invalid_panels_and_bankruptcy_fail(months, returns):
    with pytest.raises(ResearchError):
        compute_metrics(months, returns)


def test_short_proceeds_do_not_make_invested_month_a_cash_month():
    x = compute_metrics(['2023-01', '2023-02'], [.01, .02], gross_exposure=[1, 0])
    assert x.values['cash_ratio'] == .5


def test_missing_optional_inputs_are_explicitly_undefined():
    x = compute_metrics(['2023-01', '2023-02'], [.01, .02])
    for key in ('mean_turnover', 'total_cost', 'cash_ratio'):
        assert x.values[key] is None and x.reasons[key]


def test_empty_sample_and_export_obey_null_reason_contract(tmp_path):
    x = compute_metrics([], [])
    assert x.values['n_months'] == 0 and x.values['maxdd'] is None
    record = x.to_json_record('run', 'spy')
    write_json(tmp_path/'metrics.json', record)
    saved = json.loads((tmp_path/'metrics.json').read_text(encoding='utf-8'))
    assert saved['null_reasons']['maxdd'] == 'no_observations'
    assert saved['start_month'] is None


def test_bad_optional_columns_and_series_month_mismatch_fail():
    for kwargs in ({'turnover': [-1, 0]}, {'transaction_cost': [0, np.nan]}, {'gross_exposure': [1]},
                   {'borrow_cost': pd.Series([0, 0], index=['2023-01', '2023-03'])}):
        with pytest.raises(ResearchError):
            compute_metrics(['2023-01', '2023-02'], [.1, .1], **kwargs)
