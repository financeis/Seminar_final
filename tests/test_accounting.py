import json

import numpy as np
import pandas as pd
import pytest

from regime_alloc.backtest.accounting import account_month, holding_days
from regime_alloc.config import PortfolioSettings
from regime_alloc.contracts import ResearchError


SMALL = ('A', 'B')


def account(w, r, **kwargs):
    args = dict(holding_month='2023-03', decision_at='2023-03-01T09:00:00-05:00',
        execution_at='2023-03-01T16:00:00-05:00', target_end='2023-04-03T16:00:00-04:00', tickers=SMALL)
    args.update(kwargs)
    return account_month(w, r, **args)


def test_initial_wealth_short_proceeds_and_zero_cost_self_financing():
    x = account([.6, -.4], [.10, -.05])
    assert x.pretrade_weights.tolist() == [0, 0, 1]
    assert x.target_weights.tolist() == pytest.approx([.6, -.4, .8])
    assert x.turnover == 1
    assert x.gross_return == pytest.approx(.08)
    assert x.net_return == pytest.approx(.08)
    assert x.wealth == pytest.approx(1.08)
    assert x.end_weights.tolist() == pytest.approx([.66/1.08, -.38/1.08, .8/1.08])
    assert x.end_weights.sum() == pytest.approx(1)
    assert x.gross_exposure == 1 and x.net_exposure == pytest.approx(.2)


def test_costs_are_deducted_from_month_end_cash_and_drift_drives_turnover():
    settings = PortfolioSettings(transaction_cost_bps=10, borrow_rate=.02, financing_rate=.02, cash_rate=.03)
    x = account([1.5, -.5], [.10, -.10], settings=settings, base_weights=[.75, -.25])
    assert x.holding_days == 33
    assert x.gross_return == pytest.approx(.20)
    assert x.transaction_cost == .002
    assert x.borrow_cost == pytest.approx(.5*.02*33/365)
    assert x.financing_cost == x.cash_return == 0
    end_wealth = 1.2-.002-.01*33/365
    assert x.wealth == pytest.approx(end_wealth)
    assert x.end_weights.tolist() == pytest.approx([1.65/end_wealth, -.45/end_wealth, (-.002-.01*33/365)/end_wealth])
    y = account([1, 0], [0, 0], holding_month='2023-04', decision_at='2023-04-03T09:00:00-04:00',
        execution_at=x.target_end, target_end='2023-05-01T16:00:00-04:00', previous=x, settings=settings)
    assert y.pretrade_weights.tolist() == pytest.approx(x.end_weights.tolist())
    assert y.turnover == pytest.approx(abs(1-1.65/end_wealth)+.45/end_wealth)
    assert y.transaction_cost == pytest.approx(y.turnover*.001)
    assert y.wealth == pytest.approx(end_wealth*(1-y.transaction_cost))
    assert y.end_weights.sum() == pytest.approx(1)


def test_positive_cash_interest_and_negative_cash_financing():
    settings = PortfolioSettings(cash_rate=.03, financing_rate=.02)
    cash = account([0, 0], [.1, -.1], settings=settings)
    assert cash.cash_return == pytest.approx(.03*33/365)
    assert cash.end_weights.tolist() == [0, 0, 1]
    short = account([-.5, -.5], [0, 0], settings=settings)
    assert short.cash_return == pytest.approx(2*.03*33/365)
    leverage = account([2, 0], [0, 0], settings=settings, base_weights=[1, 0])
    assert leverage.financing_cost == pytest.approx(.02*33/365)
    assert leverage.cash_return == 0
    assert leverage.end_weights.sum() == pytest.approx(1)


def test_act365_uses_new_york_calendar_dates_across_dst():
    assert holding_days('2023-03-01T21:00:00+00:00', '2023-04-03T20:00:00+00:00') == 33
    assert holding_days('2023-11-01T16:00:00-04:00', '2023-12-01T16:00:00-05:00') == 30


def test_bankruptcy_is_structured_error_and_has_no_log_return():
    with pytest.raises(ResearchError) as error:
        account([2, 0], [-.5, 0], base_weights=[1, 0])
    assert error.value.details['bankruptcy_month'] == '2023-03'
    assert error.value.details['wealth'] == 0
    assert 'log_return' not in error.value.details


@pytest.mark.parametrize('w,r', [([1, 0], [np.nan, 0]), ([1, 0], [0, np.nan]),
    ([np.inf, 0], [0, 0]), ([3, 0], [0, 0]), ([1, 0], [-1.01, 0]), ([1], [0, 0])])
def test_full_panel_nonfinite_and_out_of_range_inputs_fail(w, r):
    with pytest.raises(ResearchError):
        account(w, r)


def test_month_order_and_alignment_are_not_silently_repaired():
    x = account([1, 0], [0, 0])
    with pytest.raises(ResearchError):
        account([1, 0], [0, 0], previous=x)
    with pytest.raises(ResearchError):
        account([1, 0], [0, 0], execution_at='2023-02-01T16:00:00-05:00')
    with pytest.raises(ResearchError):
        account(pd.Series([1, 0], index=['A', 'A']), [0, 0])


def test_artifact_records_expose_cash_and_unique_asset_keys():
    x = account([1, 0], [.1, 0], base_weights=[.5, 0])
    rows = x.weight_records('run', 'naive_lo_2')
    assert [r['ticker'] for r in rows] == ['A', 'B', 'CASH']
    assert rows[-1] == dict(run_id='run', decision_month='2023-03', strategy_id='naive_lo_2', ticker='CASH',
        base_weight=.5, target_weight=0., pretrade_weight=1.)
    record = x.to_record('run', 'naive_lo_2')
    assert record['holding_month'] == '2023-03' and record['net_return'] == pytest.approx(.1)
    json.dumps(record, allow_nan=False)


def test_leveraged_target_requires_explicit_unscaled_base_for_honest_weight_records():
    with pytest.raises(ResearchError, match='base'):
        account([2, 0], [0, 0])


def test_losing_first_month_and_rebound_match_metrics_drawdown():
    x = account([1, 0], [-.1, 0])
    y = account([1, 0], [.05, 0], holding_month='2023-04', decision_at='2023-04-03T09:00:00-04:00',
        execution_at=x.target_end, target_end='2023-05-01T16:00:00-04:00', previous=x)
    assert x.drawdown == pytest.approx(-.1)
    assert y.wealth == pytest.approx(.945)
    assert y.drawdown == pytest.approx(-.055)


def test_default_complete_ticker_panel_applies_even_to_benchmarks():
    with pytest.raises(ResearchError, match='dimensions'):
        account_month([1, 0], [0, 0], holding_month='2023-03', decision_at='2023-03-01T09:00:00-05:00',
            execution_at='2023-03-01T16:00:00-05:00', target_end='2023-04-03T16:00:00-04:00')
