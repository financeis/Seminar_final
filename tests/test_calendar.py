import pandas as pd
import pytest
from regime_alloc.contracts import TICKERS, ResearchError
from regime_alloc.data.calendar import first_session, decision_ledger, monthly_returns


def test_nyse_first_session():
    assert first_session('2003-02') == '2003-02-03'
    assert first_session('2023-01') == '2023-01-03'
    assert first_session('2001-09') == '2001-09-04'


def test_february_2003_independent_calendar_ledger():
    x = decision_ledger('2003-02', ['2002-11', '2002-12', '2003-01'])
    assert x['vintage_month'] == '2002-12'
    assert x['decision_at'] == '2003-02-03T09:00:00-05:00'
    assert x['current_base_month'] == '2002-11'
    assert x['next_state_base_month'] == '2002-12'
    assert len(x['training_rows']) == 48
    assert x['training_rows'][0]['target_month'] == '1999-01'
    assert x['training_rows'][-1]['target_month'] == '2002-12'
    assert x['training_rows'][0]['condition_base_month'] == '1998-11'
    assert x['training_rows'][-1]['condition_base_month'] == '2002-10'
    assert x['training_rows'][0]['predictor_base_month'] == '1998-10'
    assert x['training_rows'][-1]['target_end_at'] == '2003-01-02T16:00:00-05:00'
    assert all(c['status'] == 'pass' for c in x['checks'])


def test_fixed_never_backdates_availability():
    x = decision_ledger('2003-02', ['2023-02'], profile='fixed_snapshot')
    assert x['assumed_available_at'] == '2023-04-01T00:00:00-04:00'
    check = next(c for c in x['checks'] if c['name'] == 'macro_vintage_available_before_decision')
    assert check['status'] == 'not_applicable' and check['reason']


def test_returns_require_exact_first_session_and_finalized():
    rows = [{'session': date, 'ticker': ticker, 'adj_close': price, 'finalized': True} for date, price in [('2022-12-01', 100), ('2023-01-03', 105)] for ticker in TICKERS]
    p = pd.DataFrame(rows)
    assert monthly_returns(p, '2022-12', '2022-12').iloc[0].tolist() == pytest.approx([.05]*10)
    p.loc[0, 'session'] = '2022-12-02'
    with pytest.raises(ResearchError, match='calendar_gap'):
        monthly_returns(p, '2022-12', '2022-12')
    p.loc[0, 'session'] = '2022-12-01'
    p.loc[0, 'finalized'] = False
    with pytest.raises(ResearchError, match='missing_data'):
        monthly_returns(p, '2022-12', '2022-12')
