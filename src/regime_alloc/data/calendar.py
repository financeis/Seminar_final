"""XNYS timing and monthly research assumptions; no IO or price substitution."""
from __future__ import annotations
from datetime import datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ..config import month_string
from ..contracts import TICKERS, ErrorCode, ResearchError, require_unique

ET = ZoneInfo('America/New_York')


def add_months(month: str, offset: int) -> str:
    return str(pd.Period(month_string(month), freq='M') + offset)


def month_range(start: str, end: str) -> list[str]:
    return [str(m) for m in pd.period_range(month_string(start), month_string(end), freq='M')]


@lru_cache(maxsize=4)
def nyse_calendar():
    return xcals.get_calendar('XNYS', start='1990-01-01', end='2035-12-31')


def first_session(month: str) -> str:
    m = pd.Period(month_string(month), freq='M')
    sessions = nyse_calendar().sessions_in_range(m.start_time, m.end_time.normalize())
    if not len(sessions):
        raise ResearchError(ErrorCode.CALENDAR_GAP, f'no XNYS session in {month}')
    return sessions[0].strftime('%Y-%m-%d')


def session_close(session: str) -> pd.Timestamp:
    try:
        return nyse_calendar().session_close(pd.Timestamp(session)).tz_convert(ET)
    except (ValueError, KeyError) as exc:
        raise ResearchError(ErrorCode.CALENDAR_GAP, f'not an XNYS session: {session}') from exc


def decision_at(month: str) -> datetime:
    return datetime.combine(datetime.fromisoformat(first_session(month)).date(), time(9), ET)


def assumed_available_at(vintage_month: str, lag_months: int = 2) -> datetime:
    return datetime.fromisoformat(add_months(vintage_month, lag_months) + '-01').replace(tzinfo=ET)


def select_vintage(month: str, available_vintages, *, lag_months: int = 2, profile: str = 'vintage_lagged', fixed_vintage: str = '2023-02') -> str:
    if profile not in ('fixed_snapshot', 'vintage_lagged'):
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'unknown profile: {profile}')
    available = sorted({month_string(x) for x in available_vintages})
    eligible = [x for x in available if x <= add_months(month, -lag_months)] if profile == 'vintage_lagged' else [fixed_vintage] if fixed_vintage in available else []
    if not eligible:
        raise ResearchError(ErrorCode.MISSING_DATA, f'no eligible macro vintage for {month}')
    return eligible[-1]


def decision_ledger(month: str, available_vintages, *, profile: str = 'vintage_lagged', lag_months: int = 2, train_months: int = 48, fixed_vintage: str = '2023-02') -> dict:
    """Explicit predictor, condition, target, execution and availability months.

    Target s ranges m-(train_months+1)..m-2. U_s has base s-(b+1)
    and the condition U_(s+1) has base s-b. This is a declared research
    assumption because exact release/decision times are unspecified in the paper.
    """
    if lag_months not in (1, 2, 3) or train_months < 2:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid timing lag/window')
    decision = decision_at(month)
    vintage = select_vintage(month, available_vintages, lag_months=lag_months, profile=profile, fixed_vintage=fixed_vintage)
    available = assumed_available_at(vintage, lag_months)
    rows = []
    for target in month_range(add_months(month, -train_months-1), add_months(month, -2)):
        rows.append({
            'target_month': target,
            'predictor_month': target,
            'condition_month': add_months(target, 1),
            'predictor_base_month': add_months(target, -lag_months-1),
            'condition_base_month': add_months(target, -lag_months),
            'target_start_session': first_session(target),
            'target_end_session': first_session(add_months(target, 1)),
            'target_start_at': session_close(first_session(target)).isoformat(),
            'target_end_at': session_close(first_session(add_months(target, 1))).isoformat(),
        })
    prices_known = all(datetime.fromisoformat(r['target_end_at']) < decision for r in rows)
    cutoff = add_months(month, -lag_months-1)
    cutoff_valid = all(r['predictor_base_month'] <= cutoff and r['condition_base_month'] <= cutoff for r in rows)
    checks = [
        {'name': 'target_price_end_before_decision', 'status': 'pass' if prices_known else 'fail', 'reason': 'all training target closing prices precede the decision'},
        {'name': 'macro_base_month_at_or_before_cutoff', 'status': 'pass' if cutoff_valid else 'fail', 'reason': 'fit/predictor base months are bounded by the current macro cutoff'},
        {'name': 'macro_vintage_available_before_decision', 'status': 'not_applicable' if profile == 'fixed_snapshot' else 'pass' if available < decision else 'fail', 'reason': 'fixed snapshot is an ex-post comparison; actual assumed availability is retained' if profile == 'fixed_snapshot' else 'vintage month plus b months, first calendar day 00:00 ET is a research assumption'},
    ]
    if any(c['status'] == 'fail' for c in checks):
        raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'decision timing check failed', details={'checks': checks})
    return {
        'decision_month': month, 'decision_session': first_session(month),
        'decision_at': decision.isoformat(), 'execution_at': session_close(first_session(month)).isoformat(),
        'target_return_end_at': session_close(first_session(add_months(month, 1))).isoformat(),
        'profile': profile, 'lag_months': lag_months, 'vintage_month': vintage,
        'assumed_available_at': available.isoformat(),
        'current_base_month': cutoff, 'macro_cutoff_base_month': cutoff,
        'next_state_base_month': add_months(month, -lag_months),
        'training_rows': rows, 'checks': checks,
        'timing_policy_status': 'assumption_documented',
        'timing_policy_reason': '09:00 ET decision, first-session close execution, and b-month availability lag are declared research assumptions',
    }


def monthly_returns(prices: pd.DataFrame, start: str, end: str, *, tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    """Return decimal R_m for every requested month; no missing-month deletion."""
    require_unique(prices, ('session', 'ticker'))
    if not {'adj_close', 'finalized'} <= set(prices.columns):
        raise ResearchError(ErrorCode.MISSING_DATA, 'prices require adj_close and finalized')
    months = month_range(start, end)
    if not months:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'empty return interval')
    lookup = prices.set_index(['session', 'ticker'])
    values = []
    for month in months:
        pairs = []
        for m in (month, add_months(month, 1)):
            day = first_session(m)
            row = []
            for ticker in tickers:
                if (day, ticker) not in lookup.index:
                    first = prices.loc[prices.ticker.eq(ticker), 'session'].min()
                    reason = 'partial_listing_month_warmup' if isinstance(first, str) and day < first else 'first_session_missing'
                    raise ResearchError(ErrorCode.CALENDAR_GAP, f'{reason}: {ticker} {day}; cannot substitute another date')
                obs = lookup.loc[(day, ticker)]
                if not bool(obs.finalized):
                    raise ResearchError(ErrorCode.MISSING_DATA, f'unfinalized price: {ticker} {day}')
                if not np.isfinite(obs.adj_close) or obs.adj_close <= 0:
                    raise ResearchError(ErrorCode.NONFINITE_VALUE, f'invalid adjusted price: {ticker} {day}')
                row.append(float(obs.adj_close))
            pairs.append(np.asarray(row))
        values.append(pairs[1]/pairs[0]-1)
    return pd.DataFrame(values, index=pd.Index(months, name='month'), columns=list(tickers))
