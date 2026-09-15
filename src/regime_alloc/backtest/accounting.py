"""Self-financing monthly ledger with an explicit month-end cash-cost approximation.

Zero default cash interest, borrowing, stock-loan and trading costs are research
assumptions, not claims about executable trading conditions. Costs are charged
to cash at the holding interval's end, not solved as an execution-time order
model. The next pretrade weights include all asset drift and cash costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PortfolioSettings, month_string
from ..contracts import TICKERS, ErrorCode, ResearchError
from ..portfolio._validation import finite_scalar, ticker_order, timestamp, vector

ACCOUNTING_POLICY = {
    'cost_timing': 'deduct transaction, stock-loan, and financing costs from cash at holding end',
    'day_count': 'ACT/365 using New York local execution calendar dates',
    'cash_interest': 'positive target cash times annual cash rate times holding days / 365',
    'cash_ratio': 'fraction of holding months with exactly zero total absolute ETF exposure',
    'zero_cost_assumption': 'default zero rates and costs are research assumptions, not actual trading terms',
}


def holding_days(execution_at, target_end) -> int:
    """New York date difference, immune to a 23-hour daylight-saving day."""
    start = timestamp(execution_at, 'execution_at').tz_convert('America/New_York')
    end = timestamp(target_end, 'target_end').tz_convert('America/New_York')
    days = (end.date()-start.date()).days
    if end <= start or days <= 0:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'holding interval must have positive calendar days')
    return days


@dataclass(frozen=True)
class MonthlyAccount:
    holding_month: str
    decision_at: str
    execution_at: str
    target_end: str
    holding_days: int
    asset_returns: pd.Series
    base_weights: pd.Series
    target_weights: pd.Series
    pretrade_weights: pd.Series
    end_weights: pd.Series
    gross_return: float
    transaction_cost: float
    borrow_cost: float
    financing_cost: float
    cash_return: float
    net_return: float
    turnover: float
    gross_exposure: float
    net_exposure: float
    wealth: float
    drawdown: float
    peak_wealth: float

    def to_record(self, run_id: str, strategy_id: str) -> dict:
        """One returns.csv row; returns/costs are decimal fractions of starting NAV."""
        fields = ('holding_month', 'decision_at', 'execution_at', 'target_end',
            'gross_return', 'transaction_cost', 'borrow_cost', 'financing_cost',
            'cash_return', 'net_return', 'turnover', 'gross_exposure', 'net_exposure',
            'wealth', 'drawdown')
        return {'run_id': run_id, 'strategy_id': strategy_id, **{key: getattr(self, key) for key in fields}}

    def weight_records(self, run_id: str, strategy_id: str) -> list[dict]:
        """One weights.csv row per ETF and CASH; unique within month/strategy."""
        return [{'run_id': run_id, 'decision_month': self.holding_month, 'strategy_id': strategy_id,
            'ticker': ticker, 'base_weight': float(self.base_weights[ticker]),
            'target_weight': float(self.target_weights[ticker]),
            'pretrade_weight': float(self.pretrade_weights[ticker])} for ticker in self.target_weights.index]


def _with_cash(weights: pd.Series, label: str) -> pd.Series:
    result = weights.copy()
    result.loc['CASH'] = 1 - float(weights.sum())
    result.name = label
    return result


def _sum_one(weights: pd.Series, label: str) -> None:
    if not np.isfinite(weights.to_numpy()).all():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'{label} contains nonfinite holdings')
    # Floating summation can lose absolute precision at very high drifted leverage.
    # Do not silently renormalize: a failed conservation check is an explicit error.
    if not np.isclose(float(weights.sum()), 1, rtol=0, atol=1e-10):
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, f'{label} does not sum to one')


def account_month(target_weights, asset_returns, *, holding_month: str, decision_at,
                  execution_at, target_end, previous: MonthlyAccount | None = None,
                  settings: PortfolioSettings | None = None, base_weights=None,
                  tickers: tuple[str, ...] = TICKERS) -> MonthlyAccount:
    """Account one complete monthly holding interval, starting at initial NAV=1.

    All ETF returns must be finite, including inactive assets and benchmarks.
    Explicit alternate tickers support independent small-asset test fixtures;
    production callers use the default fixed ten-asset panel.
    `previous` is the immediately preceding result, including its drifted CASH.
    Supply `base_weights` whenever targets were scaled; a leveraged target
    cannot serve as its own unscaled base. If omitted, unleveraged targets
    are treated as their own base (an ordinary unscaled ledger).
    Insolvency raises ResearchError with bankruptcy month/NAV; no log is made.
    """
    names = ticker_order(tickers)
    month = month_string(holding_month)
    decision = timestamp(decision_at, 'decision_at')
    execution = timestamp(execution_at, 'execution_at')
    end = timestamp(target_end, 'target_end')
    for label, instant in [('decision_at', decision), ('execution_at', execution)]:
        if instant.tz_convert('America/New_York').strftime('%Y-%m') != month:
            raise ResearchError(ErrorCode.CALENDAR_GAP, f'{label} does not match holding month')
    if end.tz_convert('America/New_York').strftime('%Y-%m') != str(pd.Period(month, 'M')+1):
        raise ResearchError(ErrorCode.CALENDAR_GAP, 'holding end must be in the next month')
    if not decision < execution < end:
        raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'require decision < execution < holding end')
    days = holding_days(execution, end)
    config = settings if settings is not None else PortfolioSettings()
    cap = finite_scalar(config.gross_cap, 'gross_cap', positive=True)
    trading_rate = finite_scalar(config.transaction_cost_bps, 'transaction_cost_bps', nonnegative=True) / 10000
    loan_rate = finite_scalar(config.borrow_rate, 'borrow_rate', nonnegative=True)
    finance_rate = finite_scalar(config.financing_rate, 'financing_rate', nonnegative=True)
    cash_rate = finite_scalar(config.cash_rate, 'cash_rate')
    w = vector(target_weights, names, 'target weights')
    base = vector(target_weights if base_weights is None else base_weights, names, 'base weights')
    r = vector(asset_returns, names, 'asset returns')
    gross_exposure = float(w.abs().sum())
    if gross_exposure > cap+1e-12:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'target gross exposure exceeds configured cap')
    if base_weights is None and gross_exposure > 1+1e-12:
        raise ResearchError(ErrorCode.MISSING_DATA, 'leveraged targets require explicit unscaled base weights')
    if base_weights is not None and base.abs().sum() > 1+1e-12:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'unscaled base gross exposure exceeds one')
    if (r < -1).any():
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'asset returns cannot be below -1')
    target = _with_cash(w, 'target_weight')
    base = _with_cash(base, 'base_weight')
    if previous is None:
        pretrade = pd.Series([0.]*len(names)+[1.], index=[*names, 'CASH'], name='pretrade_weight')
        prior_wealth, prior_peak = 1., 1.
    else:
        if pd.Period(previous.holding_month, 'M')+1 != pd.Period(month, 'M'):
            raise ResearchError(ErrorCode.CALENDAR_GAP, 'previous holding month must immediately precede this month')
        if timestamp(previous.target_end, 'previous target_end') != execution:
            raise ResearchError(ErrorCode.CALENDAR_GAP, 'previous interval end and current execution do not match')
        pretrade = vector(previous.end_weights, (*names, 'CASH'), 'pretrade_weight')
        prior_wealth = finite_scalar(previous.wealth, 'previous wealth', positive=True)
        prior_peak = finite_scalar(previous.peak_wealth, 'previous peak wealth', positive=True)
        if prior_peak < max(1., prior_wealth):
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'previous peak omits initial or current NAV')
    _sum_one(pretrade, 'pretrade weights')
    _sum_one(target, 'target weights')
    turnover = float((w-pretrade.loc[list(names)]).abs().sum())
    transaction_cost = float(turnover * trading_rate)
    borrow_cost = float(-w.clip(upper=0).sum() * loan_rate * days/365)
    financing_cost = float(max(-target['CASH'], 0.) * finance_rate * days/365)
    cash_return = float(max(target['CASH'], 0.) * cash_rate * days/365)
    gross_return = float(w.to_numpy() @ r.to_numpy())
    net_return = float(gross_return + cash_return - transaction_cost - borrow_cost - financing_cost)
    growth = 1 + net_return
    wealth = float(prior_wealth * growth)
    numerics = [turnover, transaction_cost, borrow_cost, financing_cost, cash_return, gross_return, net_return, wealth]
    if not np.isfinite(numerics).all():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'monthly accounting overflow or nonfinite result', details={'holding_month': month})
    if wealth <= 0:
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'portfolio bankruptcy: net asset value is nonpositive',
            details={'bankruptcy_month': month, 'wealth': wealth, 'net_return': net_return, 'previous_wealth': prior_wealth})
    end_holdings = w * (1+r)
    end_holdings.loc['CASH'] = float(target['CASH'] + cash_return - transaction_cost - borrow_cost - financing_cost)
    end_weights = end_holdings / growth
    end_weights.name = 'end_weight'
    _sum_one(end_weights, 'end weights')
    peak = float(max(prior_peak, wealth))
    return MonthlyAccount(month, decision.isoformat(), execution.isoformat(), end.isoformat(), days,
        r, base, target, pretrade, end_weights, gross_return, transaction_cost, borrow_cost,
        financing_cost, cash_return, net_return, turnover, gross_exposure, float(w.sum()),
        wealth, float(wealth/peak-1), peak)
