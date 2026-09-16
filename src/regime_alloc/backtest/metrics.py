"""Declared monthly performance conventions with explicit undefined-value reasons.

MaxDD includes starting NAV=1. AvgDD averages negative holding-month drawdowns;
avgdd_all_months includes every holding month (neither includes an initial row).
total_cost is the sum of monthly transaction, stock-loan and financing costs,
each expressed as a fraction of that month's initial NAV, not a dollar total.
cash_ratio counts only months with zero total absolute ETF exposure; positive
cash from short proceeds does not make an invested month a cash month.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ..contracts import ErrorCode, ResearchError
from ..portfolio._validation import months as validated_months, vector


@dataclass(frozen=True)
class MetricResult:
    values: dict[str, float | int | str | None]
    reasons: dict[str, str]
    wealth: pd.Series
    drawdown: pd.Series

    def to_record(self, run_id: str, strategy_id: str) -> dict:
        """One metrics.csv row; serialize reasons separately in metrics_reasons.json."""
        return {'run_id': run_id, 'strategy_id': strategy_id, **self.values}

    def to_json_record(self, run_id: str, strategy_id: str) -> dict:
        """Embed sibling null reasons for contracts.write_json's validation."""
        return {**self.to_record(run_id, strategy_id), 'null_reasons': dict(self.reasons)}


def _nonnegative(values, names, label):
    result = vector(values, names, label).to_numpy()
    if (result < 0).any():
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} cannot be negative')
    return result


def compute_metrics(months, net_returns, *, turnover=None, transaction_cost=None,
                    borrow_cost=None, financing_cost=None, gross_exposure=None,
                    risk_free_monthly=0.) -> MetricResult:
    """Compute statistics for one consecutive monthly series in decimal units.

    Inputs may be aligned Series or positional vectors. Supplied optional
    columns must be complete and finite. Missing optional data produces null
    and a reason rather than pretending that unobserved costs/turnover are zero.
    Sharpe uses sample standard deviation; Sortino uses downside RMS over ALL
    excess-return months. Both require at least two months. A zero denominator
    is undefined, while zero annual volatility itself is a valid statistic.
    """
    labels = validated_months(months)
    n = len(labels)
    r = vector(net_returns, labels, 'net returns').to_numpy()
    if np.isscalar(risk_free_monthly):
        rf = vector([risk_free_monthly]*n, labels, 'risk-free returns').to_numpy()
        # Still validate the scalar when n=0.
        if n == 0:
            vector([risk_free_monthly], ['risk_free'], 'risk-free returns')
    else:
        rf = vector(risk_free_monthly, labels, 'risk-free returns').to_numpy()
    if (rf < -1).any():
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'risk-free return cannot be below -1')
    if (r <= -1).any():
        i = int(np.flatnonzero(r <= -1)[0])
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'portfolio bankruptcy: monthly growth is nonpositive',
            details={'bankruptcy_month': labels[i], 'net_return': float(r[i])})
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        wealth = np.cumprod(1+r)
    if not np.isfinite(wealth).all() or (wealth <= 0).any():
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE, 'nonfinite or nonpositive cumulative wealth')
    peaks = np.maximum.accumulate(np.concatenate(([1.], wealth)))[1:]
    drawdown = wealth/peaks-1
    values = {'n_months': n, 'start_month': labels[0] if n else None, 'end_month': labels[-1] if n else None}
    reasons = {}

    def undefined(key, reason):
        values[key] = None
        reasons[key] = reason

    def finite_metric(key, value):
        if math.isfinite(float(value)):
            values[key] = float(value)
        else:
            undefined(key, 'numerical_overflow')

    if n:
        exponent = math.log(float(wealth[-1]))*12/n
        try:
            finite_metric('cagr', math.expm1(exponent))
        except OverflowError:
            undefined('cagr', 'numerical_overflow')
        values['maxdd'] = float(drawdown.min())
        negative = drawdown[drawdown < 0]
        values['avgdd'] = float(negative.mean()) if len(negative) else 0.
        values['avgdd_all_months'] = float(drawdown.mean())
        values['positive_ratio'] = float(np.count_nonzero(r > 0)/n)
    else:
        for key in ('start_month', 'end_month', 'cagr', 'maxdd', 'avgdd', 'avgdd_all_months', 'positive_ratio'):
            undefined(key, 'no_observations')
    if n < 2:
        for key in ('ann_vol', 'sharpe', 'sortino'):
            undefined(key, 'fewer_than_two_months')
    else:
        excess = r-rf
        with np.errstate(over='ignore', invalid='ignore'):
            mean = float(np.mean(excess))
            # Exact constant observations have exactly zero dispersion.
            std_return = 0. if np.ptp(r) == 0 else float(np.std(r, ddof=1))
            std_excess = 0. if np.ptp(excess) == 0 else float(np.std(excess, ddof=1))
            downside = float(np.sqrt(np.mean(np.minimum(excess, 0)**2)))
        finite_metric('ann_vol', std_return*math.sqrt(12))
        if std_excess == 0:
            undefined('sharpe', 'zero_excess_return_standard_deviation')
        elif not math.isfinite(std_excess) or not math.isfinite(mean):
            undefined('sharpe', 'numerical_overflow')
        else:
            finite_metric('sharpe', mean/std_excess*math.sqrt(12))
        if downside == 0:
            undefined('sortino', 'zero_downside_rms')
        elif not math.isfinite(downside) or not math.isfinite(mean):
            undefined('sortino', 'numerical_overflow')
        else:
            finite_metric('sortino', mean/downside*math.sqrt(12))
    if turnover is None:
        undefined('mean_turnover', 'turnover_not_supplied')
    else:
        t = _nonnegative(turnover, labels, 'turnover')
        finite_metric('mean_turnover', float(t.mean())) if n else undefined('mean_turnover', 'no_observations')
    costs = [transaction_cost, borrow_cost, financing_cost]
    validated_costs = [_nonnegative(cost, labels, name) for cost, name in
        zip(costs, ('transaction_cost', 'borrow_cost', 'financing_cost')) if cost is not None]
    if len(validated_costs) != 3:
        undefined('total_cost', 'transaction_stock_loan_or_financing_cost_not_supplied')
    elif not n:
        undefined('total_cost', 'no_observations')
    else:
        finite_metric('total_cost', float(np.asarray(validated_costs).sum()))
    if gross_exposure is None:
        undefined('cash_ratio', 'gross_exposure_not_supplied')
    else:
        exposure = _nonnegative(gross_exposure, labels, 'gross exposure')
        if n:
            values['cash_ratio'] = float(np.count_nonzero(exposure == 0)/n)
        else:
            undefined('cash_ratio', 'no_observations')
    return MetricResult(values, reasons, pd.Series(wealth, index=list(labels), name='wealth'),
                        pd.Series(drawdown, index=list(labels), name='drawdown'))
