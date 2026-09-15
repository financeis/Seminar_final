import numpy as np
import pandas as pd
import pytest

from regime_alloc.contracts import TICKERS, ResearchError
from regime_alloc.portfolio.sizing import allocate, benchmark_weights, build_base_weights, strategy_ids
from regime_alloc.portfolio.volatility import scale_to_volatility


SMALL = ('A', 'B', 'C')


@pytest.mark.parametrize('method,size,scores,expected', [
    ('lo', 2, [3, 1, -2], [.75, .25, 0]),
    ('lns', 2, [3, 1, -2], [.5, 1/6, -1/3]),
    ('los', 2, [3, 1, -2], [.6, 0, -.4]),
    ('los', 2, [1, -1, 1], [.5, -.5, 0]),
    ('lo', 2, [1, 1, 1], [.5, .5, 0]),
    ('lo', 3, [-3, -1, 0], [0, 0, 0]),
    ('lns', 3, [-3, -1, 0], [-.75, -.25, 0]),
    ('los', 4, [0, 0, 0], [0, 0, 0]),
])
def test_independent_allocation_examples(method, size, scores, expected):
    w = allocate(scores, method, size, tickers=SMALL)
    assert w.index.tolist() == list(SMALL)
    assert w.tolist() == pytest.approx(expected)
    assert np.abs(w).sum() == pytest.approx(0 if not any(expected) else 1)


def test_lns_selects_up_to_l_on_each_side_and_fixed_ticker_ties():
    tickers = ('A', 'B', 'C', 'D', 'E', 'F')
    w = allocate([3, 2, 1, -3, -2, -1], 'lns', 2, tickers=tickers)
    assert w.tolist() == pytest.approx([.3, .2, 0, -.3, -.2, 0])
    scores = pd.Series([1, 1, 1], index=['C', 'B', 'A'])
    assert allocate(scores, 'lo', 2, tickers=SMALL).tolist() == [.5, .5, 0]


def test_mx_probability_ties_resolve_to_r0_then_regime_number():
    p = pd.Series([.5, .5], index=['R1', 'R0'])
    assert allocate([3, 1, -2], 'mx', 2, next_probabilities=p, tickers=SMALL).tolist() == pytest.approx([.6, 0, -.4])
    assert allocate([3, 1, -2], 'mx', 2, next_probabilities=[.2, .8], tickers=SMALL).tolist() == [.75, .25, 0]


def test_full_strategy_grid_and_benchmarks():
    ids = strategy_ids()
    assert len(ids) == len(set(ids)) == 50
    assert set(ids[-2:]) == {'spy', 'ew'}
    assert 'mvo_mx_4' in ids
    assert benchmark_weights('spy').tolist() == [1.] + [0.]*9
    assert benchmark_weights('ew').tolist() == [.1]*10
    out = build_base_weights({m: np.arange(10) for m in ('naive', 'ridge', 'bl', 'mvo')}, next_probabilities=[.5, .5])
    assert tuple(out) == ids
    assert all(tuple(w.index) == TICKERS for w in out.values())


@pytest.mark.parametrize('scores,method,size,p', [
    ([1, np.nan, 2], 'lo', 2, None), ([1, np.inf, 2], 'lo', 2, None),
    ([1, 2], 'lo', 2, None), ([1, 2, 3], 'lo', 1, None),
    ([1, 2, 3], 'lo', 2.0, None),
    ([1, 2, 3], 'other', 2, None), ([1, 2, 3], 'mx', 2, None),
    ([1, 2, 3], 'mx', 2, [.2, .2]), ([1, 2, 3], 'mx', 2, [-.1, 1.1]),
])
def test_invalid_allocations_fail(scores, method, size, p):
    with pytest.raises(ResearchError):
        allocate(scores, method, size, next_probabilities=p, tickers=SMALL)


def history():
    months = pd.period_range('2019-01', '2022-12', freq='M').astype(str)
    frame = pd.DataFrame({'A': [-.01, .01]*24, 'B': [0.]*48, 'C': [0.]*48}, index=months)
    ends = pd.Series([(pd.Period(m, 'M')+1).strftime('%Y-%m')+'-03T16:00:00-05:00' for m in months], index=months)
    return frame, ends


def scale(frame=None, ends=None, **kwargs):
    r, e = history()
    return scale_to_volatility([1, 0, 0], r if frame is None else frame,
        decision_month='2023-02', decision_at='2023-02-01T09:00:00-05:00',
        return_end_at=e if ends is None else ends, tickers=SMALL, **kwargs)


def test_volatility_sample_covariance_cap_and_cash():
    # 36 values +/- .01 have sample variance .0036/35, annualized x12.
    result = scale(vol_target=.10, gross_cap=2)
    assert result.expected_volatility == pytest.approx(np.sqrt(12*.0036/35))
    assert result.multiplier == 2
    assert result.target_weights.tolist() == [2, 0, 0]
    assert result.cash_weight == -1
    assert result.used_months == tuple(pd.period_range('2020-01', '2022-12', freq='M').astype(str))
    assert scale(gross_cap=1).multiplier == 1
    assert scale(lookback=12).expected_volatility == pytest.approx(np.sqrt(12*.0012/11))


def test_volatility_uses_only_tail_of_training_and_handles_zero_variance():
    frame, ends = history()
    reference = scale()
    frame.iloc[:12, 0] = .5
    assert scale(frame).expected_volatility == reference.expected_volatility
    frame.loc[:, :] = .02
    zero = scale(frame)
    assert zero.multiplier == 0 and zero.cash_weight == 1
    result = scale_to_volatility([0, 0, 0], frame, decision_month='2023-02',
        decision_at='2023-02-01T09:00:00-05:00', return_end_at=ends, tickers=SMALL)
    assert result.multiplier == 0 and result.target_weights.tolist() == [0, 0, 0]


def test_volatility_rejects_future_months_availability_gaps_and_nonfinite_panel():
    frame, ends = history()
    bad = ends.copy()
    bad.iloc[-1] = '2023-02-01T16:00:00-05:00'
    with pytest.raises(ResearchError, match='decision|availability|future'):
        scale(frame, bad)
    future = pd.concat([frame.iloc[1:], pd.DataFrame([[1, 1, 1]], index=['2023-01'], columns=SMALL)])
    with pytest.raises(ResearchError):
        scale(future)
    with pytest.raises(ResearchError):
        scale(frame.drop(index='2020-03'))
    badframe = frame.copy()
    badframe.iloc[0, 1] = np.nan
    with pytest.raises(ResearchError):
        scale(badframe)


def test_volatility_uncapped_multiplier_matches_manual_formula():
    result = scale(vol_target=.02)
    expected = .02/np.sqrt(12*.0036/35)
    assert result.multiplier == pytest.approx(expected)
    assert result.target_weights.iloc[0] == pytest.approx(expected)
    assert result.cash_weight == pytest.approx(1-expected)


def test_volatility_duplicate_month_and_missing_timezone_are_rejected():
    frame, ends = history()
    frame.index = [*frame.index[:-1], frame.index[-2]]
    with pytest.raises(ResearchError, match='duplicate'):
        scale(frame)
    frame, ends = history()
    ends.iloc[-1] = '2023-01-03T16:00:00'
    with pytest.raises(ResearchError, match='timezone'):
        scale(frame, ends)
