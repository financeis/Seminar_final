"""Independent, hand-computed monthly transformation examples."""
import numpy as np
import pandas as pd
import pytest

from regime_alloc.contracts import ResearchError
from regime_alloc.features import transform_tcodes


@pytest.mark.parametrize('code, expected', [
    (1, [1, 2, 8, 32]), (2, [np.nan, 1, 6, 24]),
    (3, [np.nan, np.nan, 5, 18]),
    (4, [0, np.log(2), 3*np.log(2), 5*np.log(2)]),
    (5, [np.nan, np.log(2), 2*np.log(2), 2*np.log(2)]),
    (6, [np.nan, np.nan, np.log(2), 0]),
    (7, [np.nan, np.nan, 2, 0]),
])
def test_hand_computed_tcodes(code, expected):
    raw = pd.DataFrame({'a': [1., 2., 8., 32.]}, index=['2020-01','2020-02','2020-03','2020-04'])
    out = transform_tcodes(raw, {'a': code})
    np.testing.assert_allclose(out.a, expected, atol=1e-14, equal_nan=True)


def test_calendar_gap_is_inserted_before_differencing():
    raw = pd.DataFrame({'a': [1., 4., 8.]}, index=['2020-01','2020-03','2020-04'])
    out = transform_tcodes(raw, {'a': 2})
    assert list(out.index) == ['2020-01','2020-02','2020-03','2020-04']
    np.testing.assert_allclose(out.a, [np.nan,np.nan,np.nan,4], equal_nan=True)


def test_invalid_log_and_nonfinite_values_remain_missing():
    raw = pd.DataFrame({'a': [0., -2., np.inf, 2., 4.]}, index=pd.period_range('2020-01', periods=5, freq='M').astype(str))
    assert transform_tcodes(raw, {'a': 4}).a.isna().tolist() == [True,True,True,False,False]
    assert transform_tcodes(raw, {'a': 5}).a.isna().tolist() == [True,True,True,True,False]
    assert transform_tcodes(raw, {'a': 1}).a.isna().tolist() == [False,False,True,False,False]


def test_tcode_seven_never_fills_or_divides_by_zero():
    raw = pd.DataFrame({'a': [1., 0., 2., 4., np.nan, 8.,16.]}, index=pd.period_range('2020-01', periods=7, freq='M').astype(str))
    assert transform_tcodes(raw, {'a': 7}).a.isna().all()


@pytest.mark.parametrize('problem', ['duplicate','code','metadata','month'])
def test_invalid_transform_inputs_fail_explicitly(problem):
    raw = pd.DataFrame({'a': [1., 2.]}, index=['2020-01','2020-02'])
    codes = {'a': 1}
    if problem == 'duplicate': raw.index = ['2020-01','2020-01']
    if problem == 'code': codes = {'a': 8}
    if problem == 'metadata': codes = {}
    if problem == 'month': raw.index = ['2020-1','2020-02']
    with pytest.raises(ResearchError): transform_tcodes(raw, codes)
