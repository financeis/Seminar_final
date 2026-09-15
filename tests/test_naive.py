import numpy as np
import pytest
from regime_alloc.contracts import ResearchError
from regime_alloc.models import conditional_sharpe


def test_conditional_sharpe_sample_standard_deviation():
    result = conditional_sharpe([[.01, -.02], [.03, -.04]])
    np.testing.assert_allclose(result.values, [2 ** .5, -3 / 2 ** .5])
    assert result.status == ('ok', 'ok')
    np.testing.assert_allclose(result.mean, [.02, -.03])
    assert result.raw_n == 2


@pytest.mark.parametrize('returns', [np.empty((0, 2)), [[.1, .2]], [[.1, .2], [.1, .2]], [[.1, .2], [.1, .2], [.1, .2]]])
def test_undefined_is_tagged_zero_not_measured_sharpe(returns):
    result = conditional_sharpe(returns)
    np.testing.assert_array_equal(result.values, np.zeros(2))
    assert result.status == ('undefined', 'undefined')
    assert all(r == 'undefined_conditional_sharpe' for r in result.reasons)


def test_nonfinite_conditional_input_rejected():
    with pytest.raises(ResearchError): conditional_sharpe([[np.nan], [1.]])


def test_tiny_real_variation_kept_and_overflow_rejected():
    result = conditional_sharpe([[.1], [.1 + 1e-14], [.1 - 1e-14]])
    assert result.status == ('ok',)
    assert np.isfinite(result.values[0]) and result.values[0] > 1e10
    with pytest.raises(ResearchError): conditional_sharpe([[1e308], [-1e308]])
