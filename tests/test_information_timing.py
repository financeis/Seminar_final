"""Feature construction must preserve the calendar's distinct row roles."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from regime_alloc.config import load_config, ResearchSettings
from regime_alloc.contracts import ResearchError
from regime_alloc.data import MacroVintage, load_macro_vintage, available_vintages
from regime_alloc.data.calendar import decision_ledger
from regime_alloc.features import build_feature_window


def synthetic_vintage(ledger):
    months = pd.period_range('1998-06','2003-02',freq='M').astype(str)
    values = pd.DataFrame({'a':np.arange(len(months), dtype=float)+1, 'b':np.sin(np.arange(len(months)))}, index=months)
    return MacroVintage(ledger['vintage_month'], values, pd.Series({'a':1,'b':1}), pd.Series({'a':1,'b':2}), {}, ledger['assumed_available_at'], 'synthetic')


def test_first_window_rows_have_distinct_condition_predictor_current_dates():
    ledger = decision_ledger('2003-02',['2002-12'])
    vintage = synthetic_vintage(ledger)
    window = build_feature_window(vintage, ledger)
    assert window.preprocessor.fit_months == tuple(pd.period_range('1998-11','2002-10',freq='M').astype(str))
    assert list(window.predictor.scores.index) == list(pd.period_range('1998-10','2002-09',freq='M').astype(str))
    assert list(window.current.scores.index) == ['2002-11']
    assert window.condition.scores.shape[0] == window.predictor.scores.shape[0] == 48
    changed = replace(vintage, values=vintage.values.copy())
    changed.values.loc['2002-11':] = np.nan
    again = build_feature_window(changed, ledger)
    assert again.preprocessor.transform_hash == window.preprocessor.transform_hash
    assert again.preprocessor.feature_hash == window.preprocessor.feature_hash


def test_window_refuses_wrong_vintage_timing_and_non48_settings():
    ledger = decision_ledger('2003-02',['2002-12'])
    vintage = synthetic_vintage(ledger)
    with pytest.raises(ResearchError): build_feature_window(replace(vintage,vintage_month='2003-01'),ledger)
    bad = dict(ledger, current_base_month='2002-12')
    with pytest.raises(ResearchError): build_feature_window(vintage,bad)
    with pytest.raises(ResearchError): build_feature_window(vintage,ledger,replace(ResearchSettings(), train_months=47))


def test_fixed_profile_retains_actual_later_availability():
    ledger = decision_ledger('2003-02',['2023-02'],profile='fixed_snapshot')
    window = build_feature_window(synthetic_vintage(ledger), ledger, ResearchSettings(profile='fixed_snapshot'))
    assert window.preprocessor.metadata['provenance']['assumed_available_at'] == '2023-04-01T00:00:00-04:00'
    assert window.preprocessor.metadata['provenance']['profile'] == 'fixed_snapshot'


@pytest.mark.real_data
def test_real_first_feature_window():
    root = Path(__file__).resolve().parents[1]
    data_root = next((p/'data' for p in [root,*root.parents] if (p/'data'/'raw'/'fred').is_dir()), None)
    if data_root is None: pytest.skip('immutable acquired FRED data not present')
    config = load_config(root/'configs'/'data.toml')
    data = replace(config.data, root=data_root)
    ledger = decision_ledger('2003-02', available_vintages(data.root,data.fred_dataset_id))
    vintage = load_macro_vintage(data.root,ledger['vintage_month'],dataset_id=data.fred_dataset_id)
    window = build_feature_window(vintage,ledger,config.research)
    assert ledger['vintage_month'] == '2002-12'
    assert (window.preprocessor.fit_months[0],window.preprocessor.fit_months[-1]) == ('1998-11','2002-10')
    assert (window.predictor.scores.index[0],window.predictor.scores.index[-1]) == ('1998-10','2002-09')
    assert list(window.current.scores.index) == ['2002-11']
    assert window.preprocessor.n_components <= 47
    assert all(vintage.groups[c] != 6 for c in window.preprocessor.columns)
    assert np.isfinite(window.current.scores.to_numpy()).all()
