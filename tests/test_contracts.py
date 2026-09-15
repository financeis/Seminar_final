import json
import numpy as np
import pandas as pd
import pytest
from regime_alloc.contracts import ResearchError, canonical_id, write_json, checked_join
from regime_alloc.config import load_config


def test_content_id_is_order_independent_and_nonfinite_rejected():
    assert canonical_id({'a': 1, 'b': 2}) == canonical_id({'b': 2, 'a': 1})
    with pytest.raises(ResearchError, match='nonfinite_value'):
        canonical_id({'x': float('nan')})


def test_atomic_immutable_json_and_null_reason(tmp_path):
    p = tmp_path / 'result.json'
    write_json(p, {'value': None, 'reason': 'no observations'})
    assert json.loads(p.read_text())['value'] is None
    with pytest.raises(ResearchError, match='run_conflict'):
        write_json(p, {'value': 1})
    with pytest.raises(ResearchError, match='missing_data'):
        write_json(tmp_path / 'bad.json', {'value': None})


def test_join_rejects_duplicates_and_different_partition():
    a = pd.DataFrame({'month': ['2000-01'], 'partition_id': ['a'], 'x': [1]})
    b = pd.DataFrame({'month': ['2000-01'], 'partition_id': ['b'], 'y': [2]})
    with pytest.raises(ResearchError, match='partition_mismatch'):
        checked_join(a, b, ['month'])
    with pytest.raises(ResearchError, match='duplicate_key'):
        checked_join(pd.concat([a, a]), a, ['month'])


@pytest.mark.parametrize('text', ['surprise=1', '[research]\ntrain_months=0', '[data]\ntickers=["SPY","SPY"]', '[research]\nstart_month="2003-13"', '[research]\nstrategies=["bad"]', '[research]\nseed=true'])
def test_config_strict(tmp_path, text):
    p = tmp_path / 'bad.toml'
    p.write_text(text)
    with pytest.raises(ResearchError, match='invalid_config'):
        load_config(p)


def test_config_defaults_and_relative_paths(tmp_path):
    p = tmp_path / 'settings.toml'
    p.write_text('[data]\nroot="inputs"\n')
    c = load_config(p)
    assert c.data.root == tmp_path / 'inputs'
    assert c.research.train_months == 48
    assert c.research.lag_months == 2
    assert len(c.research.lambdas) == 41
    assert c.research.lambdas[0] == pytest.approx(1e-4)
    assert c.research.lambdas[-1] == pytest.approx(1e4)


def test_npz_round_trip_and_manifest_tampering(tmp_path):
    from regime_alloc.contracts import save_npz, load_npz
    p = tmp_path / 'model.npz'
    expected = np.array([[1.,2.], [3.,4.]])
    manifest = save_npz(p, {'beta': expected})
    assert manifest['arrays']['beta']['shape'] == [2,2]
    np.testing.assert_array_equal(load_npz(p, manifest)['beta'], expected)
    manifest['arrays']['beta']['shape'] = [4]
    with pytest.raises(ResearchError, match='hash_mismatch'):
        load_npz(p, manifest)
    with pytest.raises(ResearchError, match='nonfinite_value'):
        save_npz(tmp_path/'object.npz', {'objects': np.array([{}])})


def test_shared_row_keys():
    from regime_alloc.contracts import WINDOW_KEY, FORECAST_KEY
    assert WINDOW_KEY == ('run_id','decision_month')
    assert FORECAST_KEY == ('run_id','decision_month','model','ticker')


@pytest.mark.parametrize('text', ['[research]\ntrain_months=12', '[research]\nfallback="pooled_mean"', '[research]\nfallback="prior_mean"'])
def test_research_window_and_fallback_policies_strict(tmp_path, text):
    p = tmp_path/'research.toml'
    p.write_text(text)
    with pytest.raises(ResearchError, match='invalid_config'):
        load_config(p)


def test_fixed_snapshot_release_is_not_user_selectable(tmp_path):
    p = tmp_path/'different-vintage.toml'
    p.write_text('[data]\nfixed_vintage="2022-02"\n')
    with pytest.raises(ResearchError, match='invalid_config'):
        load_config(p)


def test_suite_randomness_and_block_settings(tmp_path):
    p = tmp_path/'suite.toml'
    p.write_text('[suite]\nbootstrap_seed=17\nauxiliary_seed=31\nbootstrap_sensitivity_blocks=[6,24]\n')
    settings = load_config(p).suite
    assert (settings.bootstrap_seed, settings.auxiliary_seed) == (17, 31)
    assert settings.bootstrap_sensitivity_blocks == (6, 24)
    assert len(settings.sensitivity_variants) == 16


@pytest.mark.parametrize('setting', ['bootstrap_seed=-1', 'auxiliary_seed=-1',
    'bootstrap_seed=true', 'bootstrap_sensitivity_blocks=[6,6]',
    'bootstrap_sensitivity_blocks=[0,24]', 'bootstrap_sensitivity_blocks=[]',
    'sensitivity_variants=["future"]', 'sensitivity_variants=["lag_1","lag_1"]', 'sensitivity_variants=[]',
    'sensitivity_lags=[1,2,3]'])
def test_suite_randomness_and_block_settings_reject_invalid(tmp_path, setting):
    p = tmp_path/'suite.toml'
    p.write_text('[suite]\n'+setting+'\n')
    with pytest.raises(ResearchError, match='invalid_config'):
        load_config(p)
