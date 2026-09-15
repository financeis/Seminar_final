"""Fitted-state isolation, hand statistics and replay contracts."""
import json
import numpy as np
import pandas as pd
import pytest

from regime_alloc.contracts import ResearchError
from regime_alloc.features import FittedPreprocessor, fit_preprocessor


def frame(values):
    return pd.DataFrame(values, index=pd.period_range('2020-01', periods=len(next(iter(values.values()))), freq='M').astype(str))


def test_original_observation_median_and_bounded_causal_fill():
    x = frame({'a':[100.,1.,np.nan,np.nan,np.nan,9.,20.]})
    fit = fit_preprocessor(x, list(x.index[1:6]), {'a':1}, missing_rate=.7)
    assert fit.medians.tolist() == [5.]
    result = fit.transform(x, list(x.index[1:6]))
    np.testing.assert_array_equal(result.imputed.a, [1.,1.,1.,5.,9.])
    np.testing.assert_allclose(fit.mean, [3.4])
    np.testing.assert_allclose(fit.scale, [3.2])  # ddof=0
    assert result.counts == {'observed':2, 'ffill':2, 'median':1}
    np.testing.assert_array_equal(result.observed_mask | result.ffill_mask | result.median_mask, True)
    assert not (result.ffill_mask & result.median_mask).any().any()


def test_prior_context_can_fill_fit_but_is_excluded_from_median():
    x = frame({'a':[100.,np.nan,1.,3.,5.]})
    fit = fit_preprocessor(x, list(x.index[1:]), {'a':1}, missing_rate=.3)
    assert fit.medians.tolist() == [3.]
    assert fit.fit_result.imputed.a.tolist() == [100.,1.,3.,5.]


def test_group_missing_and_constant_exclusions_are_auditable():
    x = frame({'good':[1.,2.,3.,4.,5.], 'group6':[2.,1.,4.,3.,9.], 'empty':[np.nan]*5,
               'constant':[2.,2.,np.nan,2.,2.], 'sparse':[1.,np.nan,np.nan,4.,5.]})
    fit = fit_preprocessor(x, list(x.index), {'good':1,'group6':6,'empty':2,'constant':3,'sparse':4})
    assert fit.columns == ('good',)
    reasons = {r['column']:r['reasons'] for r in fit.selection}
    assert 'excluded_group_6' in reasons['group6']
    assert 'no_finite_observations' in reasons['empty']
    assert 'constant_observed_values' in reasons['constant']
    assert 'missing_rate_exceeded' in reasons['sparse']
    assert next(r for r in fit.selection if r['column']=='sparse')['missing_rate'] == .4


@pytest.mark.parametrize('values, group', [([np.nan]*4,1),([1.]*4,1),([1.,2.,3.,4.],6)])
def test_no_valid_columns_is_an_explicit_failure(values, group):
    x = frame({'a':values})
    with pytest.raises(ResearchError, match='missing_data'):
        fit_preprocessor(x, list(x.index), {'a':group})


def test_current_and_future_rows_cannot_change_fit_state():
    x = frame({'a':[1.,2.,np.nan,4.,5.,6.], 'b':[8.,3.,4.,1.,2.,3.]})
    months = list(x.index[:4])
    first = fit_preprocessor(x, months, {'a':1,'b':2}, missing_rate=.3)
    changed = x.copy(); changed.iloc[4:] = [[np.nan,1e12],[-1e20,np.nan]]
    second = fit_preprocessor(changed, months, {'a':1,'b':2}, missing_rate=.3)
    assert first.feature_hash == second.feature_hash
    assert first.transform_hash == second.transform_hash
    before = first.transform_hash
    first.transform(changed, list(changed.index[4:]))
    assert first.transform_hash == before
    np.testing.assert_array_equal(first.components, second.components)


def test_query_missingness_never_selects_columns_again_and_saved_tail_fills():
    x = frame({'a':[1.,2.,3.,4.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1})
    query = pd.DataFrame({'a':[np.nan,np.nan,np.nan,99.]}, index=['2020-05','2020-06','2020-07','2020-08'])
    out = fit.transform(query)
    assert out.imputed.a.tolist() == [4.,4.,2.5,99.]
    assert out.counts == {'observed':1,'ffill':2,'median':1}


def test_pca_minimum_component_count_and_inverse_in_original_units():
    x = frame({'a':[-1.,-1.,1.,1.], 'b':[-1.,1.,-1.,1.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1,'b':2}, pca_variance=.95)
    assert fit.n_components == 2
    np.testing.assert_allclose(fit.explained_variance_ratio, [.5,.5])
    out = fit.transform(x)
    np.testing.assert_allclose(fit.inverse_transform(out.scores), x, atol=1e-14)
    one = fit_preprocessor(x, list(x.index), {'a':1,'b':2}, pca_variance=.5)
    assert one.n_components == 1


def test_48_row_pca_respects_numerical_rank_and_95_percent_minimum():
    rng = np.random.default_rng(4815)
    x = pd.DataFrame(rng.normal(size=(48,70)), index=pd.period_range('2000-01', periods=48, freq='M').astype(str))
    x.columns = [f'x{i}' for i in range(70)]
    fit = fit_preprocessor(x, list(x.index), {c:1 for c in x})
    assert fit.n_components <= fit.numerical_rank <= 47
    assert sum(fit.explained_variance_ratio[:fit.n_components]) >= .95
    assert sum(fit.explained_variance_ratio[:fit.n_components-1]) < .95


def test_missing_calendar_month_counts_towards_fill_limit():
    x = frame({'a':[1.,2.,3.,4.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1})
    query = pd.DataFrame({'a':[np.nan]}, index=['2020-07'])
    assert fit.transform(query).imputed.iloc[0,0] == 2.5


def test_nonconsecutive_fold_fit_does_not_fill_from_heldout_inside_span():
    x = frame({'a':[1.,999.,np.nan,4.,5.]})
    fit = fit_preprocessor(x, ['2020-01','2020-03','2020-04','2020-05'], {'a':1}, missing_rate=.3)
    assert fit.fit_result.imputed.loc['2020-03','a'] == 1.
    changed = x.copy(); changed.loc['2020-02','a'] = -999.
    other = fit_preprocessor(changed, list(fit.fit_months), {'a':1}, missing_rate=.3)
    assert fit.transform_hash == other.transform_hash


def test_serialization_replays_masks_scores_and_original_units(tmp_path):
    x = frame({'a':[1.,2.,np.nan,4.,5.], 'b':[2.,5.,3.,7.,1.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1,'b':2}, pca_variance=1.)
    fit.save(tmp_path/'state')
    restored = FittedPreprocessor.load(tmp_path/'state')
    assert restored.transform_hash == fit.transform_hash
    assert restored.feature_hash == fit.feature_hash
    assert restored.columns == fit.columns and restored.fit_months == fit.fit_months
    out = restored.transform(x)
    np.testing.assert_array_equal(out.scores, fit.transform(x).scores)
    np.testing.assert_array_equal(restored.fit_result.ffill_mask, fit.fit_result.ffill_mask)
    np.testing.assert_allclose(restored.inverse_transform(out.scores), out.imputed, atol=1e-14)
    meta = json.loads((tmp_path/'state'/'preprocessor.json').read_text(encoding='utf-8'))
    assert all('sha256' in a and 'shape' in a and 'dtype' in a for a in meta['archive']['arrays'].values())
    with np.load(tmp_path/'state'/'model.npz', allow_pickle=False) as arrays:
        assert all(np.isfinite(arrays[k]).all() and arrays[k].dtype.kind in 'biuf' for k in arrays.files)
    with pytest.raises(ResearchError, match='run_conflict'): fit.save(tmp_path/'state')
    meta['metadata']['scope'] = 'full_sample'
    (tmp_path/'state'/'preprocessor.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ResearchError, match='hash_mismatch'): FittedPreprocessor.load(tmp_path/'state')


def test_scope_guard_and_invalid_dimensions():
    x = frame({'a':[1.,2.,3.,4.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1}, scope='full_sample')
    with pytest.raises(ResearchError, match='partition_mismatch'): fit.transform(x, expected_scope='rolling')
    with pytest.raises(ResearchError): fit.inverse_transform(np.zeros((1,2)))
    with pytest.raises(ResearchError): fit_preprocessor(x, ['2020-01'], {'a':1})


def test_preimputation_missing_rate_boundary_and_five_percent_sensitivity():
    x = frame({'a':[1.,2.,np.inf,4.,5.], 'b':[5.,4.,3.,2.,1.]})
    default = fit_preprocessor(x, list(x.index), {'a':1,'b':2})
    stricter = fit_preprocessor(x, list(x.index), {'a':1,'b':2}, missing_rate=.05)
    assert default.columns == ('a','b')
    assert stricter.columns == ('b',)
    assert default.fit_result.counts_by_column['a'] == {'observed':4,'ffill':1,'median':0}
    assert default.metadata['fit_imputation_counts'] == default.fit_result.counts_by_column


def test_disabled_forward_fill_uses_original_median():
    x = frame({'a':[1.,2.,np.nan,4.,9.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1}, ffill_limit=0)
    assert fit.fit_result.imputed.a.tolist() == [1.,2.,3.,4.,9.]
    assert fit.fit_result.counts == {'observed':4,'ffill':0,'median':1}


def test_archive_tampering_is_detected(tmp_path):
    x = frame({'a':[1.,2.,3.,4.]})
    fit = fit_preprocessor(x, list(x.index), {'a':1})
    fit.save(tmp_path)
    archive = tmp_path/'model.npz'
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(ResearchError, match='hash_mismatch'): FittedPreprocessor.load(tmp_path)


def test_transform_replays_frozen_fold_fit_and_preserves_caller_requested_months():
    x = frame({'a':[1.,999.,np.nan,4.,5.]})
    fit = fit_preprocessor(x, ['2020-01','2020-03','2020-04','2020-05'], {'a':1}, missing_rate=.3)
    replay = fit.transform(x, list(fit.fit_months))
    np.testing.assert_array_equal(replay.imputed, fit.fit_result.imputed)
    assert list(replay.scores.index) == list(fit.fit_months)
