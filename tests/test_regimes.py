"""Algorithm 1, identity and exact state replay checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from regime_alloc.config import load_config, ResearchSettings
from regime_alloc.contracts import ResearchError
from regime_alloc.data import available_vintages, load_macro_vintage
from regime_alloc.data.calendar import decision_ledger
from regime_alloc.features import fit_preprocessor
from regime_alloc.features.windows import FeatureWindow, build_feature_window
from regime_alloc.regimes import (spherical_kmeans, fit_partition, partition_from_labels,
    build_window_state, build_inner_window_state, WindowState, checked_state_join)


def points():
    rng = np.random.default_rng(4)
    return np.vstack([rng.normal(0,1,(45,4)), rng.normal(18,.1,(3,4))])


def feature_window(train_months=48, scope='rolling'):
    ledger = decision_ledger('2003-02',['2002-12'],train_months=train_months)
    rows = pd.DataFrame(ledger['training_rows'])
    months = pd.period_range(rows.predictor_base_month.iloc[0],'2002-11',freq='M').astype(str)
    values = pd.DataFrame(np.vstack([np.zeros((1,4)), points()[-train_months:], np.ones((1,4))]), index=months, columns=list('abcd'))
    prep = fit_preprocessor(values, rows.condition_base_month, dict.fromkeys(values,1), pca_variance=1.,scope=scope)
    return FeatureWindow(prep, prep.fit_result, prep.transform(values, rows.predictor_base_month),
                         prep.transform(values,[ledger['current_base_month']]),rows,ledger)


def test_spherical_centers_unit_and_reproducible():
    x = np.array([[1.,0],[2,0],[0,1],[0,2],[-1,0],[-2,0]])
    a = spherical_kmeans(x,3,seed=7)
    b = spherical_kmeans(x,3,seed=7)
    np.testing.assert_array_equal(a.labels,b.labels)
    np.testing.assert_allclose(a.centers, [[-1,0],[0,1],[1,0]])
    np.testing.assert_allclose(np.linalg.norm(a.centers,axis=1), 1)
    assert a.inertia == pytest.approx(0)


def test_spherical_elbow_supports_k_one_through_ten():
    angle = np.arange(20)*2*np.pi/20
    x = np.column_stack([np.cos(angle),np.sin(angle),np.ones(20)])
    for k in range(1,11):
        result = spherical_kmeans(x,k,n_init=2)
        assert len(np.unique(result.labels)) == k
        assert result.centers.shape == (k,3)


def test_zero_direction_and_empty_reseed_are_explicit():
    x = np.array([[0.,0],[0,0],[1,0],[1,0],[0,1],[0,1]])
    result = spherical_kmeans(x,3,initial_centers=np.zeros((3,2)),n_init=1)
    assert len(np.unique(result.labels)) == 3
    assert result.diagnostics['zero_vector_count'] == 2
    assert result.diagnostics['empty_reseeds'] > 0


@pytest.mark.parametrize('x,k', [(np.ones((8,3)),2),(np.zeros((8,3)),2),(np.array([[1.,0],[2,0],[3,0]]),2)])
def test_too_few_directions_fail(x,k):
    with pytest.raises(ResearchError,match='degenerate_partition'): spherical_kmeans(x,k)


def test_two_stage_crisis_is_smaller_and_ties_lexical():
    with threadpool_limits(1):
        result = fit_partition(points())
        assert (result.labels == 0).sum() == 3
        assert set(result.labels) == set(range(6))
        tie = fit_partition([[-10.,0],[-9,1],[-9,-1],[9,-1],[9,1],[10,0]],k_normal=2)
    assert np.all(tie.labels[:3] == 0)
    assert tie.diagnostics['stage1_tie_policy'] == 'lexicographically_smallest_center'


def test_labels_api_recalculates_centers_without_relabeling():
    x = points()
    labels = np.tile(np.arange(6),8)
    result = partition_from_labels(x,labels,k_normal=5)
    np.testing.assert_array_equal(result.labels,labels)
    np.testing.assert_allclose(result.stage1_centers[0], x[labels==0].mean(axis=0))
    with pytest.raises(ResearchError): partition_from_labels(x,np.zeros(48,dtype=int),k_normal=5)


def test_window_identity_replay_and_shared_alignment(tmp_path):
    with threadpool_limits(1): state = build_window_state(feature_window())
    assert state.regime_ids == tuple(f'R{i}' for i in range(6))
    assert state.labels.index.tolist() == feature_window().training_rows.target_month.tolist()
    assert state.condition_scores.index[0] == '1998-11'
    assert state.predictor_scores.index[0] == '1998-10'
    assert state.current_probability.sum() == pytest.approx(1)
    assert state.next_probability.sum() == pytest.approx(1)
    assert (state.transition_matrix.sum(axis=1) == pytest.approx(np.ones(6)))
    state.save(tmp_path/'state')
    restored = WindowState.load(tmp_path/'state')
    assert restored.partition_id == state.partition_id
    np.testing.assert_array_equal(restored.current_probability,state.current_probability)
    pd.testing.assert_frame_equal(restored.predictor_scores,state.predictor_scores)
    pd.testing.assert_frame_equal(restored.centers_in_feature_units(),state.centers_in_feature_units())
    assert restored.probability_records('x')[0].keys() == dict.fromkeys(['run_id','decision_month','partition_id','regime_id','current_probability','next_probability']).keys()
    state.require_identity(state.partition_id,decision_month='2003-02')
    with pytest.raises(ResearchError): state.require_identity('wrong')
    with pytest.raises(ResearchError): state.require_identity(state.partition_id,decision_month='2003-03')
    with pytest.raises(ResearchError): state.require_scope('full_sample')
    with pytest.raises(ResearchError): state.save(tmp_path/'state')


def test_identity_changes_with_input_query_and_policy():
    with threadpool_limits(1):
        a = build_window_state(feature_window())
        b = build_window_state(feature_window(),replace(ResearchSettings(),seed=1))
    assert a.partition_id != b.partition_id
    features = feature_window()
    features.current.scores.iloc[0,0] += 1
    with pytest.raises(ResearchError): build_window_state(features)


def test_state_public_views_cannot_mutate_computation_and_private_tamper_detected(tmp_path):
    with threadpool_limits(1): state = build_window_state(feature_window())
    old = state.current_probability.copy()
    metadata = state.metadata
    metadata['scope'] = 'full_sample'
    value = state.current_probability
    try: value[0] = 999
    except ValueError: pass
    np.testing.assert_array_equal(state.current_probability,old)
    state._metadata['scope'] = 'full_sample'
    with pytest.raises(ResearchError,match='hash_mismatch'): state.require_scope('full_sample')
    with pytest.raises(ResearchError): state.save(tmp_path/'bad')


def test_corrupt_saved_metadata_rejected(tmp_path):
    with threadpool_limits(1): state = build_window_state(feature_window())
    state.save(tmp_path)
    path=tmp_path/'window_state.json'
    meta=json.loads(path.read_text(encoding='utf-8'))
    meta['metadata']['regime_ids'].reverse()
    path.write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ResearchError): WindowState.load(tmp_path)


def test_inner_fold_stays_rolling_but_cannot_alias_outer():
    features=feature_window(23)
    with threadpool_limits(1): inner=build_inner_window_state(features)
    assert len(inner.labels)==23
    assert inner.scope=='rolling'
    assert inner.metadata['window_kind']=='inner_fold'
    with pytest.raises(ResearchError,match='48'): build_window_state(features)
    full=feature_window(scope='full_sample')
    with pytest.raises(ResearchError,match='partition_mismatch'): build_window_state(full)
    with pytest.raises(ResearchError,match='partition_mismatch'): build_inner_window_state(full)


def test_actual_array_private_tamper_and_supplied_memberships(tmp_path):
    with threadpool_limits(1): state=build_window_state(feature_window())
    original=state.labels.copy()
    control=state.with_labels(np.roll(original.to_numpy(),7),provenance={'seed':7,'method':'test_permutation'})
    assert control.partition_id!=state.partition_id
    pd.testing.assert_series_equal(state.labels,original)
    np.testing.assert_array_equal(np.bincount(control.labels),np.bincount(original))
    np.testing.assert_allclose(control.transition_matrix.sum(axis=1),np.ones(6))
    control.save(tmp_path/'control')
    assert WindowState.load(tmp_path/'control').partition_id==control.partition_id
    state._arrays['current_probability'].setflags(write=True)
    state._arrays['current_probability'][0] += .1
    with pytest.raises(ResearchError,match='hash_mismatch'): state.probability_records('a')


def test_state_rejects_duplicate_roles_and_explicit_policy_disagreement():
    features=feature_window()
    features.training_rows.iloc[1,features.training_rows.columns.get_loc('target_month')]='1999-01'
    with pytest.raises(ResearchError): build_window_state(features)
    with pytest.raises(ResearchError):
        build_window_state(feature_window(),replace(ResearchSettings(),lag_months=3))


@pytest.mark.parametrize('values', [[[np.nan,1],[0,1]], [[np.inf,1],[0,1]]])
def test_nonfinite_clustering_input_rejected(values):
    with pytest.raises(ResearchError): spherical_kmeans(values,2)
    with pytest.raises(ResearchError): fit_partition(values,k_normal=2)


def test_join_checks_partition_dates_and_duplicates():
    a=pd.DataFrame({'decision_month':['2003-02'],'partition_id':['a'],'x':[1]})
    b=pd.DataFrame({'decision_month':['2003-02'],'partition_id':['b'],'y':[2]})
    with pytest.raises(ResearchError,match='partition_mismatch'): checked_state_join(a,b,['decision_month'])
    b['partition_id']='a'
    assert checked_state_join(a,b,['decision_month']).y.tolist()==[2]
    with pytest.raises(ResearchError,match='duplicate_key'): checked_state_join(pd.concat([a,a]),b,['decision_month'])
    b['decision_month']='2003-03'
    with pytest.raises(ResearchError): checked_state_join(a,b,['decision_month'])


@pytest.mark.real_data
def test_real_first_fred_state(tmp_path):
    data=os.environ.get('REGIME_DATA_ROOT')
    if not data: pytest.skip('immutable FRED snapshot not configured')
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs'/'data.toml')
    ledger=decision_ledger('2003-02',available_vintages(data,cfg.data.fred_dataset_id))
    vintage=load_macro_vintage(data,ledger['vintage_month'],dataset_id=cfg.data.fred_dataset_id)
    with threadpool_limits(1): state=build_window_state(build_feature_window(vintage,ledger,cfg.research),cfg.research)
    assert len(state.labels)==48
    assert set(state.labels)==set(range(6))
    assert state.next_probability.sum()==pytest.approx(1)
    state.save(tmp_path/'real')
    assert WindowState.load(tmp_path/'real').partition_id==state.partition_id
