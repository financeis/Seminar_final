from dataclasses import asdict
import json
import numpy as np
import pytest
from regime_alloc.config import ResearchConfig
from regime_alloc.backtest.experiments import permute_memberships, sensitivity_variants, run_suite


@pytest.fixture(scope='module')
def actual_config():
    import os
    from pathlib import Path
    from dataclasses import replace
    from regime_alloc.config import load_config
    root=os.environ.get('REGIME_DATA_ROOT')
    if not root: pytest.skip('immutable market snapshot not configured')
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/reproduction.toml')
    return replace(cfg,data=replace(cfg.data,root=Path(root)))


def test_memberships_preserve_counts_not_names():
    labels=np.repeat(np.arange(6),8)
    result=permute_memberships(labels,0)
    np.testing.assert_array_equal(np.bincount(result),np.bincount(labels))
    assert any(len(set(result[labels==k]))>1 for k in range(6))
    np.testing.assert_array_equal(result,np.random.default_rng(0).permutation(labels))


def test_sensitivity_one_at_a_time():
    base=ResearchConfig()
    variants=sensitivity_variants(base)
    assert len(variants)==16
    for item in variants:
        before,after=asdict(base),asdict(item.config)
        changes={f'{section}.{key}' for section in ['research','portfolio'] for key in before[section] if before[section][key]!=after[section][key]}
        assert changes==set(item.changes)
    assert {v.name for v in variants} >= {'lag_1','lag_3','validation_forward','gross_cap_1','vol_lookback_12'}
    assert next(v for v in variants if v.name=='gross_cap_1').curve=='scaled'


def test_plan_published_before_child_failure(tmp_path,monkeypatch):
    import regime_alloc.backtest.experiments as module
    from regime_alloc.contracts import ResearchError,ErrorCode
    class Context:
        def feasible_start(self,c):return c.research.start_month
        def require_config(self,c):pass
        counts={}
    calls=[]
    def child(c,out,**kw):
        plan=json.loads((tmp_path/'suite'/'planned_runs.json').read_text())
        assert len(plan['runs'])==38  # two profiles * (baseline+2seeds+16variants)
        calls.append(str(out))
        out.mkdir(parents=True)
        (out/'partial.txt').write_text('kept')
        raise ResearchError(ErrorCode.NUMERICAL_FAILURE,'injected child failure')
    monkeypatch.setattr(module,'run_research',child)
    with pytest.raises(ResearchError,match='injected'):
        run_suite(ResearchConfig(),tmp_path/'suite',smoke=True,context=Context())
    manifest=json.loads((tmp_path/'suite'/'run_manifest.json').read_text())
    assert manifest['status']=='failed' and len(calls)==1
    assert list((tmp_path/'suite').rglob('partial.txt'))


def test_failed_seed_is_retained_and_never_replaced(tmp_path,monkeypatch):
    import regime_alloc.backtest.experiments as module
    from regime_alloc.contracts import ResearchError,ErrorCode
    class Context:
        counts={}
        def feasible_start(self,c):return c.research.start_month
        def require_config(self,c):pass
        def state(self,c,m):return object()
    calls=[]
    def child(c,out,**kw):
        calls.append(out.name); out.mkdir(parents=True)
        if out.name=='control_000':
            (out/'failure-evidence.txt').write_text('seed0 failed')
            raise ResearchError(ErrorCode.NUMERICAL_FAILURE,'seed0 must remain failed',details={'seed':0})
        return {'status':'succeeded','run_id':'main-run'}
    monkeypatch.setattr(module,'run_research',child)
    monkeypatch.setattr(module,'random_state',lambda *args:object())
    with pytest.raises(ResearchError,match='seed0 must remain failed'):
        run_suite(ResearchConfig(),tmp_path/'suite',smoke=True,context=Context())
    assert calls==['main','control_000']
    evidence=json.loads((tmp_path/'suite'/'suite_failure.json').read_text())
    assert evidence['active_child']=='fixed_snapshot/control_000'
    assert evidence['details']['details']['seed']==0
    assert evidence['completed'][0]['id']=='fixed_snapshot/main'
    plan=json.loads((tmp_path/'suite'/'planned_runs.json').read_text())
    assert plan['control_seeds']==[0,1]


@pytest.mark.real_data
def test_real_random_state_recomputes_centers_and_preserves_time(actual_config):
    from regime_alloc.backtest.experiments import random_state
    from regime_alloc.backtest.engine import ExecutionContext
    from threadpoolctl import threadpool_limits
    from dataclasses import replace
    for profile in ('fixed_snapshot','vintage_lagged'):
        cfg=replace(actual_config,research=replace(actual_config.research,profile=profile))
        ctx=ExecutionContext(cfg)
        with threadpool_limits(1):
            original=ctx.state(cfg,'2020-04')
            control=random_state(original,0)
        np.testing.assert_array_equal(np.bincount(original.labels,minlength=6),np.bincount(control.labels,minlength=6))
        assert control.partition_id!=original.partition_id
        assert control.ledger==original.ledger and control.transform_hash==original.transform_hash
        labels=control.labels.to_numpy()
        assert any(len(set(labels[original.labels.to_numpy()==k]))>1 for k in range(6) if (original.labels.to_numpy()==k).any())
        points=original.condition_scores.to_numpy()
        centers=[points[labels==0].mean(axis=0)]
        for k in range(1,6):
            group=points[labels==k]
            directions=group/np.linalg.norm(group,axis=1)[:,None]
            mean=directions.mean(axis=0)
            centers.append(mean/np.linalg.norm(mean))
        np.testing.assert_allclose(control.centers,np.vstack(centers))
        assert not np.array_equal(control.transition_matrix,original.transition_matrix)
        assert control.metadata['parent_partition_id']==original.partition_id
