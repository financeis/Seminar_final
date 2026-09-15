import math
import numpy as np
import pytest
from regime_alloc.backtest.statistics import paired_tests, holm, circular_blocks, compare_profile


def test_paired_direction_ranks_and_holm_restore():
    r = paired_tests([3., 4., 8., 9.], [1., 2., 4., 4.])
    # differences 2,2,4,5: mean3.25, sample variance2.25.
    assert r['paired_t']['statistic'] == pytest.approx(3.25 / .75)
    assert r['paired_t']['p'] < .02
    assert r['nemenyi']['treatment_mean_rank'] == 1.
    assert r['nemenyi']['control_mean_rank'] == 2.
    assert r['nemenyi']['p'] == pytest.approx(math.erfc(math.sqrt(2)))
    assert holm([.04, .01, None, .03]) == [.09, .04, None, .09]
    assert paired_tests([1., 2., 4., 4.], [3., 4., 8., 9.])['paired_t']['p'] > .98


def test_undefined_constant_and_small_true_variation():
    for a,b,reason in [([1.,2.],[0.,1.],'constant_metric_differences'), ([None,2.],[1.,1.],'fewer_than_two_pairs')]:
        r=paired_tests(a,b)
        assert r['paired_t']['p'] is None and r['paired_t']['reason']==reason
        assert r['nemenyi']['p'] is None
    r=paired_tests([1e-20, 2e-20, 4e-20], [0.,0.,0.])
    assert r['paired_t']['p'] is not None
    assert paired_tests([1.,2.,4.], [0.,0.,0.])['nemenyi']['p'] is not None


def test_circular_blocks_exact_and_joint():
    indices=circular_blocks(7,3,4,seed=19)
    starts=np.random.default_rng(19).integers(0,7,size=(4,3))
    expected=np.array([[(s+j)%7 for s in row for j in range(3)][:7] for row in starts])
    np.testing.assert_array_equal(indices,expected)
    assert np.array_equal(indices,circular_blocks(7,3,4,seed=19))


def test_primary_is_mean_of_seed_metrics_and_shared_indices():
    months=['2020-%02d'%m for m in range(1,7)]
    t=np.array([.03,-.01,.02,-.02,.04,.01])
    controls=np.array([[.01,-.01,.01,-.01,.01,-.01],[.09,-.02,.03,-.04,.06,.02]])
    pairs={name:(t,controls if name!='bl_vs_mvo' else controls[:1]) for name in ['naive_vs_random','ridge_vs_random','bl_vs_mvo']}
    r=compare_profile('fixed_snapshot',months,pairs,seeds=[0,1],repetitions=4,block_lengths=(3,),seed=19,auxiliary_seed=5)
    row=next(x for x in r['comparisons'] if x['pair_id']=='naive_vs_random' and x['metric']=='sharpe')
    idx=np.array(r['bootstrap_indices']['3'][0])
    sharpe=lambda a:a.mean()/a.std(ddof=1)*math.sqrt(12)
    expected=sharpe(t[idx])-np.mean([sharpe(c[idx]) for c in controls])
    assert row['bootstrap_difference_ci']['3']['draws'][0]['value']==pytest.approx(expected)
    assert len(row['seed_distribution'])==2
    assert r['paired_monthly_returns']['naive_vs_random']==pytest.approx((t-controls.mean(axis=0)).tolist())


def test_undefined_bootstrap_has_reasons_and_writes_strict_json(tmp_path):
    from regime_alloc.contracts import write_json
    t=np.full(6,.1); c=np.full((2,6),.2)
    pairs={name:(t,c if name!='bl_vs_mvo' else c[:1]) for name in ['naive_vs_random','ridge_vs_random','bl_vs_mvo']}
    result=compare_profile('fixed_snapshot',['2020-%02d'%m for m in range(1,7)],pairs,seeds=[0,1],repetitions=2,block_lengths=(3,))
    row=result['comparisons'][0]
    assert row['paired_t']['p'] is None and row['bootstrap_difference_ci']['3']['valid_draws']==0
    assert row['bootstrap_difference_ci']['3']['draws'][0]['valid_controls']==0
    write_json(tmp_path/'stats.json',result)


def test_two_method_extreme_tail_stays_positive():
    r=paired_tests(list(range(1,101)),[0.]*100)
    assert r['nemenyi']['p']==pytest.approx(1.523970604832094e-23,rel=1e-13)


def test_invalid_nonfinite_input_is_execution_error():
    from regime_alloc.contracts import ResearchError
    with pytest.raises(ResearchError,match='nonfinite'):
        paired_tests([1.,float('nan')],[1.,2.])
