"""S07: independent regime oracles followed by execution of original source ASTs.

No original script is imported or run as a program. Only listed numerical AST
blocks execute; FRED CSV is read, while all output stays in the audit directory.
CLI exit 1 means an evidenced original-code property failed, not a harness crash.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from .common import load_functions, record, source_ref, write_results, jsonable
except ImportError:
    from common import load_functions, record, source_ref, write_results, jsonable

S1, S2, S3, AGG = 'Section5_step1.py', 'Section5_step2.py', 'Section3.py', 'Section5_step3.py'
TOL = 1e-10


def close(a, b, atol=TOL):
    return np.allclose(a, b, atol=atol, rtol=0)


def block(repo, filename, first, last, scope):
    """Execute whole original statements within a reviewed, I/O-free line range.

    The data-preparation ranges are the only exception: their read_csv calls are
    explicitly allowed and are supplied an absolute path to the original CSV.
    Never descend into a selected statement, so loops/ifs execute exactly once.
    """
    tree = ast.parse((repo / filename).read_text(encoding='utf-8-sig'))
    nodes = []
    def visit(node):
        if isinstance(node, ast.stmt) and first <= node.lineno and node.end_lineno <= last:
            nodes.append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    assert nodes, (filename, first, last)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(repo / filename), 'exec'), scope)
    return scope


def ref(repo, filename, symbol=None, lines=None):
    result = source_ref(repo, filename, symbol)
    if symbol:
        # common.load_functions loads only top-level definitions, last duplicate wins.
        tree = ast.parse((repo / filename).read_text(encoding='utf-8-sig'))
        node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol][-1]
        result.update(line_start=node.lineno, line_end=node.end_lineno)
    if lines:
        result.update(symbol='original AST block', line_start=lines[0], line_end=lines[1])
    return result


def eq1(distances):
    # Independent scalar expression; deliberately undefined for all-zero or K=1.
    d = [float(x) for x in distances]
    assert len(d) > 1 and sum(d) > 0 and min(d) >= 0
    return np.array([(1 - x / sum(d)) / (len(d) - 1) for x in d])


def eq4(p0, typical):
    if p0 == 1:
        return np.r_[1., np.zeros(len(typical))]
    score = -max(typical) * math.log2(1 - p0)
    denominator = score + sum(typical)
    return np.array([score / denominator] + [p / denominator for p in typical])


def transitions(labels, k, outgoing=True):
    counts = np.zeros((k, k), int)
    for origin, destination in zip(labels, labels[1:]):
        counts[origin, destination] += 1
    denominator = counts.sum(axis=1) if outgoing else np.bincount(labels, minlength=k)
    result = np.zeros((k, k), float)
    for i in range(k):
        if denominator[i]:
            result[i] = counts[i] / denominator[i]
        elif outgoing:
            result[i, i] = 1
        else:
            result[i] = np.nan
    return counts, denominator, result


def valid(p):
    return bool(np.isfinite(p).all() and np.min(p) >= 0 and abs(np.sum(p) - 1) < TOL)


def original_combine(repo, scope, p0, typical):
    state = dict(scope, P_reg0=p0, p_stage2=np.array(typical), r=len(typical), hard=np.array([0]))
    block(repo, S1, 211, 217, state)
    return state['probs']


def original_aggregate(repo, p, predictions):
    date = pd.Timestamp('2003-12-01')
    date_next = pd.Timestamp('2004-01-01')
    a = {'sasdate_t': date, 'sasdate_tp1': date_next}
    b = dict(a)
    a.update({f'ptp1_R{i}': x for i, x in enumerate(p)})
    b.update({f'yhat_R{i+1}_AUDIT': x for i, x in enumerate(predictions)})
    state = {'np': np, 'pd': pd, 'r': 5, 'TICKERS': ['AUDIT'],
             'step1': pd.DataFrame([a]), 'step2': pd.DataFrame([b])}
    block(repo, AGG, 329, 341, state)
    return float(state['df_agg']['AUDIT'].iloc[0])


def window(repo, filename, funcs, X, prev=None):
    state = dict(funcs, scores=funcs['pca_svd_95'](funcs['zscore_window'](X))[0],
                 W=len(X), r=5, prevC2=prev)
    block(repo, filename, 182 if filename == S1 else 234, 199 if filename == S1 else 251, state)
    return state


def window_probs(repo, s1, state):
    state = dict(s1, **{k: state[k] for k in ['scores', 'cen2', 'outlier_id', 'C2', 'rr', 'hard', 'W', 'r']})
    block(repo, S1, 202, 232, state)
    return state['p_t'], state['p_tp1'], state['E']


def original_forecast(repo, s2, state, target):
    # Controlled diagnostic asset, NOT actual ETF performance.
    n = len(state['hard'])
    scope = dict(s2, scores=state['scores'], hard=state['hard'], W=n, r=5,
                 dates=pd.date_range('2000-01-01', periods=n+1, freq='MS'), t_end=n-1,
                 idx=np.arange(n), R=pd.DataFrame({'AUDIT': target}), TICKERS=['AUDIT'],
                 lambdas=np.logspace(-4, 2, 8))
    block(repo, S2, 254, 286, scope)
    return np.array([scope['yhat_row'][f'yhat_R{i}_AUDIT'] for i in range(1, 6)]), scope['cnt_row']


def run(repo: Path, output: Path) -> list[dict]:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    results = []
    s1 = load_functions(repo, S1)
    s2 = load_functions(repo, S2)
    s3 = load_functions(repo, S3, names=['euclid_sq_dists_to_centers', 'kmeanspp_init_l2', 'l2_kmeans',
        'normalize_rows', 'cosine_dists_to_centers', 'kmeanspp_init_cosine', 'spherical_kmeans',
        'probs_from_distances', 'cosine_dist'])
    matching3 = load_functions(repo, AGG, names=['match_centroids_cosine'])
    def add(number, title, passed, expected, observed, *, refs=(), inputs=None, oracle='',
            classification='defined_property', confidence='confirmed', status=None, notes=()):
        item = record(f'REG-{number:03}', title, passed, expected, observed,
                      source_refs=refs, inputs=inputs, oracle=oracle, tolerance={'atol':TOL,'rtol':0},
                      notes=notes, status=status)
        item.update(classification=classification, confidence=confidence)
        results.append(item)

    # Self-check each independent oracle with hand-computable constants first.
    manual_E = np.array([[.5, .5, 0], [0, 0, 1], [1, 0, 0]])
    ct, den, Eo = transitions([0, 1, 2, 0, 0], 3)
    self_ok = (close(eq1([1, 3]), [.75, .25]) and
               close(eq1([1, 2, 3]), [5/12, 1/3, 1/4]) and
               close(eq4(.5, [.6, .4]), [.375, .375, .25]) and
               close(Eo, manual_E) and close(np.array([.2,.3,.5]) @ Eo, [.6,.1,.3]))
    add(1, 'Independent equations (1), (4), (5), (7) hand self-check', self_ok, True, self_ok,
        oracle='Scalar equations and manual sequence 0→1→2→0→0; no source function used.')
    assert self_ok, 'Independent oracle self-check failed'

    D = np.array([[1., 3.], [2., 2.], [0., 4.]])
    expected = np.array([eq1(row) for row in D])
    observed = {'Section3': s3['probs_from_distances'](D),
                'Step1': np.array([s1['fuzzy_probs_from_dist'](row) for row in D])}
    add(2, 'Equation (1): finite nonnegative normalized probabilities on valid distances',
        all(close(v, expected, atol=TOL) and all(valid(x) for x in v) for v in observed.values()),
        expected, observed, refs=[ref(repo,S3,'probs_from_distances'),ref(repo,S1,'fuzzy_probs_from_dist')],
        inputs=D, oracle='P_i=(1-d_i/sum(d))/(K-1)')

    stage1_scope = dict(s3, X_full=np.array([[0.,0.]]), C2=np.array([[1.,0.],[3.,0.]]))
    block(repo,S3,291,292,stage1_scope)
    dl2 = np.array([np.linalg.norm(np.array([0.,0.])-c) for c in stage1_scope['C2']])
    add(3, 'L2 distance is square-rooted before probability conversion',
        close(stage1_scope['D2'], [[1,3]]) and close(s1['fuzzy_probs_from_dist'](dl2), [.75,.25]),
        {'L2_probability':[.75,.25],'squared_distance_probability':[.9,.1]},
        {'Section3_distance':stage1_scope['D2'], 'Step1_probability':s1['fuzzy_probs_from_dist'](dl2),
         'counterfactual_squared':eq1(dl2**2)}, refs=[ref(repo,S3,lines=(285,297)),ref(repo,S1,lines=(202,205))],
        inputs={'x':[0,0],'centers':[[1,0],[3,0]]}, oracle='Distances 1,3 versus squared distances 1,9.')

    ptyp = [.6,.4]
    vals = [0.,.25,.5,.75,1.]
    cp = [original_combine(repo,s1,p,ptyp) for p in vals]
    add(4, 'Equations (2)-(4), (6): ordinary p0 and equality at 0.5',
        all(close(cp[i],eq4(vals[i],ptyp),atol=TOL) and valid(cp[i]) for i in [1,2,3]),
        [eq4(v,ptyp) for v in vals[1:4]], cp[1:4], refs=[ref(repo,S1,lines=(211,217))],
        inputs={'p0':vals[1:4],'typical':ptyp}, oracle='q0=-max(Ptyp)*log2(1-p0); normalize with sum.')
    add(5, 'p0 endpoints: epsilon clipping differs from the exact limit',
        all(valid(p) for p in cp) and abs(cp[0][0])<TOL and cp[-1][0]<1-TOL,
        {'p0=0':eq4(0,ptyp),'p0=1_limit':eq4(1,ptyp)},
        {'p0=0':cp[0],'p0=1':cp[-1],'finite_and_normalized':all(valid(p) for p in cp)},
        refs=[ref(repo,S1,lines=(211,217))], inputs={'p0':[0,1],'typical':ptyp,'eps':1e-12},
        oracle='Exact endpoint at p0=1 follows the q0→infinity limit.',
        classification='implementation_choice', notes=['Clipping is a finite approximation; exact R0=1 is not attained.'])

    edge = {}
    for name, d in [('all_zero',[0.,0.,0.]),('one_cluster',[2.]),('equal_positive',[2.,2.,2.])]:
        edge[name]={'Step1':s1['fuzzy_probs_from_dist'](d),
                    'Section3':s3['probs_from_distances'](np.array([d]))[0]}
    edge_ok = (all(valid(p) for policies in edge.values() for p in policies.values()) and
               not np.array_equal(edge['all_zero']['Step1'],edge['all_zero']['Section3']) and
               all(close(p,[1/3]*3) for p in edge['equal_positive'].values()) and
               all(np.array_equal(p,[1.]) for p in edge['one_cluster'].values()))
    add(6, 'Identical centers and insufficient normal clusters: unspecified Eq (1) boundaries', edge_ok,
        {'all_zero':'undefined; any fallback must be declared','one_cluster':'undefined K-1=0',
         'equal_positive':[1/3]*3}, edge, refs=[ref(repo,S1,'fuzzy_probs_from_dist'),ref(repo,S3,'probs_from_distances')],
        inputs={'all_zero':[0,0,0],'one_cluster':[2],'equal_positive':[2,2,2]},
        oracle='Sum(d)=0 and K=1 invalidate the original denominator.', classification='implementation_choice',
        notes=['Step1 all-zero chooses first minimum; Section3 all-zero chooses uniform. Neither is the unique paper-defined answer.'])
    zero_dist = s1['cosine_distance_matrix'](np.zeros((1,2)),np.array([[1.,0.],[0.,1.]]))
    tie = original_combine(repo,s1,.5,[.5,.5])
    add(7,'Zero-vector cosine and hard-label ties are deterministic conventions',
        close(zero_dist,[[1,1]]) and int(tie.argmax())==0,
        {'zero_vector_distances':[[1,1]],'tie_argmax':0}, {'distances':zero_dist,'tie_probs':tie,'argmax':tie.argmax()},
        refs=[ref(repo,S1,'cosine_distance_matrix'),ref(repo,S1,lines=(211,217))],
        inputs={'x':[0,0],'centers':[[1,0],[0,1]],'p0':.5,'typical':[.5,.5]},
        oracle='Cosine at zero is mathematically undefined; norm floor implies distance=1. np.argmax takes first tie.',
        classification='implementation_choice')
    tie_scope=dict(s1, lab2=np.array([0,0,1,1]), W=4)
    block(repo,S1,183,185,tie_scope)
    small = window(repo,S1,s1,np.array([[0.,0.],[1.,0.],[0.,1.],[2.,2.]]))
    pp, pn, _ = window_probs(repo,s1,small)
    add(8,'Algorithm 1 equal-sized first-stage groups and normal-count padding',
        tie_scope['outlier_id']==0 and small['rr']<5 and valid(pp) and valid(pn),
        {'equal_size_outlier_id':0,'small_window':'rr<5 and 6 finite normalized probabilities'},
        {'outlier_id':tie_scope['outlier_id'],'rr':small['rr'],'hard':small['hard'],'p':pp,'p_next':pn},
        refs=[ref(repo,S1,lines=(183,209))], inputs={'lab2':[0,0,1,1],'small_X':[[0,0],[1,0],[0,1],[2,2]]},
        oracle='Paper tie chooses A; rr=min(r, typical_count), omitted regimes padded with zero.',
        classification='implementation_choice', notes=['W=4 is a bounded branch diagnostic. In an ordinary W=48 split the larger group has at least 24 points.'])

    errors={}
    for name in ['l2_kmeans','spherical_kmeans']:
        try:
            with np.errstate(all='ignore'):
                s3[name](np.array([[1.,0.]]*4),k=2,n_init=1,rng=np.random.default_rng(42))
            errors[name]='no exception'
        except Exception as exc:
            errors[name]=f'{type(exc).__name__}: {exc}'
    add(9,'Section3 k-means++ initialization on identical vectors',all(v=='no exception' for v in errors.values()),
        'Declared handling when all remaining initialization weights are zero',errors,
        refs=[ref(repo,S3,'kmeanspp_init_l2'),ref(repo,S3,'kmeanspp_init_cosine')],
        inputs={'X':[[1,0]]*4,'k':2,'n_init':1,'seed':42},
        oracle='All distance weights are zero: 0/sum(0) is undefined.',
        classification='undefined_boundary', notes=['This is a degenerate-input limitation, not evidence of failure on the ordinary FRED panel.'])

    labels=np.array([0,1,2,0,0]); p=np.array([.2,.3,.5,0.])
    state=dict(s1,hard=labels,W=len(labels),r=3,probs=p)
    block(repo,S1,220,232,state)
    counts, outden, oracleE=transitions(labels,4)
    literalcounts, literalden, literalE=transitions(labels,4,outgoing=False)
    s3t=dict(s3,regime_df=pd.DataFrame({'final_regime_label':labels}))
    block(repo,S3,503,517,s3t)
    add(10,'Eq (5): counts, last-month denominator and empty row in both source paths',
        close(state['E'],oracleE) and close(s3t['E'],literalE[:3,:3]),
        {'counts':counts,'outgoing_denominator':outden,'occurrence_denominator':literalden,
         'Step1':oracleE,'paper_literal':literalE},
        {'Step1':state['E'],'Section3':s3t['E'],'Step1_row_sums':state['E'].sum(axis=1),
         'Section3_row_sums':s3t['E'].sum(axis=1)},
        refs=[ref(repo,S1,lines=(220,229)),ref(repo,S3,lines=(503,517))],inputs={'labels':labels,'K':4},
        oracle='Manual edges 0→1,1→2,2→0,0→0. N(0)=3 but N_out(0)=2; state 3 absent.',
        classification='paper_ambiguity',notes=['Section3 follows literal occurrence denominator. Step1 makes an explicit stochastic correction and uses empty-row self-loops.'])
    add(11,'Eq (7): row-origin/column-destination direction and probability conservation',
        close(state['p_tp1'],[.6,.1,.3,0]) and valid(state['p_tp1']), [.6,.1,.3,0],
        {'original_p_next':state['p_tp1'],'wrong_direction_E_times_p':state['E']@p,
         'literal_p_next_present_states':p[:3]@literalE[:3,:3],
         'literal_next_sum':float((p[:3]@literalE[:3,:3]).sum())}, refs=[ref(repo,S1,lines=(231,232))],
        inputs={'p':p,'labels':labels},oracle='p_next[j]=sum_i p[i]*E[i,j]; expected [.6,.1,.3,0].')
    # Sparse Section3 label ids expose a truly empty row (0/0), distinct from the finite final-month deficit.
    sparse=dict(s3,regime_df=pd.DataFrame({'final_regime_label':[0,2,0]}))
    with np.errstate(all='ignore'):
        block(repo,S3,503,517,sparse)
    add(12,'Section3 absent intermediate regime creates nonfinite transition row',bool(np.isfinite(sparse['E']).all()),
        'Empty-row policy must be declared before E can be treated as a probability kernel',
        {'E':sparse['E'],'occurs':sparse['occurs']},refs=[ref(repo,S3,lines=(503,517))],
        inputs={'labels':[0,2,0]},oracle='Regime 1 has 0 occurrences; division 0/0 produces NaN.',
        classification='undefined_boundary',notes=['FAIL concerns finite output on this boundary input. The paper leaves empty rows undefined; incidence in real FRED windows is not established.'])

    centers=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    transforms={'sign_flip':np.diag([-1.,1.]), 'quarter_turn':np.array([[0.,-1.],[1.,0.]])}
    matching={S1:s1,S2:s2,AGG:matching3}
    for num,(case,Q) in enumerate(transforms.items(),13):
        curr=centers@Q
        obs={name:sc['match_centroids_cosine'](centers,curr) for name,sc in matching.items()}
        aligned=s1['match_centroids_cosine'](centers,curr@Q.T)
        add(num,f'PCA {case}: identical physical centroids retain identity only after basis alignment',
            all(np.array_equal(x,np.arange(4)) for x in obs.values()), np.arange(4),
            {'unaligned_source_permutations':obs,'correct_common_basis_permutation':aligned},
            refs=[ref(repo,name,'match_centroids_cosine') for name in matching],
            inputs={'physical_centers':centers,'previous_basis':np.eye(2),'current_basis':Q,'current_scores':curr},
            oracle='The rows refer to the same four physical centers; changing coordinates alone must preserve row identity.',
            classification='coordinate_invariance',notes=['This establishes an implementation invariance failure, not a measured prevalence over all rolling windows.'])
    prev=np.c_[centers,np.zeros(4)]
    basis=np.array([[0.,-1.],[1.,0.],[0.,0.]])
    current=prev@basis
    dimperm=s1['match_centroids_cosine'](prev,current)
    add(15,'Changed PCA dimension: truncation does not establish a common coordinate system',
        np.array_equal(dimperm,np.arange(4)),np.arange(4),
        {'source_permutation':dimperm,'common_space_permutation':s1['match_centroids_cosine'](prev,current@basis.T)},
        refs=[ref(repo,S1,'match_centroids_cosine'),ref(repo,S2,'match_centroids_cosine')],
        inputs={'previous_scores':prev,'current_basis':basis,'current_scores':current},
        oracle='The dropped third coordinate is identically zero; identity is fully recoverable, but prefix truncation miscompares rotated axes.',
        classification='coordinate_invariance')
    lost=np.array([[1.,1.],[1.,-1.]])
    add(16,'Dimension reduction may also destroy identity information',
        not np.array_equal(lost[0],lost[1]) and np.array_equal(lost[0,:1],lost[1,:1]),
        'Cannot identify which physical center produced the same projected value',
        {'previous_centers':lost,'projected':lost[:,:1],
         'source_arbitrary_tie_permutation':s1['match_centroids_cosine'](lost,lost[:,:1])},
        refs=[ref(repo,S1,'match_centroids_cosine')],inputs={'projection':[[1],[0]]},
        oracle='Both centers project to [1]; no matching algorithm can recover identity from this projection alone.',
        classification='mathematical_limit')

    perm=np.array([4,0,3,1,2]); p=np.array([.1,.2,.1,.25,.15,.2]); y=np.array([.04,-.02,.01,.03,-.05])
    base=original_aggregate(repo,p,y)
    both=original_aggregate(repo,np.r_[p[0],p[1:][perm]],y[perm])
    only=original_aggregate(repo,np.r_[p[0],p[1:][perm]],y)
    add(17,'Eq (14) simultaneous permutation invariance and one-sided permutation counterexample',
        abs(base-both)<TOL and abs(base-only)>TOL,
        {'both_permuted_equals_base':True,'one_side_changes_result':True},
        {'base':base,'both_permuted':both,'probabilities_only_permuted':only,'delta':only-base},
        refs=[ref(repo,AGG,lines=(329,341))],inputs={'p':p,'y':y,'perm_0based_normal':perm},
        oracle='Reindex both summands leaves sum invariant; reindex only p changes the paired products.',
        notes=['A passing control test establishes why equal dates and column strings alone do not prove shared regime identity.'])

    # Actual FRED panel, original source preprocessing, bounded deterministic windows.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        prep=dict(s1,CSV=repo/'FRED-MD_2024m12.csv')
        block(repo,S1,14,68,prep)
        prep2=dict(s2,FRED_CSV=repo/'FRED-MD_2024m12.csv')
        block(repo,S2,22,69,prep2)
    df=prep['df_cc']; X=df.drop(columns='sasdate').to_numpy(float)
    assert df.equals(prep2['df_cc'])
    cut=int(np.flatnonzero(df.sasdate.ge('1999-01-01'))[0])
    starts=[0,cut,cut+1]
    rawhash=hashlib.sha256((repo/'FRED-MD_2024m12.csv').read_bytes()).hexdigest()
    actual=[]
    target_rng=np.random.default_rng(714)
    # Synthetic diagnostic returns shared by both labelings, fixed independent of labels.
    targets=target_rng.normal(0,.02,len(X)+1)
    for start in starts:
        a=window(repo,S1,s1,X[start:start+48])
        b=window(repo,S2,s2,X[start:start+48])
        assert np.isfinite(a['scores']).all() and np.array_equal(a['scores'],b['scores'])
        contingency=np.zeros((5,5),int)
        for u,v in zip(a['hard'],b['hard']):
            if u>0 and v>0: contingency[u-1,v-1]+=1
        row,col=linear_sum_assignment(-contingency)
        mapping=np.r_[0,np.zeros(5,dtype=int)]
        for u,v in zip(row,col): mapping[v+1]=u+1
        aligned=mapping[b['hard']]
        pa,pna,Ea=window_probs(repo,s1,a)
        pb,pnb,Eb=window_probs(repo,s1,b)
        pnb_aligned=np.zeros(6)
        for label_b in range(6): pnb_aligned[mapping[label_b]]=pnb[label_b]
        yb,counts_b=original_forecast(repo,s2,b,targets[start:start+49])
        yb_aligned=np.zeros(5)
        for label_b in range(1,6): yb_aligned[mapping[label_b]-1]=yb[label_b-1]
        source_value=original_aggregate(repo,pna,yb)
        mapped_value=original_aggregate(repo,pna,yb_aligned)
        stage_consistency={}
        for name,s in [(S1,a),(S2,b)]:
            nearest=np.argmin(((s['scores'][:,None,:]-s['cen2'][None,:,:])**2).sum(axis=2),axis=1)
            st2nearest=np.argmax((s['scores'][s['typical_ids']]/np.maximum(np.linalg.norm(s['scores'][s['typical_ids']],axis=1,keepdims=True),1e-12))@s['C2'].T,axis=1)
            stage_consistency[name]={'stage1_mismatches':int(np.sum(nearest!=s['lab2'])),
                                    'stage2_mismatches':int(np.sum(st2nearest!=s['lab_s2']))}
        item={'start_index':start,'dates':df.sasdate.iloc[start:start+48].dt.strftime('%Y-%m-%d').tolist(),
              'shape':X[start:start+48].shape,'pca_dimensions':a['scores'].shape[1],
              'Step1_hard':a['hard'],'Step2_hard':b['hard'],
              'raw_mismatches':int(np.sum(a['hard']!=b['hard'])),
              'best_normal_label_mapping_Step2_to_Step1':mapping,
              'best_aligned_mismatches':int(np.sum(a['hard']!=aligned)),
              'outlier_membership_mismatches':int(np.sum((a['hard']==0)!=(b['hard']==0))),
              'contingency_normal':contingency,'Step1_p_t':pa,'Step1_p_next':pna,
              'Step2_recomputed_p_t_diagnostic':pb,'Step2_recomputed_p_next_diagnostic':pnb,
              'p_next_raw_L1_difference':float(np.abs(pna-pnb).sum()),
              'p_next_aligned_L1_difference':float(np.abs(pna-pnb_aligned).sum()),
              'both_probabilities_valid':all(valid(v) for v in [pa,pna,pb,pnb]),
              'Step2_Ridge_diagnostic_forecasts':yb,'Step2_training_counts_before_fallback':counts_b,
              'Step3_unmapped_weighted_prediction':source_value,
              'best_membership_mapping_weighted_prediction':mapped_value,
              'absolute_prediction_difference':abs(source_value-mapped_value),
              'final_labels_vs_centers':stage_consistency}
        actual.append(item)
    add(18,'Actual identical FRED 48-row inputs: Step1/2 regime identifiers and partitions',
        all(x['raw_mismatches']==0 for x in actual),
        'Shared identifiers require same memberships or an explicit inter-stage mapping before aggregation',actual,
        refs=[ref(repo,S1,lines=(14,68)),ref(repo,S1,lines=(182,199)),ref(repo,S2,lines=(234,251)),ref(repo,AGG,lines=(329,341))],
        inputs={'FRED_sha256':rawhash,'windows_start_indices':starts,'Step1':{'l2_seed':42,'cos_seed':123,'n_init':25,'iters':200},
                'Step2':{'l2_seed':1,'cos_seed':2,'n_init':8,'iters':80},'both_prevC2':None},
        oracle='Exact per-observation membership comparison; maximize normal-label agreement with Hungarian assignment while fixing outlier R0.',
        classification='cross_stage_identity',
        notes=['Cold starts on the same real window isolate actual partition/identifier differences; they do not claim to replay every prior production matching state.',
               'Different seeds alone were not used as proof. Original functions were executed; raw and best-aligned differences are both shown.'])
    add(19,'Original Ridge and Step3 aggregation: consequence of unmapped regime identifiers',
        all(x['absolute_prediction_difference']<TOL for x in actual),
        'Relabeling Step2 into the Step1 identity should not change the intended correctly paired aggregation',
        [{k:x[k] for k in ['start_index','Step2_Ridge_diagnostic_forecasts','Step3_unmapped_weighted_prediction',
                           'best_membership_mapping_weighted_prediction','absolute_prediction_difference']} for x in actual],
        refs=[ref(repo,S2,lines=(254,286)),ref(repo,AGG,lines=(329,341))],
        inputs={'synthetic_target_seed':714,'distribution':'N(0,0.02^2)','length':len(targets),
                'FRED_sha256':rawhash,'window_start_indices':starts},
        oracle='Use the same actual macro window, original Ridge loop, fixed synthetic asset target, original aggregation; only Step2 regime column order changes.',
        classification='cross_stage_identity', confidence='supported',
        notes=['This is controlled prediction sensitivity, not real ETF performance or a claim that optimal membership alignment is a full repair.',
               'Residual mismatches after optimal relabeling mean a pure permutation cannot make the two partitions equal.'])
    add(20,'Actual bounded windows: returned cluster labels agree with returned centers',
        all(all(c['stage1_mismatches']==0 and c['stage2_mismatches']==0 for c in x['final_labels_vs_centers'].values()) for x in actual),
        'Every returned label selects a closest/farthest-similarity returned center',
        [x['final_labels_vs_centers'] for x in actual],
        refs=[ref(repo,S1,'kmeans_l2'),ref(repo,S1,'spherical_kmeans'),ref(repo,S2,'kmeans_l2'),ref(repo,S2,'spherical_kmeans')],
        inputs={'FRED_sha256':rawhash,'window_start_indices':starts},
        oracle='Reassign each observation using the returned centers and compare with the returned label.',
        notes=['PASS applies only to these windows; finite iteration caps still need a final reassignment policy.'])

    # Execute original calendar intersection with a declared synthetic ETF calendar.
    calendar=df.sasdate.iloc[cut:].copy()
    prep2['R']=pd.DataFrame({'AUDIT':np.zeros(len(calendar))},index=pd.DatetimeIndex(calendar))
    block(repo,S2,72,77,prep2)
    calendar_evidence={'ETF_calendar_kind':'declared diagnostic calendar; no actual ETF CSV available',
        'Step1_first_decision':df.sasdate.iloc[47], 'Step2_first_decision':prep2['df_cc'].sasdate.iloc[47],
        'first_Step1_window':df.sasdate.iloc[[0,47]].tolist(),
        'first_Step2_window':prep2['df_cc'].sasdate.iloc[[0,47]].tolist(),
        'Step1_prevC2_initial':None,'Step2_prevC2_initial':None,
        'Step1_has_prior_windows_before_Step2_first_decision':cut}
    add(21,'Calendar clipping and independent matching histories have different first anchors',
        df.sasdate.iloc[47]!=prep2['df_cc'].sasdate.iloc[47] and
        len(prep2['df_cc'])==len(calendar) and cut>0,
        'Step2 intersects ETF dates before rolling; each script independently initializes prevC2=None',calendar_evidence,
        refs=[ref(repo,S1,lines=(167,175)),ref(repo,S2,lines=(72,77)),ref(repo,S2,lines=(194,225))],
        inputs={'diagnostic_ETF_calendar_start':calendar.iloc[0], 'diagnostic_ETF_calendar_end':calendar.iloc[-1]},
        oracle='Apply original set intersection, then select index W-1 in each independent calendar.',
        classification='source_path',confidence='supported',
        notes=['Actual ETF calendar/forecast CSVs are missing from tracked input; exact production anchor and full historical mappings remain unresolved.'])
    add(22,'Actual source connection does not carry partition identity across Step1→Step2→Step3',
        all(x['raw_mismatches']==0 for x in actual),
        'A shared partition/model artifact or a verified mapping accompanying probabilities and conditional predictors',
        {'Step1_exports':['dates','probabilities','pca dimension','r','E matrices'],
         'Step2_reads':['FRED CSV','ETF return CSV'], 'Step2_reclusters':True,
         'Step3_merge_keys':['sasdate_t','sasdate_tp1'],
         'Step3_multiplication':'ptp1_Ri * yhat_Ri_AUDIT; no partition/basis provenance check'},
        refs=[ref(repo,S1,lines=(234,246)),ref(repo,S2,lines=(13,23)),ref(repo,S2,lines=(234,251)),ref(repo,AGG,lines=(329,341))],
        inputs={'same_window_counterexamples':['REG-018','REG-019'],'permutation_control':'REG-017'},
        oracle='Read the actual save/read/recluster/merge/product code path; compare concrete membership evidence, not matching column names.',
        classification='cross_stage_identity')

    write_results(output,'regimes',results)
    folder=output/'regimes'
    counts=Counter(x['status'] for x in results)
    lines=['# 레짐·확률·연결 감사 (S07)','',
           '## 범위와 판정','',
           '지정 원문의 해석은 `reports/paper_audit/methodology.md`를 기준으로 한다. 독립 손계산을 먼저 확인하고, 원본 함수 또는 명시된 AST 구간을 실행했다. 원 연구 파일·CSV·NPZ를 변경하거나 다운로드하지 않았다.',
           '',f'실행 결과: {dict(counts)}. FAIL은 검사 대상 성질의 실패이며 감사 실행 오류가 아니다. undefined_boundary 분류의 FAIL은 실제 예외/NaN 관측을 뜻하며, 원문 미정 경계와 실제 FRED 발생 여부를 분리한다.',
           '', '## Confirmed — 확인된 사실','',
           '- 식 (1)의 정상 입력, 실제 L2/제곱거리 구분, 식 (4)의 내부 p0 값과 합 정규화, 식 (7)의 행벡터 방향이 통과했다.',
           '- Section3 식 (5)는 전체 출현 횟수를 분모로 쓰므로 마지막 레짐 행 합이 1 미만이다. Step1은 출발 횟수로 정규화하고 빈 행에 self-loop를 넣는다. 이는 원문의 문자 그대로 분모와 다른 명시적 보정이며, 원문 자체의 모호함과 분리한다.',
           '- 원 매칭 함수에 같은 실물 중심의 PCA 부호·회전·차원 변화 반례를 직접 넣으면 식별 순열이 바뀐다. 중심 좌표의 앞 m개를 자르는 것만으로 서로 다른 PCA 기저가 정렬되지 않는다.',
           '- Step1과 Step2는 실제 같은 FRED 48행 입력에서도 서로 다른 원시 레짐 식별자 및 일부 서로 다른 분할을 반환한다. 최적 정상 레짐 순열 정렬 후 차이도 REG-018에 보존했다.',
           '- Step2는 Step1 군집 아티팩트를 읽지 않고 재군집하며 Step3는 두 날짜와 R 번호로 곱한다. REG-017의 동시 순열 불변성 및 단측 반례가 연결 의미를 검증한다.',
           '', '## Supported — 범위가 제한된 영향 근거','',
           '- 실제 FRED 창과 고정 합성 자산 목표(seed 714)를 원 Ridge 루프에 넣었다. Step3의 정상 레짐 확률과 예측 열을 최적 멤버십 정렬 전후로 곱한 차이는 REG-019에 기록했다. 실제 ETF 수익·비중·성과 손실을 추정한 결과가 아니다.',
           '- 진단용 ETF 달력에 대한 원 코드 교집합 실행은 Step2 첫 창과 매칭 시작점이 달라짐을 보인다. 같은 실제 창을 양쪽 cold start로 비교한 REG-018은 seed/n_init 차이의 실제 효과를 분리한 진단이며 전체 과거 매칭 상태의 재생은 아니다.',
           '', '## Unresolved — 미확정 및 경계','',
           '- 추적된 실제 ETF CSV와 두 단계의 모델/군집 ID 계보가 없어 생산 실행의 정확한 시작 달력·전체 이전 PCA 기저·매칭 순열은 확정할 수 없다.',
           '- 0벡터의 cosine, 모든 거리 0, 일반 군집 1개와 빈 전이 행은 원식이 유일한 답을 정하지 않는다. Step1 argmin fallback과 Section3 균등 fallback을 구현 선택으로 기록했다. p0=1의 epsilon clipping은 극한 R0=1의 유한 근사다.',
           '- Section3 k-means++의 identical-vector 초기화 오류와 빈 중간 레짐의 NaN 전이 행은 경계 한계다. 정상 FRED 창 실패로 확대 해석하지 않는다.',
           '- 차원 축소로 서로 다른 실물 중심이 완전히 같은 좌표가 되면 어느 알고리즘도 투영 좌표만으로 유일한 정체성을 복원할 수 없다.',
           '', '## 논문 충실도와 통계적 타당성','',
           '논문 충실도: 식 (1), (4), (7)의 정상 경로는 수식과 일치한다. 식 (5)의 두 분모 구현과 p0/0거리 보정은 논문 경계·명세 문제로 별도 분류했다. 창 매칭에 특정 알고리즘을 논문 정답으로 가정하지 않았다.',
           '',
           '통계적 타당성: 확률과 레짐별 조건부 예측은 같은 집합을 가리켜야 한다. 실제 분할 비교와 원 함수 기반 반례가 현재 연결의 미보장 상태를 확인한다. 좌표계 변환에 따른 임의 ID 변경도 별도의 수학적 실패다. 본 감사는 성과 재현이나 새 전략 설계가 아니다.',
           '', '## 실제 FRED 창 수치 요약','',
           '| 창 | 원시 불일치 /48 | 최적 정상레짐 정렬 후 /48 | 정렬 전후 집계 예측 차이 |','|---|---:|---:|---:|']
    lines += [f"| {x['dates'][0]} ~ {x['dates'][-1]} | {x['raw_mismatches']} | {x['best_aligned_mismatches']} | {x['absolute_prediction_difference']:.10f} |" for x in actual]
    lines += ['', '집계 예측 차이는 고정 합성 목표와 원 Ridge·Step3 코드로 계산한 진단값이다. 실제 ETF 성과가 아니다.',
              '', '## 증거 색인','', '| ID | 상태 | 분류 | 제목 |','|---|---|---|---|']
    lines += [f"| {x['id']} | {x['status']} | {x['classification']} | {x['title']} |" for x in results]
    (folder/'notes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or args.repo/'reports/paper_audit/evidence'
    result=run(args.repo,output)
    for item in result:
        print(f"{item['id']} {item['status']}: {item['title']}")
    print('Summary:',json.dumps(dict(Counter(x['status'] for x in result)),sort_keys=True))
    return 1 if any(x['status']=='FAIL' for x in result) else 0


if __name__=='__main__':
    raise SystemExit(main())
