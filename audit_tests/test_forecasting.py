"""Forecasting diagnostics: execute original ASTs, never original script I/O.

FAIL is a preserved counterexample in the research code, not a runner failure.
The independent ridge oracle uses an augmented normal equation with an
unpenalized intercept; it never calls sklearn or the original ridge helpers.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

try:
    from .common import jsonable, load_functions, record, sha256, source_ref, write_results
    from .test_data_timing import assigned, exec_nodes, ref_range, tree_of
except ImportError:
    from common import jsonable, load_functions, record, sha256, source_ref, write_results
    from test_data_timing import assigned, exec_nodes, ref_range, tree_of


RTOL = ATOL = 1e-10
TOLERANCE = {'atol': ATOL, 'rtol': RTOL, 'equal_nan': False}
S2, S3, S4 = 'Section5_step2.py', 'Section5_step3.py', 'Section5_step4.py'


def close(a, b):
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=False))


def ridge_oracle(X, y, lam):
    """Build normal equations by scalar sums, with penalty diag(0,lam,...)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    rows = [[1.0, *map(float, row)] for row in X]
    width = X.shape[1] + 1
    gram = np.array([[sum(row[j] * row[k] for row in rows)
                      for k in range(width)] for j in range(width)])
    rhs = np.array([sum(row[j] * float(yi) for row, yi in zip(rows, y))
                    for j in range(width)])
    for j in range(1, width):
        gram[j, j] += float(lam)
    coef = np.linalg.solve(gram, rhs)
    return float(coef[0]), coef[1:]


def loo_oracle(X, y, lambdas):
    """Each held-out row gets a new intercept and ridge coefficient fit."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    predictions, losses = [], []
    for lam in lambdas:
        pred = []
        for held_out in range(len(y)):
            keep = [j for j in range(len(y)) if j != held_out]
            intercept, beta = ridge_oracle(X[keep], y[keep], lam)
            pred.append(intercept + sum(float(x) * float(b)
                                        for x, b in zip(X[held_out], beta)))
        predictions.append(pred)
        losses.append(sum((float(yi) - pi) ** 2 for yi, pi in zip(y, pred)) / len(y))
    return np.asarray(losses), np.asarray(predictions)


def equation14(probs, forecasts):
    """Printed Eq.14, indices 1..r; no R0 or probability renormalization."""
    return np.array([sum(float(probs[i + 1]) * float(forecasts[i, j])
                         for i in range(len(probs) - 1))
                     for j in range(forecasts.shape[1])])


def lo_oracle(values, l):
    """Finite-input Eq.18; index order resolves ties only in the oracle."""
    selected = sorted([i for i, x in enumerate(values) if x > 0],
                      key=lambda i: (-values[i], i))[:l]
    result = np.zeros(len(values))
    denom = sum(float(values[i]) for i in selected)
    if denom:
        for i in selected:
            result[i] = float(values[i]) / denom
    return result


def observe_selector(selector, call):
    """Observe actual original function locals at return; do not alter its AST."""
    observations = []
    prior = sys.getprofile()

    def observer(frame, event, arg):
        if event == 'return' and frame.f_code is selector.__code__:
            local = frame.f_locals
            item = {key: local[key].copy() if isinstance(local[key], np.ndarray) else local[key]
                    for key in ['X', 'y', 'x_pred', 'best_mse', 'best_idx',
                                'lam_best', 'beta_best', 'y_pred_centered']}
            caller = frame.f_back.f_locals
            for key in ['i_reg', 'tkr', 'Xi_mu', 'y_mu', 'y_vec', 'tau_idx']:
                if key in caller:
                    item[key] = caller[key].copy() if isinstance(caller[key], np.ndarray) else caller[key]
            observations.append(item)
        return observer

    try:
        sys.setprofile(observer)
        result = call()
    finally:
        sys.setprofile(prior)
    return result, observations


def original_regime_loop(repo, scores, hard, R, lambdas):
    scope = load_functions(repo, S2, ['ridge_precompute_B_Hdiag', 'ridge_select_lambda_and_predict'])
    rolling = next(n for n in tree_of(repo, S2).body
                   if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 't_end')
    nodes = [n for n in rolling.body if assigned(n, 'x_t') or
             (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'i_reg')]
    scope.update(scores=scores.copy(), hard=hard.copy(), R=R.copy(), lambdas=lambdas.copy(),
                 W=len(scores), r=5, idx=np.arange(len(scores)), TICKERS=list(R.columns),
                 yhat_row={}, lam_row={}, cnt_row={})
    _, observed = observe_selector(scope['ridge_select_lambda_and_predict'],
                                  lambda: exec_nodes(repo, S2, nodes, scope))
    ref = ref_range(repo, S2, nodes[0].lineno, nodes[-1].end_lineno, 'original_regime_ridge_AST')
    return scope, observed, ref


def original_aggregation(repo, step1, step2):
    body = tree_of(repo, S3).body
    first = next(i for i, n in enumerate(body) if assigned(n, 'prob_cols'))
    last = next(i for i, n in enumerate(body) if assigned(n, 'df_agg'))
    nodes = body[first:last + 1]
    scope = {'np': np, 'pd': pd, 'step1': step1.copy(), 'step2': step2.copy(), 'r': 5, 'TICKERS': ['A', 'B']}
    exec_nodes(repo, S3, nodes, scope)
    ref = ref_range(repo, S3, nodes[0].lineno, nodes[-1].end_lineno, 'original_merge_and_aggregation_AST')
    return scope['df_agg'], ref


def run(repo: Path, output: Path) -> list[dict]:
    repo, output = Path(repo), Path(output)
    destination = output / 'forecasting'
    destination.mkdir(parents=True, exist_ok=True)
    results, fixtures = [], {}

    def fixture(name, payload, seed=None):
        payload = jsonable(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        fixtures[name] = {'seed': seed, 'sha256': digest, 'data': payload}
        return {'kind': 'synthetic_diagnostic', 'fixture': 'forecasting/fixtures.json#' + name,
                'seed': seed, 'sha256': digest}

    def add(number, title, passed, expected, observed, *, refs=(), inputs=None,
            oracle='', classification='defined_property', notes=(), status=None):
        result = record(f'FOR-{number:03d}', title, bool(passed), expected, observed,
                        source_refs=refs, inputs=inputs, oracle=oracle, tolerance=TOLERANCE,
                        notes=notes, status=status)
        result['classification'] = classification
        results.append(result)

    source_scope = load_functions(repo, S2, ['ridge_precompute_B_Hdiag', 'ridge_select_lambda_and_predict'])
    precompute = source_scope['ridge_precompute_B_Hdiag']
    select = source_scope['ridge_select_lambda_and_predict']
    ridge_refs = [source_ref(repo, S2, name) for name in
                  ['ridge_precompute_B_Hdiag', 'ridge_select_lambda_and_predict']]
    lo = load_functions(repo, S4, ['long_only_weights'])['long_only_weights']
    lo_ref = source_ref(repo, S4, 'long_only_weights')

    # Oracle checks use small rational arithmetic results established by hand.
    X_hand, y_hand = np.array([[0.], [1.], [2.]]), np.array([1., 2., 4.])
    intercept, beta = ridge_oracle(X_hand, y_hand, 2.)
    hand_mse, hand_loo = loo_oracle(X_hand, y_hand, [2.])
    p = np.array([.5, .1, .15, .05, .1, .1])
    forecasts = np.array([[.01, .05], [.02, .01], [-.03, .02], [.04, .03], [.05, .04]])
    oracle_checks = {
        'intercept_19_over_12': close(intercept, 19/12), 'beta_3_over_4': close(beta, [.75]),
        'prediction_23_over_6': close(intercept + 3 * beta[0], 23/6),
        'loo_predictions': close(hand_loo[0], [2.4, 2.5, 1.8]), 'loo_mse_47_over_20': close(hand_mse, [2.35]),
        'equation14': close(equation14(p, forecasts), [.0115, .0145]),
        'lo_weights': close(lo_oracle([4., -2., 2., 1.], 2), [2/3, 0, 1/3, 0])}
    add(1, '독립 정규방정식·LOOCV·식14·식18 손계산 자체검사', all(oracle_checks.values()),
        '모든 손계산과 일치', oracle_checks,
        inputs=fixture('hand_checks', {'X': X_hand, 'y': y_hand, 'lambda': 2., 'p': p, 'forecasts': forecasts}),
        oracle='3행 단변량 ridge: 절편 19/12, 기울기 3/4; LOO 예측 [2.4,2.5,1.8], MSE 47/20.')
    if not all(oracle_checks.values()):
        raise RuntimeError('Independent oracle hand-check failed; do not issue downstream judgments.')

    rng = np.random.default_rng(814)
    scores = rng.normal(size=(48, 3)) + [2., -1., .3]
    hard = np.array([2]*5 + [3]*6 + [4]*10 + [5]*26 + [0])
    R = pd.DataFrame(rng.normal(scale=.03, size=(50, 2)) + [.012, -.004], columns=['A', 'B'],
                     index=pd.date_range('2000-01-01', periods=50, freq='MS'))
    fixed = np.array([2.])
    loop_input = fixture('ridge_loop_48', {'scores': scores, 'hard': hard, 'R': R.to_dict('list'),
                                          'dates': list(R.index), 'lambdas': fixed}, 814)
    state, traces, loop_ref = original_regime_loop(repo, scores, hard, R, fixed)
    coefficient_details = []
    for trace in traces:
        tau = trace['tau_idx']
        intercept_expected, beta_expected = ridge_oracle(scores[tau], trace['y_vec'], 2.)
        intercept_actual = trace['y_mu'] - float(trace['Xi_mu'].ravel() @ trace['beta_best'])
        pred_expected = intercept_expected + float(scores[-1] @ beta_expected)
        pred_actual = state['yhat_row'][f"yhat_R{trace['i_reg']}_{trace['tkr']}"]
        coefficient_details.append({'regime': trace['i_reg'], 'ticker': trace['tkr'],
                                    'intercept_expected': intercept_expected, 'intercept_observed': intercept_actual,
                                    'beta_expected': beta_expected, 'beta_observed': trace['beta_best'],
                                    'prediction_expected': pred_expected, 'prediction_observed': pred_actual,
                                    'max_beta_error': float(np.max(np.abs(beta_expected-trace['beta_best']))),
                                    'passed': close(beta_expected, trace['beta_best']) and
                                    close(intercept_expected, intercept_actual) and close(pred_expected, pred_actual)})
    add(2, 'Step2 고정 λ 계수·복원 절편·예측은 독립 정규방정식과 일치',
        all(v['passed'] for v in coefficient_details), '절편 비벌점 ridge 해와 일치', coefficient_details,
        refs=[*ridge_refs, loop_ref], inputs=loop_input,
        oracle='[1,X] 증강 행렬 정규방정식; penalty=diag(0,λ,λ,λ); 원 i_reg 루프를 그대로 실행.')

    shifted, translated, _ = original_regime_loop(repo, scores + [7., -3., 2.], hard, R + [1.25, -.75], fixed)
    centered_checks = []
    for original, changed in zip(traces, translated):
        key = f"yhat_R{original['i_reg']}_{original['tkr']}"
        y_delta = 1.25 if original['tkr'] == 'A' else -.75
        centered_checks.append({'regime': original['i_reg'], 'ticker': original['tkr'],
                                'Xc_mean': original['X'].mean(axis=0), 'yc_mean': original['y'].mean(),
                                'prediction_shift': shifted['yhat_row'][key] - state['yhat_row'][key],
                                'expected_shift': y_delta,
                                'passed': close(original['X'].mean(axis=0), np.zeros(3)) and
                                close(original['y'].mean(), 0) and close(original['X'], changed['X']) and
                                close(original['beta_best'], changed['beta_best']) and
                                close(shifted['yhat_row'][key] - state['yhat_row'][key], y_delta)})
    add(3, 'Step2 부분집합 중심화와 절편 복원은 평행이동에 일관적',
        all(v['passed'] for v in centered_checks), 'X 이동에 계수 불변; y 이동만큼 예측 이동', centered_checks,
        refs=[loop_ref], inputs={**loop_input, 'X_translation': [7., -3., 2.], 'y_translation': [1.25, -.75]},
        oracle='절편이 있는 ridge의 평행이동 불변성; 실제 함수 입력의 Xc/yc 평균과 원 루프 최종 예측을 관측.')

    # Fixed seed 4 is a small, reproducible counterexample, not a market backtest.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 3)) + [2., -1., .3]
    y = rng.normal(size=8) + 2.
    grid_node = next(n for n in tree_of(repo, S2).body if assigned(n, 'lambdas'))
    exec_nodes(repo, S2, [grid_node], source_scope)
    lambdas = source_scope['lambdas']
    x_query = np.array([-.5, 1., 2.])
    cv_input = fixture('loocv_counterexample', {'X': X, 'y': y, 'lambdas': lambdas, 'x_query': x_query}, 4)
    Xc, yc, query_centered = X-X.mean(axis=0), y-y.mean(), x_query-X.mean(axis=0)
    B, hdiag = precompute(Xc, lambdas)
    original_mse = []
    for k, lam in enumerate(lambdas):
        _, observation = observe_selector(select, lambda: select(Xc, yc, [lam], [B[k]], [hdiag[k]], query_centered))
        original_mse.append(observation[0]['best_mse'])
    oracle_mse, held_out_predictions = loo_oracle(X, y, lambdas)
    corrected_mse = [float(np.mean(((yc - Xc @ (b @ yc)) / (1 - h - 1/len(y))) ** 2))
                     for b, h in zip(B, hdiag)]
    add(4, 'Step2 analytic LOOCV는 절편 포함 한 행씩 재적합과 불일치', close(original_mse, oracle_mse),
        {'refit_mse': oracle_mse}, {'source_mse': original_mse, 'intercept_corrected_mse': corrected_mse,
                                  'corrected_matches_refits': close(corrected_mse, oracle_mse),
                                  'max_abs_mse_error': float(np.max(np.abs(np.array(original_mse)-oracle_mse))),
                                  'missing_hat_diagonal': 1/len(y), 'held_out_predictions': held_out_predictions},
        refs=ridge_refs + [loop_ref], inputs=cv_input,
        oracle='각 fold에서 원 X/y 중 한 행을 제외하고 비벌점 절편과 계수를 재적합. H 전체 대각은 1/n + Xc B.',
        classification='confirmed_cv_error',
        notes=['원문의 λ/CV 명세 부족과 별개로 원 코드가 주장하는 fast LOOCV 자체의 수치 불일치다.',
               'oracle은 고정 PCA 좌표에서 회귀만 재적합한다. PCA·군집의 fold별 재적합 또는 시간순 CV를 주장하지 않는다.'])
    chosen, _ = observe_selector(select, lambda: select(Xc, yc, lambdas, B, hdiag, query_centered))
    source_lambda, source_beta, source_pred_centered = chosen
    oracle_lambda = float(lambdas[int(np.argmin(oracle_mse))])
    oracle_intercept, oracle_beta = ridge_oracle(X, y, oracle_lambda)
    expected_pred = oracle_intercept + float(x_query @ oracle_beta)
    observed_pred = source_pred_centered + float(y.mean())
    add(5, '절편 누락은 선택 λ와 예측을 바꾸는 합성 반례를 만든다', close(source_lambda, oracle_lambda),
        {'lambda': oracle_lambda, 'prediction': expected_pred},
        {'lambda': source_lambda, 'prediction': observed_pred, 'prediction_difference': observed_pred-expected_pred,
         'source_selected_refit_mse': float(oracle_mse[int(np.where(lambdas == source_lambda)[0][0])]),
         'oracle_selected_refit_mse': float(oracle_mse.min())},
        refs=ridge_refs, inputs=cv_input, oracle='독립 재적합 MSE가 가장 작은 원본 후보 λ 선택; 동률은 후보 순서.',
        classification='confirmed_cv_error', notes=['합성 8행 반례이며 실제 ETF 예측·성과 차이 추정이 아니다.'])

    counts = []
    for regime in range(1, 6):
        observed = next(item for item in traces if item['i_reg'] == regime)
        original_count = int(np.sum(hard[:-1] == regime))
        expected_tau = np.where(hard[:-1] == regime)[0] if original_count >= 6 else np.arange(47)
        counts.append({'regime': regime, 'reported_n_Ri': state['cnt_row'][f'n_R{regime}'],
                       'actual_fit_rows': len(observed['tau_idx']), 'actual_tau_indices': observed['tau_idx'],
                       'passed': np.array_equal(observed['tau_idx'], expected_tau) and
                       state['cnt_row'][f'n_R{regime}'] == original_count})
    duplicate_fallback_forecasts = all(close(state['yhat_row'][f'yhat_R1_{t}'], state['yhat_row'][f'yhat_R2_{t}']) for t in R.columns)
    add(6, '원 레짐 표본 0·5개는 전체 47행 fallback, 6개부터 조건부 학습',
        all(v['passed'] for v in counts) and duplicate_fallback_forecasts,
        {'raw_regime_counts': [0, 5, 6, 10, 26], 'effective_fit_counts': [47, 47, 6, 10, 26]},
        {'counts': counts, 'R1_R2_identical_predictions': duplicate_fallback_forecasts},
        refs=[loop_ref], inputs=loop_input, oracle='hard[:-1] 레짐 집합과 원 함수 입력 tau_idx를 독립적으로 대조.',
        classification='implementation_choice',
        notes=['<6 기준과 global fallback은 원문 미명세 추가 정책이다. n_Ri CSV는 fallback 전 개수이며 실제 적합 표본 수와 다르다.',
               'fallback 예측을 같은 R 번호로 저장하므로 해당 열은 그 레짐에만 조건부인 추정이 아니다.'])

    future = R.copy(); future.iloc[48:] += 100.
    future_state, _, _ = original_regime_loop(repo, scores, hard, future, fixed)
    latest = R.copy(); latest.iloc[47] += .5
    latest_state, _, _ = original_regime_loop(repo, scores, hard, latest, fixed)
    future_error = max(abs(state['yhat_row'][k]-future_state['yhat_row'][k]) for k in state['yhat_row'])
    latest_error = max(abs(state['yhat_row'][k]-latest_state['yhat_row'][k]) for k in state['yhat_row'])
    target_checks = [close(item['y_vec'], R.iloc[item['tau_idx']+1][item['tkr']].to_numpy()) for item in traces]
    add(7, '원 Step2 학습 목표 상한은 R_t이며 R_(t+1) 이후 변경에는 불변',
        all(target_checks) and close(future_error, 0) and latest_error > ATOL,
        {'target_rows': 'tau+1, 최대 인덱스 47 (=t)', 'future_prediction_change': 0},
        {'future_prediction_change': future_error, 'latest_R_t_prediction_change': latest_error,
         'target_mapping_checks': target_checks, 't_label': R.index[47], 'latest_target_end_price_label': R.index[48]},
        refs=[loop_ref], inputs={**loop_input, 'future_mutation': 'R.iloc[48:]+=100', 'latest_mutation': 'R.iloc[47]+=.5'},
        oracle='행 인덱스+1 목표 연결 직접 대조 및 목표 행 상한 밖/안 대조 변경 실험.',
        classification='timing_contract',
        notes=['R_t=P(t+1)/P(t)-1 이므로 R_t는 P(t+1) 관측 후 완성된다. 결정을 P(t+1) 이후로 읽으면 shift 자체는 누출 반례가 아니다.',
               'scores/hard 고정 회귀 검사다. 실제 공개일·48행 달력·전처리 미래 변경은 data_timing 증거를 소비한다.'])

    # Execute the actual fallback model constructor, fit and predict statements.
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    function = next(n for n in tree_of(repo, S3).body if isinstance(n, ast.FunctionDef) and n.name == 'load_step2_or_autogen')
    alpha_nodes = [n for n in function.body if assigned(n, 'ALPHAS') or assigned(n, 'cv')]
    ticker_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'tkr')
    scope = {'np': np, 'pd': pd, 'RidgeCV': RidgeCV, 'LeaveOneOut': LeaveOneOut,
             'R': pd.DataFrame({'A': y}), 'y_idx': np.arange(8), 'Xi': X, 'x_t': x_query.reshape(1, -1),
             'TICKERS': ['A'], 'i_reg': 1, 'row': {}}
    exec_nodes(repo, S3, alpha_nodes + [ticker_loop], scope)
    sk_mse, _ = loo_oracle(X, y, scope['ALPHAS'])
    sk_lambda = float(scope['ALPHAS'][int(np.argmin(sk_mse))])
    sk_intercept, sk_beta = ridge_oracle(X, y, sk_lambda)
    model = scope['model']
    fallback_prediction = scope['row']['yhat_R1_A']
    fallback_expected = sk_intercept + float(x_query @ sk_beta)
    add(8, 'Step3 fallback RidgeCV는 명시적 음의 MSE scoring과 절편 LOO를 사용',
        model.scoring == 'neg_mean_squared_error' and close(model.alpha_, sk_lambda) and
        close(-model.best_score_, sk_mse.min()) and close(model.coef_, sk_beta) and close(model.intercept_, sk_intercept) and
        close(fallback_prediction, fallback_expected),
        {'lambda': sk_lambda, 'mse': float(sk_mse.min()), 'intercept': sk_intercept, 'beta': sk_beta,
         'prediction': fallback_expected},
        {'scoring': model.scoring, 'fit_intercept': model.fit_intercept, 'lambda': model.alpha_,
         'mse': -model.best_score_, 'intercept': model.intercept_, 'beta': model.coef_, 'alphas': scope['ALPHAS'],
         'prediction': fallback_prediction},
        refs=[ref_range(repo, S3, alpha_nodes[0].lineno, alpha_nodes[-1].end_lineno, 'fallback_cv_setup_AST'),
              ref_range(repo, S3, ticker_loop.lineno, ticker_loop.end_lineno, 'fallback_ridge_fit_AST')],
        inputs=fixture('step3_fallback_regression', {'X': X, 'y': y, 'x_query': x_query,
                                                   'lambdas': scope['ALPHAS']}, 4),
        oracle='원본 모델 생성/fit/predict AST 실행 후 동일 후보에 대한 독립 LOOCV 재적합과 대조.',
        notes=['R2 기본 scoring 오류가 아니다. Step2와 후보 λ 구간도 다르므로 두 경로의 결과 동일성을 가정하지 않는다.'])

    dates = pd.date_range('2020-01-01', periods=4, freq='MS')
    step1 = pd.DataFrame([{'sasdate_t': dates[j], 'sasdate_tp1': dates[j+1],
                           **{f'ptp1_R{k}': p[k] for k in range(6)}} for j in range(3)])
    step2 = pd.DataFrame([{'sasdate_t': dates[j], 'sasdate_tp1': dates[j+1],
                           **{f'yhat_R{i+1}_{t}': forecasts[i, k] for i in range(5) for k, t in enumerate(['A', 'B'])},
                           'yhat_R0_A': 1000., 'yhat_R0_B': -1000.} for j in range(3)])
    agg_input = fixture('aggregation', {'step1': step1.to_dict('records'), 'step2': step2.to_dict('records')})
    aggregated, agg_ref = original_aggregation(repo, step1, step2)
    altered_R0 = step2.copy(); altered_R0['yhat_R0_A'] = -1e12; altered_R0['yhat_R0_B'] = 1e12
    altered_agg, _ = original_aggregation(repo, step1, altered_R0)
    expected = equation14(p, forecasts)
    add(9, '정상 Step3 집계는 인쇄 식14의 R1~R5 합이며 R0 예측을 제외',
        close(aggregated[['A', 'B']].to_numpy(), np.tile(expected, (3, 1))) and
        close(aggregated[['A', 'B']].to_numpy(), altered_agg[['A', 'B']].to_numpy()),
        expected, {'values': aggregated[['A', 'B']].to_numpy(), 'normal_probability_mass': p[1:].sum(),
                   'R0_forecast_mutation_effect': (aggregated[['A', 'B']]-altered_agg[['A', 'B']]).to_numpy()},
        refs=[agg_ref], inputs=agg_input, oracle='각 자산에 scalar Σ(i=1..5) p_i*y_i를 계산. R0 예측 열만 대폭 바꾼 대조 실행.',
        notes=['R0 제외는 인쇄식 준수다. 논문의 위기 활용 의미와 식14 범위 불일치는 methodology AMB04에 남긴다.'])
    unnormalized_weights, _ = lo(aggregated.loc[0, ['A', 'B']].to_numpy(float), 2)
    renormalized_weights, _ = lo(aggregated.loc[0, ['A', 'B']].to_numpy(float) / p[1:].sum(), 2)
    add(10, '일반 레짐 확률 재정규화의 양의 공통 배율은 lo 비중에서 상쇄',
        close(unnormalized_weights, renormalized_weights) and close(unnormalized_weights, [23/52, 29/52]),
        [23/52, 29/52], {'without_renormalization': unnormalized_weights, 'with_renormalization': renormalized_weights},
        refs=[agg_ref, lo_ref], inputs=agg_input,
        oracle='원 집계와 원 lo 함수의 연결 실행. 0<Σp_i이고 같은 양의 공통배율이면 식18의 분자·분모에서 상쇄.',
        classification='paper_ambiguity',
        notes=['원문은 재정규화를 명시하지 않는다. 재정규화 부재만으로 lo 배분 오류를 주장하지 않는다.'])

    crisis = step1.copy(); crisis[[f'ptp1_R{k}' for k in range(1, 6)]] = 0.; crisis['ptp1_R0'] = 1.
    crisis_agg, _ = original_aggregation(repo, crisis, step2)
    crisis_expected = equation14(np.array([1., 0., 0., 0., 0., 0.]), forecasts)
    add(11, 'R0 확률 1인 경계에서 Step3 균등 fallback은 인쇄 식14의 0 예측과 다름',
        close(crisis_agg[['A', 'B']].to_numpy(), np.tile(crisis_expected, (3, 1))),
        crisis_expected, {'values': crisis_agg[['A', 'B']].to_numpy(), 'fallback_normal_probs': [.2]*5},
        refs=[agg_ref], inputs={**agg_input, 'probability_override': [1., 0., 0., 0., 0., 0.]},
        oracle='식14 모든 일반 레짐 p_i=0이면 가중합은 0. 실제 AST는 일반 레짐 균등확률로 대체한다.',
        classification='equation14_boundary_deviation',
        notes=['명시된 인쇄식과의 경계 불일치다. 이 경계가 실제 ETF 실행에서 발생했다는 증거는 없다.'])

    missing = step1.drop(index=1)
    missing_agg, _ = original_aggregation(repo, missing, step2)
    add(12, 'Step3 inner merge는 한쪽 날짜 누락을 조용히 삭제', len(missing_agg) == len(step2),
        {'forecast_date_rows': len(step2), 'unmatched_dates_should_be_reported': [dates[1]]},
        {'merged_rows': len(missing_agg), 'lost_dates': sorted(set(step2.sasdate_t)-set(missing_agg.sasdate_t)),
         'exception_or_warning': None}, refs=[agg_ref],
        inputs={**agg_input, 'mutation': 'step1.drop(index=1)'},
        oracle='forecast 날짜 3개 중 반대편 키가 없는 날짜를 집합 차이로 계산; 출력 행 개수와 비교.',
        classification='join_integrity',
        notes=['일대일 월별 파이프라인의 완전성 진단이다. inner join 선택 자체에 논문이 유일한 오류처리 정책을 정하지는 않는다.'])
    duplicate1 = pd.concat([step1, step1.iloc[[0]], step1.iloc[[0]]], ignore_index=True)
    duplicate2 = pd.concat([step2, step2.iloc[[0]]], ignore_index=True)
    duplicate_agg, _ = original_aggregation(repo, duplicate1, duplicate2)
    per_key = duplicate_agg.groupby(['sasdate_t', 'sasdate_tp1']).size()
    add(13, 'Step3 중복 날짜 merge는 3×2 카테시안 행 증식을 허용', bool((per_key <= 1).all()),
        {'maximum_rows_per_date_key': 1}, {'merged_rows': len(duplicate_agg),
                                          'rows_per_date_key': {str(k): int(v) for k, v in per_key.items()},
                                          'exception_or_warning': None}, refs=[agg_ref],
        inputs={**agg_input, 'mutation': 'first key occurs 3 times in step1 and 2 times in step2'},
        oracle='동일 키 결합의 곱셈 cardinality: 첫 키 3×2=6, 나머지 1+1, 합계 8.',
        classification='join_integrity', notes=['실제 공급 CSV의 중복 발생률을 주장하지 않는 주입 반례다.'])

    missing_forecast = step2.drop(columns=['yhat_R3_A'])
    incomplete_agg, _ = original_aggregation(repo, step1, missing_forecast)
    zero_prob = step1.copy(); zero_prob['ptp1_R3'] = 0.
    incomplete_zero, _ = original_aggregation(repo, zero_prob, missing_forecast)
    add(14, '누락 레짐 예측은 해당 자산 NaN으로 전달되며 확률 0이어도 유지',
        bool(incomplete_agg.A.isna().all() and incomplete_zero.A.isna().all()) and close(incomplete_agg.B, aggregated.B),
        {'A': 'NaN', 'B': '원 값 유지'}, {'A': incomplete_agg.A.to_numpy(), 'B': incomplete_agg.B.to_numpy(),
                                             'A_when_missing_regime_prob_zero': incomplete_zero.A.to_numpy()},
        refs=[agg_ref], inputs={**agg_input, 'mutation': 'drop yhat_R3_A; second run ptp1_R3=0'},
        oracle='원 row.get(...,NaN)와 np.isnan(...).any() 경로 직접 관측; 다른 자산과 대조.',
        classification='implementation_choice', notes=['결측 예측 정책은 논문 미명세다. 누락 날짜 처리와 누락 셀 처리를 구별한다.'])

    finite_cases = {'positive_top_l': [4., 3., 2., 1., -.5], 'positive_shortage': [2., -1., 0., 1., -3.],
                    'all_nonpositive': [-2., 0., -1., -4., 0.], 'boundary_ties': [3., 2., 2., 2., -1.]}
    finite_details = []
    for label, values in finite_cases.items():
        values = np.array(values)
        for l in [2, 3, 4]:
            weights, meta = lo(values, l)
            selected = np.where(weights > 0)[0]
            k = min(l, int((values > 0).sum()))
            expected_exposure = 1. if k else 0.
            # Identity of equally valued boundary assets is not specified in the paper.
            selected_values = sorted(values[selected].tolist(), reverse=True)
            target_values = sorted([float(v) for v in values if v > 0], reverse=True)[:l]
            tie_valid = close(selected_values, target_values)
            independent_weights = lo_oracle(values, l)
            weight_valid = (close(weights, independent_weights) if label != 'boundary_ties' else
                            tie_valid and all(close(weights[i], values[i]/sum(target_values)) for i in selected))
            passed = (bool(np.isfinite(weights).all() and (weights >= 0).all()) and len(selected) == k and
                      close(weights.sum(), expected_exposure) and close(np.abs(weights).sum(), expected_exposure) and
                      meta['n_selected'] == k and weight_valid)
            finite_details.append({'case': label, 'l': l, 'weights': weights, 'metadata': meta,
                                   'net_exposure': weights.sum(), 'gross_exposure': np.abs(weights).sum(),
                                   'selected_values': selected_values, 'passed': passed})
    add(15, '원 lo 함수: 양수 상위 l·양수 부족·전부 비양수·동률·노출 검사',
        all(v['passed'] for v in finite_details), '유효 양수 중 최대 l개; 노출 1, 양수 없으면 0', finite_details,
        refs=[lo_ref], inputs=fixture('lo_finite', {'cases': finite_cases, 'l_values': [2, 3, 4]}),
        oracle='독립 scalar 정렬/비율과 비교; 동률은 특정 종목 ID 대신 선택값·개수·노출을 검증.',
        notes=['모두 비양수일 때 현금(0 비중)은 원문 분모 0에 대한 구현 선택이다. 동률의 종목 선택 순서는 원문 미명세다.'])

    nonfinite_cases = {'nan_with_positive': [np.nan, 2., 1., -1.],
                       'positive_inf': [np.inf, 2., 1., -1.],
                       'negative_inf': [-np.inf, 2., 1., -1.],
                       'all_unusable': [np.nan, -np.inf, 0., -1.]}
    nonfinite_details = []
    for label, values in nonfinite_cases.items():
        values = np.array(values)
        weights, meta = lo(values, 2)
        expected_weights = [0., 0., 0., 0.] if label in ['positive_inf', 'all_unusable'] else [0., 2/3, 1/3, 0.]
        nonfinite_details.append({'case': label, 'weights': weights, 'metadata': meta,
                                  'actual_nonzero_weights': int(np.count_nonzero(weights)),
                                  'net_exposure': float(weights.sum()), 'gross_exposure': float(np.abs(weights).sum()),
                                  'passed': close(weights, expected_weights) and bool(np.isfinite(weights).all())})
    add(16, 'lo NaN/Inf 경계: NaN·음의 Inf 제외, 양의 Inf 포함 시 전체 현금',
        all(v['passed'] for v in nonfinite_details), '원 함수의 명시적 분기 관측값 및 유한 출력', nonfinite_details,
        refs=[lo_ref], inputs=fixture('lo_nonfinite', {'cases': nonfinite_cases, 'l': 2}),
        oracle='NaN>0 및 -Inf>0은 거짓; +Inf가 선택되면 S_l=Inf이므로 원 함수 0 비중 반환.',
        classification='undefined_input_policy',
        notes=['NaN/Inf 입력의 경제적 처리 규칙은 원문 미정이다. PASS는 관측한 정책과 유한 노출의 확인이며 권장 정책이라는 뜻이 아니다.',
               '+Inf 반례에서 n_selected=2이나 실제 비중이 양수인 종목은 0개다. S_l=Inf가 남아 있어 출력 metadata까지 유한한 것은 아니다.'])

    (destination / 'fixtures.json').write_text(json.dumps(jsonable(fixtures), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_results(output, 'forecasting', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.repo / 'reports' / 'paper_audit' / 'evidence'
    destination = output / 'forecasting'
    destination.mkdir(parents=True, exist_ok=True)
    stdout, stderr = io.StringIO(), io.StringIO()
    exit_code, counts = 0, {}
    started = datetime.now(timezone.utc).isoformat()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            results = run(args.repo, output)
            counts = dict(Counter(row['status'] for row in results))
            print('Forecasting diagnostics completed:', counts)
            print('FAIL records are retained diagnostic findings; execution completed successfully.')
        except Exception:
            exit_code = 1
            traceback.print_exc()
    import sklearn
    execution = {'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                 'command': [sys.executable, *getattr(sys, 'orig_argv', [sys.executable, *sys.argv])[1:]],
                 'interpreter_reported_orig_argv': getattr(sys, 'orig_argv', None),
                 'cwd': str(Path.cwd()), 'exit_code': exit_code,
                 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'counts': counts,
                 'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                 'source_sha256': {name: sha256(args.repo / name) for name in [S2, S3, S4]},
                 'audit_sha256': sha256(Path(__file__)),
                 'artifact_sha256': {name: sha256(destination / name) for name in ['fixtures.json', 'results.json'] if (destination / name).exists()},
                 'semantics': 'exit 0 means diagnostic run completed; FAIL does not make the runner fail.'}
    (destination / 'execution.json').write_text(json.dumps(execution, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(stdout.getvalue(), end='')
    print(stderr.getvalue(), end='', file=sys.stderr)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
