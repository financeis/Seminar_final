"""Read-only data/calendar diagnostics using unchanged original AST statements.

FAIL means an observed research-code counterexample, not a broken audit runner.
No research script is imported; no downloader or original output statement runs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import TICKERS, jsonable, load_functions, record, resolve_paper, sha256, source_ref, write_results
except ImportError:
    from common import TICKERS, jsonable, load_functions, record, resolve_paper, sha256, source_ref, write_results


FILES = ['Section3.py', 'Section5_step1.py', 'Section5_step2.py',
         'Section5_step3.py', 'Section5_step4.py']


def tree_of(repo, filename):
    return ast.parse((repo / filename).read_text(encoding='utf-8-sig'))


def ref_range(repo, filename, first, last, label=None):
    if filename.endswith('.py'):
        ref = source_ref(repo, filename)
    else:
        ref = {'path': filename, 'sha256': sha256(repo / filename)}
    ref.update(symbol=label, line_start=first, line_end=last)
    return ref


def assigned(node, name):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t))


def exec_nodes(repo, filename, nodes, scope):
    """The exact original nodes execute in memory, retaining original line numbers."""
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(repo / filename), 'exec'), scope)


def transform_function(repo, filename):
    if filename in ['Section5_step1.py', 'Section5_step2.py', 'etc_regime_stats_transformed.py']:
        return load_functions(repo, filename, ['transform_series'])['transform_series']
    node = next(n for n in ast.walk(tree_of(repo, filename))
                if isinstance(n, ast.FunctionDef) and n.name == 'transform_series')
    scope = {'pd': pd, 'np': np}
    exec_nodes(repo, filename, [node], scope)
    return scope['transform_series']


def prep_nodes(repo, filename):
    tree = tree_of(repo, filename)
    if filename == 'Section3.py':
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_data_if_needed').body
        out_name = 'df_cc_local'
    elif filename == 'Section5_step3.py':
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'load_step2_or_autogen').body
        out_name = 'df_cc'
    elif filename == 'Section5_step4.py':
        body = next(n for n in tree.body if isinstance(n, ast.If)
                    and any(assigned(k, 'start_date') for k in n.body)).body
        out_name = 'df_cc'
    else:
        body, out_name = tree.body, 'df_cc'
    first = next(i for i, n in enumerate(body) if assigned(n, 'start_date'))
    last = next(i for i, n in enumerate(body) if i >= first and assigned(n, out_name))
    return body[first:last + 1], out_name


def prepare(repo, filename, raw, codes):
    nodes, out_name = prep_nodes(repo, filename)
    scope = {'np': np, 'pd': pd, 'df_raw': raw.copy(deep=True),
             'raw': raw.copy(deep=True), 'tcode_map': dict(codes)}
    cut = next(i for i, n in enumerate(nodes) if assigned(n, 'protected'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        exec_nodes(repo, filename, nodes[:cut], scope)
        transformed = scope['df_t'].copy(deep=True)
        exec_nodes(repo, filename, nodes[cut:], scope)
    return {'panel': scope[out_name], 'transformed': transformed,
            'kept': [c for c in scope[out_name].columns if c != 'sasdate'],
            'candidate': [c for c in transformed.columns if c != 'sasdate'],
            'protected': sorted(scope['protected']), 'group6': scope['group6_all'],
            'window_raw': scope['df'], 'scope': scope,
            'ref': ref_range(repo, filename, nodes[0].lineno, nodes[-1].end_lineno, 'macro_preparation_AST')}


def oracle_transform(values, code):
    """Scalar formula oracle: no pandas diff/shift/pct_change or original helpers."""
    values = [float(x) for x in values]
    result = np.full(len(values), np.nan)
    for t, x in enumerate(values):
        def get(offset, log=False):
            if t < offset:
                return np.nan
            z = values[t - offset]
            return np.log(z) if log and z > 0 else (np.nan if log else z)
        with np.errstate(all='ignore'):
            if code == 1: result[t] = x
            if code == 2: result[t] = x - get(1)
            if code == 3: result[t] = x - 2 * get(1) + get(2)
            if code == 4: result[t] = get(0, True)
            if code == 5: result[t] = get(0, True) - get(1, True)
            if code == 6: result[t] = get(0, True) - 2 * get(1, True) + get(2, True)
            if code == 7: result[t] = np.divide(x, get(1)) - np.divide(get(1), get(2))
    return result


def same(a, b, tol=1e-11):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=tol, equal_nan=True)


def month_delta(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def self_check():
    x = [2., 4., 8., 16.]
    assert same(oracle_transform(x, 2), [np.nan, 2, 4, 8])
    assert same(oracle_transform(x, 3), [np.nan, np.nan, 2, 4])
    assert same(oracle_transform(x, 5), [np.nan, np.log(2), np.log(2), np.log(2)])
    assert same(oracle_transform(x, 6), [np.nan, np.nan, 0, 0])
    assert same(oracle_transform(x, 7), [np.nan, np.nan, 0, 0])
    assert not same(oracle_transform([2, 4, 12], 7), [np.nan, np.nan, 0])
    assert np.isnan(oracle_transform([2, 0, -2], 4)[1:]).all()
    assert month_delta(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01')) == 3
    # Known first-price oracle: 100 -> 110 -> 99 gives +10%, -10%, not a trailing return.
    assert same([110 / 100 - 1, 99 / 110 - 1], [.1, -.1])
    return {'assertions': 9, 'passed': True,
            'oracle': 'scalar t-code formulas; literal arithmetic returns; calendar year/month arithmetic'}


def panel_description(prep):
    panel, transformed, raw = prep['panel'], prep['transformed'], prep['window_raw']
    kept = prep['kept']
    dates = pd.DatetimeIndex(panel.sasdate)
    missing = raw[~raw.sasdate.isin(dates)]
    reasons = []
    for i, row in missing.iterrows():
        j = transformed.index[transformed.sasdate == row.sasdate][0]
        bad = [c for c in kept if pd.isna(transformed.loc[j, c])]
        reasons.append({'month': row.sasdate, 'missing_transformed_variables': bad})
    quality = []
    for c in prep['candidate']:
        y = transformed[c]
        a, b = y.first_valid_index(), y.last_valid_index()
        quality.append({'variable': c, 'retained': c in kept, 'protected': c in prep['protected'],
                        'first_valid': None if a is None else transformed.loc[a, 'sasdate'],
                        'last_valid': None if b is None else transformed.loc[b, 'sasdate'],
                        'internal_nan_fraction': 1. if a is None or b is None else float(y.loc[a:b].isna().mean()),
                        'nan_count': int(y.isna().sum()), 'inf_count': int(np.isinf(y.to_numpy(float)).sum()),
                        'nonpositive_raw_count': int((raw[c] <= 0).sum())})
    spans = []
    for end in range(47, len(dates) - 1):
        first = end - 47
        spans.append({'decision_label': dates[end], 'first_row': dates[first],
                      'rows': 48, 'calendar_months_inclusive': month_delta(dates[first], dates[end]) + 1,
                      'next_retained_label': dates[end + 1],
                      'forecast_calendar_step': month_delta(dates[end], dates[end + 1])})
    return {'raw_rows': len(raw), 'raw_start': raw.sasdate.min(), 'raw_end': raw.sasdate.max(),
            'candidate_variables': prep['candidate'], 'retained_variables': kept,
            'removed_gap_variables': [c for c in prep['candidate'] if c not in kept],
            'removed_group_variables': [c for c in raw.columns if c != 'sasdate' and c not in prep['candidate']],
            'retained_group6': [c for c in kept if c in prep['group6']],
            'retained_rows': len(dates), 'retained_start': dates.min(), 'retained_end': dates.max(),
            'retained_dates': dates.tolist(), 'deleted_months': reasons, 'variable_quality': quality,
            'rolling_windows': spans, 'source_refs': [prep['ref']]}


def step1_first_decision(repo, panel):
    filename = 'Section5_step1.py'
    scope = load_functions(repo, filename)
    loop = next(n for n in tree_of(repo, filename).body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'end_idx')
    scope.update(macro_dates=pd.to_datetime(panel.sasdate.values),
                 X_full=panel.drop(columns='sasdate').to_numpy(float),
                 W=48, r=5, records=[], E_mats={}, prevC2=None, end_idx=47)
    exec_nodes(repo, filename, loop.body, scope)
    return {'record': scope['records'][0], 'p_t': scope['p_t'], 'p_tp1': scope['p_tp1'],
            'window_rows': len(scope['idx']), 'zscore_rows': len(scope['Zw']),
            'pca_rows': len(scope['scores']),
            'ref': ref_range(repo, filename, loop.lineno, loop.end_lineno, 'first_rolling_iteration_AST')}


def step2_first_decision(repo, panel, returns):
    filename = 'Section5_step2.py'
    scope = load_functions(repo, filename)
    loop = next(n for n in tree_of(repo, filename).body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 't_end')
    scope.update(X_full=panel.drop(columns='sasdate').to_numpy(float), W=48, r=5,
                 dates=list(pd.to_datetime(panel.sasdate)), R=returns, TICKERS=list(returns.columns),
                 lambdas=np.logspace(-4, 2, 8), fore_rows=[], lam_rows=[], cnt_rows=[],
                 prevC2=None, t_end=47)
    exec_nodes(repo, filename, loop.body, scope)
    return {'forecasts': scope['fore_rows'][0], 'lambdas': scope['lam_rows'][0],
            'counts': scope['cnt_rows'][0],
            'window_rows': len(scope['idx']), 'zscore_rows': len(scope['Zw']),
            'pca_rows': len(scope['scores']), 'lambda_candidate_count': len(scope['lambdas']),
            'final_regime_train_rows': len(scope['tau_idx']),
            'ref': ref_range(repo, filename, loop.lineno, loop.end_lineno, 'first_rolling_iteration_AST')}


def synthetic_prices(dates):
    rows = []
    for j, ticker in enumerate(TICKERS):
        for i, month in enumerate(dates):
            # Business-day adjustment is a diagnostic calendar, not a historical exchange calendar.
            day = pd.offsets.BDay().rollforward(month)
            price = 100 * np.exp(.003 * i + .06 * np.sin(i / (3. + j / 8)) + .004 * j * np.cos(i / 5))
            rows.append({'date': day, 'ticker': ticker, 'adj_close': price})
    return pd.DataFrame(rows)


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(repo: Path, output: Path) -> list[dict]:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    dest = output / 'data_timing'
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    check = self_check()  # Audit machinery is checked before calling original implementations.
    results.append(record('DATA-SELF', '독립 oracle 자체 검사', check['passed'], '9 literal checks pass', check,
                          oracle=check['oracle'], inputs='Explicit positive, zero, negative and calendar examples'))
    raw = pd.read_csv(repo / 'FRED-MD_2024m12.csv', skiprows=[1])
    raw.sasdate = pd.to_datetime(raw.sasdate)
    trow = pd.read_csv(repo / 'FRED-MD_2024m12.csv', nrows=1)
    codes = {c: int(trow[c].iloc[0]) for c in trow if c != 'sasdate'}
    data_ref = ref_range(repo, 'FRED-MD_2024m12.csv', 1, len(raw) + 2, 'supplied_2024m12_named_snapshot')
    paper_path = resolve_paper(repo)
    paper_ref = {'path': str(paper_path), 'sha256': sha256(paper_path),
                 'pages': [7, 16], 'note': 'methodology.md records full original-paper review'}

    # Transform equivalence covers every duplicated implementation, including helper/fallback branches.
    transform_details = []
    vectors = {'positive': [2., 3., 7., 11., 17., 23.],
               'edge': [2., 0., -2., np.nan, 4., 8., 16.],
               'internal_missing': [2., 4., np.nan, 8., 16., 32.],
               'constant': [3.] * 6}
    for filename in FILES + ['etc_regime_stats_transformed.py']:
        fn = transform_function(repo, filename)
        for code in range(1, 8):
            errors = []
            for name, values in vectors.items():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    actual = fn(pd.Series(values, dtype=float), code).to_numpy(float)
                expected = oracle_transform(values, code)
                if not same(actual, expected):
                    errors.append({'case': name, 'expected': expected, 'observed': actual})
            transform_details.append({'file': filename, 'tcode': code, 'mismatches': errors})
    mismatch = [x for x in transform_details if x['mismatches']]
    refs = []
    for filename in FILES + ['etc_regime_stats_transformed.py']:
        first = transform_function(repo, filename).__code__.co_firstlineno
        node = next(n for n in ast.walk(tree_of(repo, filename))
                    if isinstance(n, ast.FunctionDef) and n.lineno == first and n.name == 'transform_series')
        refs.append(ref_range(repo, filename, node.lineno, node.end_lineno, 'transform_series'))
    refs.append({'path': 'FRED-MD_updated_appendix.pdf', 'sha256': sha256(repo / 'FRED-MD_updated_appendix.pdf'),
                 'pages': [1], 'note': 'Supplementary t-code appendix; not the designated research paper'})
    results.append(record('DATA-TCODE', 't-code 1~7 원 구현과 부록 산식', not mismatch,
                          '6 implementations × 7 codes × 4 cases agree with scalar formulas',
                          {'comparisons': len(transform_details) * len(vectors), 'mismatches': mismatch},
                          source_refs=refs, inputs=vectors, oracle='FRED-MD appendix t-code formulas in methodology.md; scalar oracle_transform',
                          tolerance=1e-11, notes=['논문 충실도: 정상 양수뿐 아니라 내부 결측 전파도 검사.',
                                                'pct_change 기본 fill 정책은 pandas 버전에 따라 달라질 수 있으며 현재 실행 환경의 결과다.']))
    write_json(dest / 'transforms.json', transform_details)

    preps = {f: prepare(repo, f, raw, codes) for f in FILES}
    baseline = preps['Section5_step1.py']
    desc = panel_description(baseline)
    write_json(dest / 'panel.json', desc)
    for filename, p in preps.items():
        results.append(record('DATA-PREP-' + filename.replace('.py', '').replace('Section', ''),
                              filename + ' 전처리 일치', p['panel'].equals(baseline['panel']),
                              {'reference': 'Section5_step1.py actual AST', 'shape': baseline['panel'].shape},
                              {'shape': p['panel'].shape, 'first': p['panel'].sasdate.min(), 'last': p['panel'].sasdate.max()},
                              source_refs=[p['ref'], baseline['ref'], data_ref],
                              inputs='Repository-supplied raw CSV and t-code row',
                              oracle='Independent AST executions of each original preprocessing branch; equality is cross-implementation consistency, not paper truth'))
    results.append(record('DATA-GROUP6', '논문 그룹6 제외와 실행 변수 집합', not desc['retained_group6'],
                          'Paper p.7 excludes group 6 from regime input',
                          {'raw_variables': len(raw.columns) - 1, 'retained': len(desc['retained_variables']),
                           'group6_in_model': desc['retained_group6'], 'removed_gap': desc['removed_gap_variables']},
                          source_refs=[paper_ref, baseline['ref'], data_ref], oracle='Set intersection of actual retained columns with original group6_all',
                          notes=['논문 명시 의무. must_keep가 그룹6 제외보다 우선한다. 실제 컬럼 전체는 panel.json에 저장.']))
    gaps = [w for w in desc['rolling_windows'] if w['calendar_months_inclusive'] != 48]
    jumps = [w for w in desc['rolling_windows'] if w['forecast_calendar_step'] != 1]
    results.append(record('DATA-CALENDAR', '48행 창과 연속 48개월·다음 달 구분', not gaps and not jumps,
                          'Paper Section 6: 48 consecutive months; one calendar-month forecast step',
                          {'windows': len(desc['rolling_windows']), 'non_48_calendar_windows': len(gaps),
                           'calendar_span_counts': dict(Counter(w['calendar_months_inclusive'] for w in desc['rolling_windows'])),
                           'nonmonthly_forecast_steps': jumps}, source_refs=[paper_ref, baseline['ref'], source_ref(repo, 'Section5_step1.py')],
                          inputs={'actual_retained_dates': 'panel.json', 'W': 48},
                          oracle='Calendar months = 12*(year_end-year_start)+month_end-month_start+1; independent of row index',
                          notes=['논문 rolling 의무 및 일반 시계열 타당성. 삭제 후 인접 행을 연결하는 전이도 다개월 전이가 된다.']))
    internal_deleted = [m for m in desc['deleted_months'] if desc['retained_start'] <= m['month'] <= desc['retained_end']]
    readme = (repo / 'README.md').read_text(encoding='utf-8-sig').splitlines()
    claim_lines = [i + 1 for i, x in enumerate(readme) if '1992' in x or '2020년 4월' in x]
    results.append(record('DATA-README', 'README 기간·삭제 설명과 실제 전처리',
                          desc['retained_start'].year == 1992 and [m['month'].strftime('%Y-%m') for m in internal_deleted] == ['2020-04'],
                          {'README_start_year': 1992, 'README_named_excluded_month': '2020-04'},
                          {'rows': desc['retained_rows'], 'first': desc['retained_start'], 'last': desc['retained_end'],
                           'internal_deleted_months': internal_deleted, 'all_deleted_count': len(desc['deleted_months'])},
                          source_refs=[ref_range(repo, 'README.md', min(claim_lines), max(claim_lines)), baseline['ref'], data_ref],
                          oracle='Actual original AST result compared to documented claims',
                          notes=['문서·구현 일치 검사이며 논문 의무가 아니다. 2020년 4월 수동 삭제문은 없고 변환/complete-case 순서에서 발생한다.']))

    # Explicit perturbations verify deletion order, protection, logs, infinity and constants.
    n = 120
    fixture = pd.DataFrame({'sasdate': pd.date_range('2000-01-01', periods=n, freq='MS'),
                            'GOOD': np.arange(n, dtype=float) + 2, 'GAP': np.arange(n, dtype=float) + 3,
                            'FEDFUNDS': np.arange(n, dtype=float) + 4, 'LOG': np.arange(n, dtype=float) + 5,
                            'ZERO': np.arange(n, dtype=float) + 6, 'CONST': np.full(n, 7.),
                            'ALLNA': np.full(n, np.nan), 'LATE': np.arange(n, dtype=float) + 9})
    fixture.loc[20:23, 'GAP'] = np.nan
    fixture.loc[40:43, 'FEDFUNDS'] = np.nan
    fixture.loc[60, 'LOG'] = 0
    fixture.loc[80, 'ZERO'] = 0
    fixture.loc[:7, 'LATE'] = np.nan
    fcodes = {c: 1 for c in fixture if c != 'sasdate'}
    fcodes.update(LOG=5, ZERO=7)
    quality = prepare(repo, 'Section5_step1.py', fixture, fcodes)
    qpanel = quality['panel']
    retained_months = set(qpanel.sasdate)
    expected_removed = set(fixture.sasdate.iloc[list(range(8)) + list(range(40, 44)) + [60, 61]])
    actual_removed = set(fixture.sasdate) - retained_months
    quality_pass = set(quality['candidate']) - set(quality['kept']) == {'GAP', 'ALLNA'} and actual_removed == expected_removed
    results.append(record('DATA-QUALITY-ORDER', '변수 선별→보호변수 유지→관측 삭제 순서', quality_pass,
                          {'removed_variables': ['GAP', 'ALLNA'], 'removed_row_indices': list(range(8)) + list(range(40, 44)) + [60, 61]},
                          {'removed_variables': sorted(set(quality['candidate']) - set(quality['kept'])),
                           'removed_months': sorted(actual_removed), 'retained_rows': len(qpanel)},
                          source_refs=[quality['ref']], inputs={'lineage': 'synthetic macro fixture', 'rows': n, 'tcode_map': fcodes},
                          oracle='4/120 internal gaps exceed 2%; protected series retained; leading gaps do not count as internal; zero log diff loses two rows',
                          notes=['결측 정책 자체는 원문 미명세. FAIL 기준은 명시된 코드 정책과 독립 손계산의 불일치.']))
    inf_count = int(np.isinf(qpanel.drop(columns='sasdate').to_numpy(float)).sum())
    results.append(record('DATA-INFINITY', 'complete-case 결과의 유한값 보장', inf_count == 0,
                          'Numerical-validity oracle: a cleaned numeric panel contains only finite values',
                          {'infinity_cells_after_dropna': inf_count, 'zero_divisor_tcode': 7},
                          source_refs=[quality['ref'], source_ref(repo, 'Section5_step1.py', 'zscore_window')],
                          inputs={'lineage': 'synthetic macro fixture', 'ZERO[80]': 0, 'tcode': 7},
                          oracle='division by zero produces infinity; pandas dropna does not remove infinity',
                          notes=['일반 수치 타당성. 원문이 이 특수값 정책을 정한 것은 아니다.', '실제 제공 CSV의 잔존 inf 개수는 panel.json에 별도 기록.']))
    zscore = load_functions(repo, 'Section5_step1.py', ['zscore_window'])['zscore_window']
    const_x = np.array([[1., 2.], [1., 4.], [1., 6.]])
    z = zscore(const_x)
    results.append(record('DATA-CONSTANT', 'rolling 표준화의 상수열 제거', same(z, [[-1.], [0.], [1.]]),
                          [[-1.], [0.], [1.]], z, source_refs=[source_ref(repo, 'Section5_step1.py', 'zscore_window')],
                          inputs=const_x, oracle='sample sd([2,4,6])=2; zero-sd column excluded', tolerance=1e-11))

    loader = load_functions(repo, 'Section5_step0_ETF_Loader.py',
                            ['compute_bom_returns_from_daily', 'align_to_macro_calendar'])
    bom = loader['compute_bom_returns_from_daily']
    simple = pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-02-03', '2020-03-02']),
                           'ticker': ['SPY'] * 4, 'adj_close': [100., 999., 110., 99.]})
    simple_return = bom(simple, tickers=['SPY'])
    results.append(record('DATA-BOM', '월초 가격과 수익률 저장 라벨', same(simple_return.SPY, [.1, -.1]),
                          {'label_2020_01': .1, 'label_2020_02': -.1, 'formula': 'R_t=P(first trading day t+1)/P(first trading day t)-1'},
                          {'labels': simple_return.index.tolist(), 'returns': simple_return.SPY.tolist()},
                          source_refs=[source_ref(repo, 'Section5_step0_ETF_Loader.py', 'compute_bom_returns_from_daily')],
                          inputs={'lineage': 'literal synthetic daily price fixture', 'records': simple.to_dict('records')},
                          oracle='100→110→99; later January price 999 must be ignored', tolerance=1e-11))
    missing_price = simple[simple.date.dt.month != 2].copy()
    gap_return = bom(missing_price, tickers=['SPY'])
    results.append(record('DATA-BOM-GAP', '가격 달력이 통째로 비면 다개월을 한 달로 표시', len(gap_return) == 0 or pd.isna(gap_return.iloc[0, 0]),
                          'Absent February price: January monthly return is missing/rejected, not January→March',
                          {'returned_label': gap_return.index.tolist(), 'returned_values': gap_return.SPY.tolist(),
                           'actual_price_span_months': 2},
                          source_refs=[source_ref(repo, 'Section5_step0_ETF_Loader.py', 'compute_bom_returns_from_daily')],
                          inputs={'lineage': 'synthetic missing-whole-month daily panel', 'removed_month': '2020-02'},
                          oracle='One-month oracle requires an explicit next calendar-month price',
                          notes=['일반 월별 자료 타당성. 단일 ETF의 결측이 전체 월 index에 남으면 NaN이 되지만 전체 panel에서 월이 없으면 shift가 건너뛴다.',
                                 '실제 Yahoo 일별 파일이 없어 이 반례의 실제 발생 여부는 미확정.']))

    # Future raw values feed the original entire preprocessing path before the real first rolling loop.
    base_decision = step1_first_decision(repo, baseline['panel'])
    future_raw = raw.copy(deep=True)
    future_mask = (future_raw.sasdate >= '2015-01-01') & (future_raw.sasdate <= '2016-12-01')
    future_raw.loc[future_mask, 'RPI'] = np.nan
    changed = prepare(repo, 'Section5_step1.py', future_raw, codes)
    altered_decision = step1_first_decision(repo, changed['panel'])
    raw_prefix_equal = raw.loc[raw.sasdate <= base_decision['record']['sasdate_t']].equals(
        future_raw.loc[future_raw.sasdate <= base_decision['record']['sasdate_t']])
    same_date = base_decision['record']['sasdate_t'] == altered_decision['record']['sasdate_t']
    invariant = same_date and same(base_decision['p_tp1'], altered_decision['p_tp1'])
    counterexample = {'cutoff_label': base_decision['record']['sasdate_t'], 'modified_raw_months': raw.loc[future_mask, 'sasdate'].tolist(),
                      'raw_at_or_before_cutoff_identical': raw_prefix_equal, 'baseline_kept': baseline['kept'], 'changed_kept': changed['kept'],
                      'baseline_first_decision': base_decision, 'changed_first_decision': altered_decision,
                      'max_abs_probability_change': float(np.max(np.abs(base_decision['p_tp1'] - altered_decision['p_tp1'])))}
    assert raw_prefix_equal and same_date, 'Counterexample must hold information cutoff and decision date fixed'
    results.append(record('DATA-FUTURE-SELECTION', '미래 raw 결측만 바꾸면 과거 레짐 결정이 변하는가', invariant,
                          'General causal-validity oracle: fixed past raw inputs imply fixed earlier decision',
                          counterexample, source_refs=[baseline['ref'], base_decision['ref'], data_ref],
                          inputs={'lineage': 'supplied FRED raw with controlled future-only RPI missing cells', 'changed_cell_count': int(future_mask.sum())},
                          oracle='Run original global internal-gap selection and exact original first rolling iteration twice; compare same-date probabilities',
                          tolerance=1e-11, notes=['일반 실시간 시계열 타당성. 논문은 변수선별 gap 정책 미명세.',
                                                '첫 rolling 결정이므로 prevC2=None은 원 실제 경로와 같다; 전체 성과 재현이 아니다.']))
    protected_raw = raw.copy(deep=True)
    protected_raw.loc[future_mask, 'FEDFUNDS'] = np.nan
    protected_prep = prepare(repo, 'Section5_step1.py', protected_raw, codes)
    protected_decision = step1_first_decision(repo, protected_prep['panel'])
    results.append(record('DATA-FUTURE-PROTECTED', '보호변수 미래 결측 대조 실험',
                          same(base_decision['p_tp1'], protected_decision['p_tp1']),
                          'Protected FEDFUNDS remains; future row deletions do not change this first earlier window',
                          {'kept_equal': baseline['kept'] == protected_prep['kept'],
                           'baseline_rows': len(baseline['panel']), 'changed_rows': len(protected_prep['panel']),
                           'probability_max_change': float(np.max(np.abs(base_decision['p_tp1'] - protected_decision['p_tp1'])))},
                          source_refs=[baseline['ref'], base_decision['ref']],
                          inputs={'lineage': 'supplied FRED raw; future FEDFUNDS 2015-01..2016-12 set NaN'},
                          oracle='Causal invariance negative control through unchanged original preprocessing and first-window loop', tolerance=1e-11))
    write_json(dest / 'future_perturbations.json', counterexample)

    # Trace forecast dependence with original Step2 loop and synthetic, explicitly labelled ETF input.
    panel = baseline['panel'].iloc[:51].copy()
    price_dates = pd.date_range(panel.sasdate.min(), panel.sasdate.max() + pd.offsets.MonthBegin(2), freq='MS')
    prices = synthetic_prices(price_dates)
    returns = bom(prices).loc[pd.DatetimeIndex(panel.sasdate)]
    decision2 = step2_first_decision(repo, panel, returns)
    t, tp1 = panel.sasdate.iloc[47], panel.sasdate.iloc[48]
    holding_start = pd.offsets.BDay().rollforward(tp1)
    later_prices = prices.copy(deep=True)
    later_prices.loc[later_prices.date > holding_start, 'adj_close'] *= 1.7
    later_returns = bom(later_prices).loc[returns.index]
    later_decision2 = step2_first_decision(repo, panel, later_returns)
    cols = [c for c in decision2['forecasts'] if c.startswith('yhat_')]
    future_prediction_delta = max(abs(decision2['forecasts'][c] - later_decision2['forecasts'][c]) for c in cols)
    boundary_returns = returns.copy(deep=True)
    boundary_returns.loc[t, 'SPY'] += .5
    boundary_decision = step2_first_decision(repo, panel, boundary_returns)
    boundary_delta = max(abs(decision2['forecasts'][c] - boundary_decision['forecasts'][c]) for c in cols)
    results.append(record('DATA-RETURN-CAUSALITY', '보유 시작 이후 가격과 이전 예측의 독립성', future_prediction_delta < 1e-11,
                          'If decision occurs after first trading price P_(t+1), changes strictly after that price must not change decision',
                          {'macro_t': t, 'holding_label_tp1': tp1, 'holding_first_price_time': holding_start,
                           'later_price_forecast_max_change': future_prediction_delta,
                           'changing_R_t_forecast_max_change': boundary_delta,
                           'training_latest_return_label': t, 'latest_training_return_end': holding_start,
                           'lambda_changed_from_later_prices': decision2['lambdas'] != later_decision2['lambdas']},
                          source_refs=[decision2['ref'], source_ref(repo, 'Section5_step0_ETF_Loader.py', 'compute_bom_returns_from_daily')],
                          inputs={'lineage': 'actual FRED first 51 retained macro rows + synthetic deterministic daily ETF prices',
                                  'price_mutation': '*1.7 strictly after P_(t+1)', 'boundary_mutation': 'SPY R_t += 0.5'},
                          oracle='Original Step2 first-iteration AST uses tau<=t-1, y=R_(tau+1); R_t completes at P_(t+1)', tolerance=1e-11,
                          notes=['shift(+1) 자체를 off-by-one/미래누출로 판정하지 않는다.',
                                 'P_(t+1) 관측 후 결정·그 가격 체결은 추가 실행 가정이다. x_t 공개 가능 여부는 별도.',
                                 '합성 ETF는 논문 성과나 WRDS/Yahoo 검증 근거가 아니다.']))

    # Retrospective full-sample Section3 normalization/PCA is distinct from rolling preprocessing.
    s3nodes, outname = prep_nodes(repo, 'Section3.py')
    body3 = next(n for n in tree_of(repo, 'Section3.py').body if isinstance(n, ast.FunctionDef)
                 and n.name == 'prepare_data_if_needed').body
    ix = next(i for i, n in enumerate(body3) if assigned(n, outname)) + 1
    end = next(i for i, n in enumerate(body3) if assigned(n, 'scores_local'))
    global_nodes = body3[ix:end + 1]
    def global_pca(p):
        scope = dict(p['scope'])
        exec_nodes(repo, 'Section3.py', global_nodes, scope)
        return scope['Z_std'], scope['scores_local']
    changed_values = raw.copy(deep=True)
    changed_values.loc[future_mask, 'RPI'] *= 2
    zbase, scoresbase = global_pca(preps['Section3.py'])
    zchanged, scoreschanged = global_pca(prepare(repo, 'Section3.py', changed_values, codes))
    oldrows = baseline['panel'].sasdate <= t
    zchange = float(np.max(np.abs(zbase.loc[oldrows].to_numpy() - zchanged.loc[oldrows].to_numpy())))
    observed_fit = {
        'Section3_fit_rows': len(zbase), 'Section3_PCA_components': scoresbase.shape[1],
        'future_RPI_value_change_past_z_max_delta': zchange,
        'Step1_window_rows': base_decision['window_rows'], 'Step1_zscore_rows': base_decision['zscore_rows'],
        'Step1_PCA_rows': base_decision['pca_rows'], 'Step2_window_rows': decision2['window_rows'],
        'Step2_zscore_rows': decision2['zscore_rows'], 'Step2_PCA_rows': decision2['pca_rows'],
        'ridge_final_regime_train_rows': decision2['final_regime_train_rows'],
        'lambda_candidate_count': decision2['lambda_candidate_count'],
        'lambda': 'analytic leave-one-out within chosen regime set; preprocessing not refit inside held-out folds'}
    fit_scope_matches = (len(zbase) == len(baseline['panel']) and zchange > 0
                         and base_decision['window_rows'] == base_decision['zscore_rows'] == base_decision['pca_rows'] == 48
                         and decision2['window_rows'] == decision2['zscore_rows'] == decision2['pca_rows'] == 48
                         and decision2['final_regime_train_rows'] <= 47 and decision2['lambda_candidate_count'] == 8)
    results.append(record('DATA-FIT-SCOPE', 'Section3 전체표본과 Step1/2 창내 적합 범위 구분', fit_scope_matches,
                          {'Section3': 'retrospective whole-sample normalization/PCA', 'Step1/2': 'window-local zscore/PCA/clustering; globally preselected variables'},
                          observed_fit,
                          source_refs=[ref_range(repo, 'Section3.py', global_nodes[0].lineno, global_nodes[-1].end_lineno, 'full_sample_standardization_PCA'),
                                       base_decision['ref'], decision2['ref']],
                          inputs={'lineage': 'supplied FRED + future RPI values multiplied by 2'},
                          oracle='Original AST execution and scope tracing; PASS confirms classification of fitting scope, not full causal validity',
                          notes=['Section3는 회고적 레짐 분석이므로 전체표본 적합 자체를 논문 위반으로 부르지 않는다.',
                                 '해당 전기간 scores/labels를 실시간 전략으로 재사용하면 시점상 추가 문제가 된다. Step1/2는 별도 창내 재계산.',
                                 'lambda CV는 시간 순서별 forward validation이 아니며 fold 밖으로 PCA/군집 재적합을 분리하지 않는다; 논문 CV 세부 미명세.']))

    # Inputs absent locally remain explicit skips, not invented empirical claims.
    results.append(record('DATA-RELEASE-VINTAGE', '관측 기준월·공개일·실시간 빈티지 일치', False,
                          'Each x_t must be the vintage released by the actual decision timestamp',
                          {'supplied_vintage_filename': 'FRED-MD_2024m12.csv', 'raw_shape': [len(raw), len(raw.columns) - 1],
                           'availability_or_vintage_columns': [], 'decision_timestamp': 'not specified in source; only sasdate_t / sasdate_tp1'},
                          source_refs=[data_ref, baseline['ref'], decision2['ref']],
                          inputs='Repository contains one fixed macro snapshot named 2024m12, without per-value release/vintage times; official vintage byte/content identity unverified',
                          oracle='As-of join by release/vintage time, independent of reference-month stamp', status='SKIP',
                          notes=['월초 P_(t+1) 결정 해석에서도 모든 x_t가 공개됐는지 확인할 입력이 없다.',
                                 '실시간 가능성 일반 요건. 논문은 공개지연/빈티지 정책을 충분히 정하지 않았다.',
                                 '2024m12로 명명된 저장소 스냅샷은 공식 원 빈티지와 byte/내용 동일성이 미확인이다. 역사적 수정 전 자료를 복원하지 않는다.']))
    results.append(record('DATA-WRDS-YAHOO', 'WRDS 논문 가격과 Yahoo 구현 자료 대조', False,
                          'Paper Table 2: WRDS 10-ETF beginning-of-month data, reported 2000-02..2022-12',
                          {'implementation': 'yfinance Ticker.history(auto_adjust=True); Close; 1993-01-01..2024-10-31',
                           'local_real_daily_prices': False, 'local_real_return_csv': False,
                           'tickers': TICKERS, 'requested_end_macro_window': '2024-09-01'},
                          source_refs=[paper_ref, source_ref(repo, 'Section5_step0_ETF_Loader.py', '_fetch_real_prices')],
                          inputs='No WRDS price input or persisted Yahoo daily/return panel supplied',
                          oracle='Date-by-date matched provider, adjustment, first-trading-day and coverage comparison', status='SKIP',
                          notes=['자료 제공자·기간 차이는 확정되나 수치 영향은 검증 불가. 다운로드는 실행하지 않았다.',
                                 '논문 746개월 서술은 기간 산술과 맞지 않으며 정확한 저자 표본은 미확정.']))
    lineage = []
    for filename in ['Section5_step3.py', 'Section5_step4.py']:
        fn = load_functions(repo, filename, ['synthesize_etf_bom_returns'])['synthesize_etf_bom_returns']
        synth = fn(pd.date_range('2000-01-01', periods=12, freq='MS'))
        source_tree = tree_of(repo, filename)
        synth_calls = [n for n in ast.walk(source_tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'synthesize_etf_bom_returns']
        fallback_elses = [n for n in ast.walk(source_tree) if isinstance(n, ast.If)
                          and any(isinstance(k, ast.Call) and isinstance(k.func, ast.Name)
                                  and k.func.id == 'synthesize_etf_bom_returns'
                                  for node in n.orelse for k in ast.walk(node))]
        lineage.append({'file': filename, 'source_ref': source_ref(repo, filename, 'synthesize_etf_bom_returns'),
                        'trigger': 'upstream forecast missing; aligned ETF file missing in autogeneration branch',
                        'shared_output': 'etf_bom_returns_aligned_demo.csv', 'seed': 123,
                        'shape': list(synth.shape), 'first_spy': float(synth.SPY.iloc[0]),
                        'source_synthesis_calls': len(synth_calls), 'source_synthesis_in_fallback_else': len(fallback_elses),
                        'all_generated_values_finite': bool(np.isfinite(synth.to_numpy(float)).all()),
                        'lineage': 'synthetic normal market factor plus ticker-specific normal noise'})
    source0 = tree_of(repo, 'Section5_step0_ETF_Loader.py')
    legacy = next(n for n in source0.body if isinstance(n, ast.FunctionDef) and n.name == '_make_synth_prices')
    real_delegation = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                          and n.func.id == '_fetch_real_prices' for n in ast.walk(legacy))
    verified_lineage = real_delegation and all(x['shape'] == [12, 10] and x['source_synthesis_calls'] >= 1
                                              and x['source_synthesis_in_fallback_else'] >= 1
                                              and x['all_generated_values_finite'] for x in lineage)
    results.append(record('DATA-SYNTHETIC-LINEAGE', '실제·합성 ETF 입력 계보 표시', verified_lineage,
                          'Explicitly distinguish real downloader from synthetic fallbacks and audit fixtures',
                          {'step0_legacy_make_synth_prices': 'delegates to real Yahoo downloader', 'fallbacks': lineage,
                           'audit_fixtures': ['literal daily prices', 'deterministic smooth daily prices', 'macro quality fixture']},
                          source_refs=[source_ref(repo, 'Section5_step0_ETF_Loader.py', '_make_synth_prices')] + [x['source_ref'] for x in lineage],
                          inputs='AST-loaded synthesis functions only, no original output writes',
                          oracle='Trace real and synthetic branches; sample function execution verifies synthetic generation',
                          notes=['원 코드 합성 fallback은 실데이터와 같은 ETF 파일명을 쓴다. 파일명만으로 실제 관측 자료라고 증명할 수 없다.']))
    write_json(dest / 'input_lineage.json', {'source_csv': data_ref, 'fallbacks': lineage,
                                           'real_etf_available': False, 'all_tests_without_download': True})

    timeline = []
    for tlabel, nextlabel in [(t, tp1)] + [(w['decision_label'], w['next_retained_label']) for w in jumps]:
        timeline.append({'macro_reference_month_t': tlabel,
                         'macro_actual_release_time': 'unknown per variable; not encoded in supplied CSV',
                         'etf_price_observation_at_t': 'first trading-day adjusted Close of t',
                         'stored_return_R_t': 'P(first trading day t+1)/P(first trading day t)-1',
                         'last_train_predictor': 'previous retained macro row before t',
                         'last_train_target_label': tlabel,
                         'last_train_target_price_interval': f'first trading day {tlabel:%Y-%m} to first trading day {(tlabel + pd.offsets.MonthBegin()):%Y-%m}',
                         'decision_label_in_file': tlabel,
                         'decision_time_in_code': 'not specified; P_(t+1) observation is earliest price-consistent interpretation for complete monthly calendar',
                         'saved_sasdate_tp1': nextlabel,
                         'holding_return_label_used_step5': nextlabel,
                         'holding_price_interval': f'first trading day {nextlabel:%Y-%m} to first trading day {(nextlabel + pd.offsets.MonthBegin()):%Y-%m}',
                         'macro_label_step_in_calendar_months': month_delta(tlabel, nextlabel)})
    write_json(dest / 'timeline.json', timeline)
    notes = make_notes(desc, internal_deleted, gaps, jumps, counterexample, future_prediction_delta, boundary_delta, results, data_ref, timeline)
    (dest / 'notes.md').write_text(notes, encoding='utf-8')
    write_results(output, 'data_timing', results)
    return results


def make_notes(desc, deleted, gaps, jumps, counterexample, future_delta, boundary_delta, results, data_ref, timeline):
    dates = ', '.join(x['month'].strftime('%Y-%m') for x in deleted)
    rows = '\n'.join('| ' + ' | '.join([str(x['macro_reference_month_t'].date()), '변수별 공개일 없음',
                     '첫 거래일 adjusted Close', str(x['last_train_target_price_interval']),
                     '미명시; 정상 달력은 P(t+1) 관측 이후 해석 가능', str(x['holding_price_interval'])]) + ' |' for x in timeline)
    return f'''# 자료·시간축 감사

## 범위와 증거

원 연구 코드·자료는 수정하지 않았다. 최상위 다운로드·원 위치 파일 저장을 실행하지 않았다. 원본 함수 및 전처리/단일 rolling 반복문 AST를 그대로 메모리에서 실행했다. `results.json` 각 항목의 `source_refs`는 실제 소스 SHA-256과 원래 줄 범위다. 원자료 해시는 `{data_ref['sha256']}`이다. 독립 산식 자체검사 9개를 먼저 통과한 뒤 원 구현과 비교했다. FAIL은 반례 발견이며 감사 도구 실패가 아니다. 전체 백테스트·수익률 성과 재현은 범위 밖이다.

## 실제 전처리 결과

- 원 CSV는 1959-01~2024-11, 791개월·121변수다. 실행은 변환 **전에** 1978-07~2024-09의 {desc['raw_rows']}행으로 자른다. 따라서 최초 1/2개월 차분 자료를 경계 밖 역사로 채우지 않는다.
- 그룹 처리 후 후보 {len(desc['candidate_variables'])}변수 → 내부 결측 비율 2% 초과 비보호변수 제거 → 남은 변수 {len(desc['retained_variables'])}개 → complete-case {desc['retained_rows']}개월이다. 시작 {desc['retained_start']:%Y-%m}, 끝 {desc['retained_end']:%Y-%m}. 삭제된 달 전체 {len(desc['deleted_months'])}개와 달별 결측 원인은 `panel.json`에 있다.
- 실제 기간 내부 삭제월: {dates}. README의 1992년 시작·2020년 4월 설명과 실제 결과를 분리해야 한다. 코드에 2020년 4월만 수동 삭제하는 문장은 없다. 제공 CSV의 4월 CP3Mx·COMPAPFFx가 결측이며, CP3Mx의 t-code 2 차분이 5월까지 결측을 전파한다. 해당 raw 결측을 누가 언제 만들었는지는 이 스냅샷만으로 확정할 수 없다.
- 그룹6 중 실제 유지 {len(desc['retained_group6'])}개: {', '.join(desc['retained_group6'])}. 논문 p.7의 그룹6 제외와 다르다.
- 내부 gap 정책으로 제거된 변수: {', '.join(desc['removed_gap_variables']) or '없음'}.
- 실제 유지 변수: {', '.join(desc['retained_variables'])}.
- 모든 유지 날짜, 삭제 날짜, 변수별 처음/마지막 유효월·내부 결측률·비양수 원값·NaN/inf 개수는 `panel.json`에 저장했다. Step2의 실 ETF 교집합 달력은 실제 ETF 파일이 없어 확정할 수 없다.

## t-code와 특수값

6개 구현의 t-code 1~7을 양수·0·음수·내부 결측·상수 fixture 168회 비교했다. 핵심 경로 5개 구현은 모두 일치했다. 유일한 변환 불일치는 `etc_regime_stats_transformed.py`의 code7에서 `pct_change` 기본 결측 채움이 관측된 경우다(현재 pandas 실행 환경); 이 utility 분기 결과를 핵심 전략 변환 오류로 확대하지 않는다. code2/5는 첫 1행, code3/6/7은 첫 2행을 잃는다. 내부 결측/비양수 로그는 차분 횟수만큼 후속 행에 전파된다. 보호변수는 내부 gap이 커도 유지되어 해당 관측을 제거한다. 끝의 결측과 시작 전 결측은 내부 비율에 포함되지 않으므로 시작이 늦은 변수는 표본 전체 시작을 뒤로 미룬다. 상수는 complete-case에서는 유지되고 rolling 표준화에서 제외된다. code7의 0 분모는 inf를 만들고 dropna만으로는 제거되지 않는 반례를 확인했다. 이 특수값 요건은 일반 수치 타당성이며 원문의 구체 결측 정책은 아니다. 후속 zscore는 NaN 표준편차의 열을 묵시적으로 제외할 수 있으므로 열 제거와 행 제거는 다르다.

## 48행과 48개월

총 {len(desc['rolling_windows'])}개 macro rolling 창 중 {len(gaps)}개가 연속 48개월이 아니다. 길이별 개수: {dict(Counter(w['calendar_months_inclusive'] for w in desc['rolling_windows']))}. 다음 retained row를 다음 달로 부르는 다개월 이동 {len(jumps)}개를 발견했다. 원문 Section 6의 48개월·1개월 앞 예측과 구분해야 한다. 같은 삭제 달력에서 전이 추정도 인접 행을 인접 월로 취급한다. ETF 로더 역시 전체 가격 panel에서 한 달이 통째로 없으면 shift(-1)가 다개월 가격변화를 한 달 라벨로 저장한다. 실제 Yahoo 자료에서 그 반례가 발생했는지는 일별 자료 부재로 미확정이다.

## 가격·목표·결정·보유 시점

월초 저장 라벨 R_t는 P(t+1)/P(t)-1이다. Step2의 x_tau→R_(tau+1), 마지막 학습 tau=t-1은 R_t를 사용한다. R_t는 P(t+1)을 관측해야 끝난다. Step5는 sasdate_tp1 라벨 수익률, 즉 정상 달력에서는 P(t+1)~P(t+2)를 보유 수익률로 쓴다. **결정을 P(t+1) 관측 후로 해석하면 target의 shift만으로 미래누출이라고 할 수 없다.** 같은 가격으로 주문·체결 가능한지는 구현에 없는 실행 가정이며 x_t 공개 가능성도 별도로 남는다. sasdate_t 자체를 매매 시각이라고 읽으면 R_t는 그때 알려져 있지 않다.

| 거시 기준월 t | 거시 실제 공개시점 | ETF 가격 관측 | 최신 학습 목표 가격구간 | 결정 시점 | 실제 보유 가격구간 |
|---|---|---|---|---|---|
{rows}

각 열의 기계 판독본은 `timeline.json`이다. 첫 정상 예시의 macro는 제공 원자료에서, ETF 가격은 명시적 합성 fixture에서 가져왔다. 2020년 점프 예시는 실제 macro 날짜지만 ETF 가격구간은 로더/Step5 정의에 따른 상징적 해석이며 실제 가격 관측을 주장하지 않는다.

## 미래 변경 실험과 적합 범위

2015-01~2016-12 RPI raw 24셀만 NaN으로 바꾸고 **전체 원 전처리**부터 첫 rolling 반복문까지 재실행했다. 과거 raw는 동일하고 최초 결정일도 {counterexample['cutoff_label']:%Y-%m}로 같은데 과거 변수집합 및 다음 레짐 확률이 변했다(최대 절대차 {counterexample['max_abs_probability_change']:.12g}). 이는 전체 표본의 >2% 내부 결측률로 변수를 고르는 경로의 일반 시계열 누출 반례다. 실제 과거 정보시점 기준 변수집합을 고정하거나 매 시점 선택하는 정책은 원문이 충분히 정하지 않는다. 보호변수 FEDFUNDS의 같은 미래 결측 변경은 미래 관측만 제거하고 해당 과거 결정은 바뀌지 않는 대조 실험을 수행했다.

Step1/2 표준화·PCA·군집은 각 48행 창에 적합한다. 상수열 제외도 창 안에서 일어난다. 그러나 그 앞의 변수/관측 선별은 전체 기간에 의존한다. Ridge의 최대 47개 학습 설명변수 행, 각 regime 표본선택, 목표 중심화와 lambda 선택의 범위를 원 loop에서 추적했다. lambda는 8개 후보의 analytic LOOCV이며 PCA/군집/중심화 등을 각 held-out fold에서 다시 적합하는 시간순 forward validation은 아니다. 이는 원문 미명세 추가 선택으로 기록한다.

원 Step2 첫 loop에 실제 macro와 합성 ETF를 넣고 P(t+1) 이후 가격만 1.7배 변경한 결과 이전 예측 최대 변화 {future_delta:.12g}, lambda 변화 없음이다. 반면 마지막 학습 target R_t를 0.5 바꾸면 예측 최대 변화 {boundary_delta:.12g}다. 이는 target의 정보 상한을 확인하며 공개지연 문제를 해결하지 않는다. Section3는 전체표본 표준화/PCA인 회고 분석이고 미래 RPI 값만 바꿔도 과거 z-score가 변한다. 이를 그대로 실시간 신호로 재사용하는 경우와 Step1/2의 재적합을 구분한다.

## 공개시점·빈티지·자료 제공자

제공 입력은 `FRED-MD_2024m12.csv`로 명명된 단일 저장소 스냅샷이다(공식 원 빈티지와 byte/내용 동일성 미확인). 거시 기준월만 있고 각 값의 발표일/수정일/당시 빈티지 및 결정 시각이 없다. 과거 거래 시점에 사용할 수 있었던 최신 값인지 검증하는 as-of 비교는 SKIP이다. [St. Louis Fed의 FRED-MD 안내](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)는 자료 수정과 역사 빈티지를 설명한다. [FRED 실시간 기간 문서](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)는 관측 기간과 실시간 기간을 구별한다. [BLS CPI 발표 일정](https://www.bls.gov/schedule/news_release/cpi.htm)은 기준월과 발표 날짜/시각을 별도 열로 제시한다. 따라서 기준월과 첫 거래일을 같게 보는 것만으로 자료 가용성을 보장하지 못한다. 모든 변수에 일률적인 한 달 lag를 충분한 것으로 가정하지 않았다.

논문은 WRDS, 10 ETF, 2000-02~2022-12를 기술한다. 코드는 Yahoo/yfinance, auto_adjust=True의 Close, 1993-01-01~2024-10-31 다운로드를 요청한다. 소스/조정 방식/수집 빈티지/거래일 달력/커버리지의 수치 차이를 확정할 실제 가격 입력은 없다. 새 다운로드로 원 자료를 가장하지 않았고 실제 수익률 기간/결정 수/성과는 SKIP이다. 논문의 746개월 수치는 적힌 기간과 산술적으로 맞지 않는다.

## 합성 자료 계보

Step0의 `_make_synth_prices`는 이름과 달리 실제 Yahoo 함수로 위임한다. Step3/4는 선행 예측 부재 시 자동생성하며 ETF CSV도 없으면 seed=123 합성 수익률을 만든다. 두 경로가 `etf_bom_returns_aligned_demo.csv`라는 같은 파일명에 저장하므로 이후 파일명으로 실제/합성을 판별할 수 없다. 감사는 함수만 불러 표본을 확인하고 파일 저장은 실행하지 않았다. 모든 합성 fixture와 원본 분기의 계보는 `input_lineage.json` 및 개별 inputs에 표시했다.

## 판정 해석

- 논문 충실도: 그룹6 제외, 48개월/한 달 forecast 의무를 실행값과 비교했다.
- 일반 타당성: 미래 결측에 의한 과거 결정 변화, inf 잔존, 월 누락 가격의 잘못된 1개월 표시는 반례로 FAIL이다.
- 문서 일치: README 기간/삭제 설명은 별도 기준이다.
- 재현 가능성 한계: release/vintage와 WRDS/Yahoo 원 가격은 SKIP이다. 이들은 논문/코드의 구체적 결정 시점 미명세와 입력 부재를 숨기지 않는다.
- 현재 결과: {dict(Counter(r['status'] for r in results))}. 원 코드를 고쳐 FAIL을 제거하지 않았다.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False))
        return 0
    output = args.output or args.repo / 'reports/paper_audit/evidence'
    results = run(args.repo, output)
    counts = dict(Counter(r['status'] for r in results))
    print(json.dumps({'domain': 'data_timing', 'results': len(results), 'counts': counts,
                      'output': str(output / 'data_timing/results.json')}, ensure_ascii=False))
    return int(counts.get('ERROR', 0) > 0)


if __name__ == '__main__':
    sys.exit(main())
