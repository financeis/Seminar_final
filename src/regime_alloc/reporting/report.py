"""Immutable Korean reports from existing evidence; never fit or rerun a model.

Bundle contract: kind=research_evidence_bundle, runs=[{path, role,
execution_scope}], references=[{path, role}], unrun=[description]. Paths can
be absolute or relative to the bundle file. References are copied, not edited.
Generation describes evidence; use verify_run separately to audit calculations.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from ..contracts import ErrorCode, ResearchError, sha256_file, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


def _ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def _table(rows, fields):
    def value(item):
        try:
            if isinstance(item, str) and '.' in item and item.count('.') == 1:
                return f'{float(item):.4f}'
        except ValueError:
            pass
        return str(item).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---']*len(fields)) + ' |'] +
        ['| ' + ' | '.join(value(row.get(key, '')) for key in fields) + ' |' for row in rows])


def _csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _source_root():
    return Path(__file__).resolve().parents[3]


def _trace_and_findings(output, references):
    root = _source_root()
    groups = [
        ('EQ01 EQ02 EQ03 EQ04 EQ06', '거리 소속도와 끝점 정규화', 'regimes/probabilities.py', 'test_probabilities.py'),
        ('EQ05 EQ07', '출발행 기준 전이와 다음 달 소속도', 'regimes/transitions.py', 'test_regimes.py'),
        ('EQ08 EQ09 EQ10', 'Naive 조건부 Sharpe 점수', 'models/naive.py', 'test_naive.py'),
        ('EQ11', 'BL 기대수익 view와 사후 평균', 'models/black_litterman.py', 'test_black_litterman.py'),
        ('EQ12 EQ13 EQ14', 'Ridge 예측·정규화·일반 레짐 집계', 'models/ridge.py', 'test_ridge.py'),
        ('EQ15 EQ16 EQ17 EQ18 EQ19', '점수의 순위와 lo/lns/los/mx 비중', 'portfolio/sizing.py', 'test_sizing.py'),
        ('ALG1', '작은 L2 군집 R0와 일반 cosine 군집', 'regimes/clustering.py', 'test_regimes.py'),
        ('FIG01 FIG02 FIG03 FIG04 FIG05 FIG06 TABLE01', '전표본 PCA·GMM·NBER 사후 대조·전이 해석', 'reporting/regime_analysis.py', 'test_regime_analysis.py'),
        ('FIG07 FIG08 FIG09 TABLE03 TABLE04 TABLE05', '무작위 레짐 대조와 통계', 'backtest/statistics.py', 'test_statistics.py'),
        ('FIG10 FIG11 FIG12 FIG13 TABLE06', '투자 성과와 과거 변동성 조정', 'backtest/engine.py', 'test_engine.py'),
        ('TABLE02', 'ETF 목록·실제 공개 가격과 시점 계약', 'data/yahoo.py', 'test_data.py'),
    ]
    items = []
    for ids, meaning, source, test in groups:
        paths = [root/'src/regime_alloc'/source, root/'tests'/test]
        # A missing named test is explicit; never manufacture a reference.
        refs = [_ref(p) for p in paths if p.exists()]
        for ident in ids.split():
            unrun = ident in {'FIG07', 'FIG08', 'FIG09', 'TABLE03', 'TABLE04', 'TABLE05'}
            items.append({'id': ident, 'meaning': meaning,
                'execution_scope': 'NOT_RUN_FULL' if unrun else 'available_evidence_only',
                'difference': '원문 수치 일치 인증이 아닌 구현 및 수행 범위 대응. 실험별 표본·버전은 본문 참조.',
                'references': refs + references})
    write_json(output/'traceability.json', {'items': items, 'references': references})
    labels = ['공유 partition ID', 'fit 표본만 특징 선택', '절편을 포함한 LOO', '48 연속 달력월',
        '누락·중복·활성 NaN 거부', '초기 NAV=1 낙폭', '고정 자료·그룹6 제외·변수 수', '원래 좌표에서 레짐 매칭',
        '작은 레짐 fallback과 진단', '4모형·4배분법', '무작위 대조·통계', 'GMM·NBER·전이 사후 해석',
        '과거 변동성 조정·로그 곡선', '산출물 계약', '암묵적 복구 없음', '실자료/fixture·해시',
        '비유한 변환 거부', '퇴화 군집·영벡터', '출발 전이 정규화와 원문 진단', '영거리·확률 끝점',
        'R0=1일 때 일반 Ridge 0/현금', 't-code 1회 적용', '지표·비중 정의', 'ETF·vintage·WRDS/공개일 한계',
        '결정 시점·BL·표본 가정']
    findings = []
    finding_modules = ['regimes/state.py', 'features/preprocessing.py', 'models/ridge.py', 'features/windows.py',
        'backtest/engine.py', 'backtest/metrics.py', 'data/fred.py', 'regimes/matching.py', 'models/ridge.py',
        'portfolio/sizing.py', 'backtest/statistics.py', 'reporting/regime_analysis.py', 'portfolio/volatility.py',
        'contracts.py', 'data/io.py', 'data/io.py', 'features/transforms.py', 'regimes/clustering.py',
        'regimes/transitions.py', 'regimes/probabilities.py', 'models/ridge.py', 'features/transforms.py',
        'backtest/metrics.py', 'data/yahoo.py', 'models/black_litterman.py']
    for index, label in enumerate(labels, 1):
        ident = f'F{index:02d}'
        disposition = 'external_limitation' if index == 24 else 'assumption_documented' if index in {19, 23, 25} else 'implemented' if index in {10, 11, 12} else 'fixed'
        findings.append({'id': ident, 'description': label, 'disposition': disposition,
            'execution_scope': 'NOT_RUN_FULL' if index == 11 else 'bounded_tests_and_versioned_evidence',
            'limitations': ['최종 코드의 전체 실험 재실행 인증이 아님'],
            'references': [_ref(root/'src/regime_alloc'/finding_modules[index-1])] + references})
    write_json(output/'findings.json', {'items': findings, 'references': references})
    write_json(output/'finding_resolution.json', {'items': findings, 'references': references,
        'alias': 'findings.json contains the same resolution records'})
    return items, findings


def _curves(run, output, label):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    smoke = _read(run/'effective_config.json').get('smoke', False)
    paths = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, folder, title in zip(axes, (run, run/'scaled'), ('Unscaled', 'Past-volatility scaled')):
        source = folder/'returns.csv'
        if not source.exists():
            ax.text(.5, .5, 'Not available', transform=ax.transAxes, ha='center')
            continue
        rows = _csv(source)
        strategies = sorted({r['strategy_id'] for r in rows})
        for strategy in strategies:
            series = sorted((r for r in rows if r['strategy_id'] == strategy), key=lambda r:r['holding_month'])
            months = [r['holding_month'] for r in series]
            featured = strategy in {'spy', 'ew', 'ridge_lo_2', 'naive_lo_2', 'bl_lo_2', 'mvo_lo_2'}
            if smoke:
                ax.scatter(range(len(series)), [float(r['net_return']) for r in series],
                    label=strategy if featured else None, s=22 if featured else 9,
                    alpha=1 if featured else .20, color=None if featured else '#64748b')
            else:
                ax.plot(range(len(series)), [float(r['wealth']) for r in series],
                    label=strategy if featured else None, linewidth=1.7 if featured else .65,
                    alpha=1 if featured else .20, color=None if featured else '#64748b')
        if not smoke:
            ax.set_yscale('log')
        if rows:
            ticks = sorted({0, len(months)//2, len(months)-1})
            ax.set_xticks(ticks, [months[i] for i in ticks])
        ax.set(title=f'{label}: {title} ({len(strategies)} strategies)'
            + (' — diagnostic independent months' if smoke else ''),
            ylabel='Independent monthly net return' if smoke else 'NAV (log scale)')
        ax.grid(alpha=.2)
        if strategies:
            ax.legend(fontsize=8, ncol=3)
    for suffix in ('png', 'svg'):
        destination = output/f'{label}_curves.{suffix}'
        fig.savefig(destination, dpi=140, metadata={'Date': None} if suffix == 'svg' else {})
        paths.append(destination.name)
    plt.close(fig)
    return paths


def generate_report(run, output=None):
    """Describe a saved single run, suite or explicit evidence bundle.

    Output must not exist. The default is a timestamped sibling of the input;
    original artifacts/manifests are never modified. This does not silently
    certify the input: financial verification is a separate explicit operation.
    """
    run = Path(run).resolve()
    source_file = run if run.is_file() else run/'run_manifest.json'
    data = _read(source_file)
    now = datetime.now(timezone.utc)
    output = Path(output).resolve() if output else source_file.parent.parent/f'report-{now.strftime("%Y%m%dT%H%M%S%fZ")}'
    if output == source_file.parent or (run.is_dir() and output.is_relative_to(run)):
        raise ResearchError(ErrorCode.RUN_CONFLICT, 'report must be outside the input directory')
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ResearchError(ErrorCode.RUN_CONFLICT, f'report output already exists: {output}') from exc
    sources = [_ref(source_file)]
    manifest = {'kind': 'research_report', 'schema_version': '1.0', 'status': 'running',
        'started_at': now.isoformat(), 'sources': sources, 'artifacts': [],
        'verification': 'not_performed_by_report_generator'}
    # This exclusively owned lifecycle file is the only file updated in place.
    def publish():
        (output/'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    publish()
    try:
        bundle = data.get('kind') == 'research_evidence_bundle'
        entries = data.get('runs', []) if bundle else [{'path': str(source_file.parent), 'role': 'run', 'execution_scope': 'saved_run_only'}]
        refs = data.get('references', []) if bundle else []
        runs = []
        copied = []
        for i, entry in enumerate(refs):
            path = Path(entry['path'])
            path = path if path.is_absolute() else source_file.parent/path
            path = path.resolve()
            if not path.is_file():
                raise ResearchError(ErrorCode.MISSING_DATA, f'report reference missing: {path}')
            if entry.get('sha256') and sha256_file(path) != entry['sha256']:
                raise ResearchError(ErrorCode.HASH_MISMATCH, f'report reference hash mismatch: {path}')
            destination = output/'evidence'/f'{i:02d}_{path.name}'
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
            sources.append(_ref(path))
            copied.append({'role': entry.get('role', path.name), 'path': destination.relative_to(output).as_posix(),
                'sha256': sha256_file(destination), 'source': str(path)})
        for entry in entries:
            path = Path(entry['path'])
            path = path if path.is_absolute() else source_file.parent/path
            path = path.resolve()
            rm = _read(path/'run_manifest.json')
            if rm.get('status') != 'succeeded':
                raise ResearchError(ErrorCode.VERIFICATION_FAILED, f'report input did not succeed: {path}')
            from .verification import _inventory
            _inventory(path, rm, 'run_manifest.json')
            sources.append(_ref(path/'run_manifest.json'))
            runs.append((path, entry, rm))
        write_json(output/'evidence_index.json', {'references': copied, 'sources': sources})
        trace, findings = _trace_and_findings(output, copied)
        lines = ['# 거시경제 레짐 자산배분 재구현 보고서', '',
            '## 먼저 읽을 결론', '',
            '이 보고서는 **실행 가능한 재구현과 수행한 범위의 증거**를 설명한다. 원 논문 전체 실험을 최종 코드로 완전 재현했다고 주장하지 않는다. 보고서 생성은 저장 결과를 읽고 그림을 만들며, 모델을 다시 학습하거나 실험을 새로 실행하지 않는다.', '',
            '수익률은 소수 단위다(0.10=10%). 기본 성과 표는 비조정 전략이며, 변동성 조정 곡선은 별도다. 여러 전략 중 높은 결과를 사후 선택한 값은 미래 투자 성과의 보장이 아니다.', '',
            '## 계산을 이해하는 작은 예', '',
            '### 경제 자료 → 레짐 → 예측', '',
            'PCA는 같이 움직이는 경제지표를 요약한다. 고용지표 다섯 개가 비슷하게 움직이면 그 공통 움직임을 하나의 축으로 압축할 수 있다. **분산 설명력 95%는 입력 움직임의 보존 정도이며 예측 정확도 95%가 아니다.** 전처리·PCA는 거래 판단의 학습 표본 안에서 적합한다.', '',
            '아주 먼 한 점은 두 집단 나누기를 좌우할 수 있다. 첫 L2 분할의 작은 집단을 R0로 두고 나머지를 cosine 기준으로 나눈다. 레짐 이름은 침체 정답이 아니다. 서로 다른 PCA 창을 비교할 때 원래 변환 변수 좌표로 되돌려 맞춘다. 거리 소속도는 부드러운 가중치이며 보정된 경기침체 확률이 아니다. NBER는 사후 대조용이며 예측 입력이 아니다.', '',
            'Ridge는 계수가 너무 커지는 것을 억제한다. λ를 늘리면 복잡한 반응을 더 강하게 제한한다. LOO는 한 관측을 숨겨 점수를 매기며 절편의 1/n 효과를 포함한다. 기본 LOO는 외부 48개월의 특징 공간·소속을 조건으로 한다. 전진 검증은 과거 시점마다 다시 적합하며, 기본의 레짐별 λ와 달리 ETF별 공통 λ를 선택하므로 두 결과 차이를 검증 방식 하나의 효과라고 단정할 수 없다.', '',
            '### 예측값 → 비중 → 순수익', '',
            '예를 들어 양수 점수가 0.02와 0.01이면 절댓값 합 0.03으로 나눈 lo 비중은 2/3와 1/3이다(설명용 숫자). Naive의 조건부 Sharpe 점수, Ridge의 기대수익, BL의 사후 기대수익, MVO의 효용 방향은 의미가 다르며 어느 것도 곧바로 최종 비중은 아니다. lo는 양수, lns는 양·음수 각각, los는 절댓값 순위, mx는 다음 레짐에 따라 배분법을 선택한다.', '',
            '초기 자산 1에서 첫 달 10% 손실이면 자산은 0.9이고 낙폭은 -10%다. 수익은 비중×ETF수익의 합에 현금수익을 더하고 거래·차입·조달 비용을 뺀다. 다음 달 거래 전 비중은 이번 달 수익으로 변한 보유액을 순자산으로 나눈 값이다. 과거 변동성 조정은 당시 알려진 수익만 사용하며 실제 달성 변동성이 정확히 10%라는 뜻은 아니다.', '',
            '## 실행 증거와 버전', '']
        inventory = []
        suite_sections = []
        for index, (path, entry, rm) in enumerate(runs):
            period = rm.get('actual_period', {})
            inventory.append({'역할': entry.get('role', 'run'), '버전': rm.get('code_revision', rm.get('code_version', 'manifest 참조')),
                '상태': rm['status'], '범위': entry.get('execution_scope', 'saved_run_only'),
                '개월': period.get('n_months', 'suite 자식별'), '원본': str(path)})
            if (path/'metrics.csv').exists():
                label = f'run{index:02d}_{rm.get("profile", "profile")}'
                rows = _csv(path/'metrics.csv')
                table_path = output/f'{label}_main_metrics.csv'
                shutil.copyfile(path/'metrics.csv', table_path)
                for filename in ['metrics.csv', 'metrics_reasons.json', 'returns.csv', 'scaled/metrics.csv', 'scaled/metrics_reasons.json', 'scaled/returns.csv']:
                    if (path/filename).exists():
                        sources.append(_ref(path/filename))
                reasons = _read(path/'metrics_reasons.json') if (path/'metrics_reasons.json').exists() else {'reason': 'not supplied'}
                write_json(output/f'{label}_metrics_reasons.json', reasons)
                curves = _curves(path, output, label)
                lines += [f'### {entry.get("role", label)} / {rm.get("profile", "")}', '',
                    f'실제 저장 표본: {period.get("start_month", "미기록")}~{period.get("end_month", "미기록")}, {period.get("n_months", "미기록")}개월, {len(rows)}전략. 범위: {entry.get("execution_scope", "saved_run_only")}.', '',
                    f'[전체 지표 CSV]({table_path.name}) · [정의 불가 사유]({label}_metrics_reasons.json)', '',
                    _table(rows, ['strategy_id', 'n_months', 'cagr', 'ann_vol', 'sharpe', 'sortino', 'maxdd', 'positive_ratio']), '',
                    f'![비조정 및 변동성 조정 자산 곡선]({curves[0]})', '', f'[편집 가능한 SVG]({curves[1]})', '']
                defined = [r for r in rows if r.get('sharpe', '')]
                if defined:
                    best = max(defined, key=lambda r: float(r['sharpe']))
                    spy = next((r for r in defined if r['strategy_id'] == 'spy'), None)
                    lines += [f'저장 결과에서 가장 높은 비조정 Sharpe는 {best["strategy_id"]}의 {float(best["sharpe"]):.3f}다. '
                        + (f'SPY는 {float(spy["sharpe"]):.3f}다. ' if spy else '')
                        + '이는 같은 표본에서 여러 전략을 본 사후 비교이며 대조군 전체실험의 유의성 결론이 아니다.', '']
            else:
                sr = _read(path/'suite_results.json')
                suite_sections += [f'### {entry.get("role", "suite")}', '',
                    f'완료 자식 {len(sr.get("completed", []))} / 예정 {sr.get("planned_count", "미기록")}. {entry.get("execution_scope", "saved_suite_only")}.', '',
                    '짧은 smoke는 연결 진단이다. 3개월·2대조군·8회 재표집의 숫자나 p값을 장기 성과의 연구 결론으로 해석하지 않는다. 연속 시장 이력의 같은 구간을 재표집한 횟수는 독립 경제 역사 개수가 아니다.', '']
                for filename in ['suite_results.json', 'comparisons.json', 'sensitivity.json', 'planned_runs.json', 'execution_plan.json']:
                    if (path/filename).exists():
                        dest = output/'evidence'/f'suite{index:02d}_{filename}'
                        dest.parent.mkdir(exist_ok=True)
                        shutil.copyfile(path/filename, dest)
                        sources.append(_ref(path/filename))
                        suite_sections += [f'[{filename}]({dest.relative_to(output).as_posix()})', '']
                for stat in sorted((path/'statistics').glob('*.json')):
                    dest = output/'evidence'/f'suite{index:02d}_statistics_{stat.name}'
                    shutil.copyfile(stat, dest)
                    sources.append(_ref(stat))
                    st = _read(stat)
                    comparisons = st.get('comparisons', [])
                    suite_sections += [f'[{stat.stem} 통계 원자료·실제 표본 수·정의 불가 사유]({dest.relative_to(output).as_posix()}) — 비교 기록 {len(comparisons)}개. 표본 수는 각 기록의 retained months/valid controls를 따른다.', '']
        lines += [_table(inventory, ['역할', '버전', '상태', '범위', '개월', '원본']), '']
        if bundle and any('full-baseline-preflight-v1' in str(path) for path, _, _ in runs):
            lines += ['생성 시점의 소스와 계산 당시 버전은 별개다. T07의 실제 전체 기본 결과는 후보 1541930에서 profile별 239개월·50전략을 수행한 증거다. 이후 local getter 재사용, public vintage, 고정 λ 후보 grid, suite 연결 수정이 들어갔다. T08 ac069ce의 38자식 smoke와 8자식 3개월 연속 진단은 그 연결을 확인하며 T07 전체 결과를 최신 코드 인증으로 바꾸지 않는다.', '']
        lines += ['## 대조군·통계·민감도', '',
            '무작위 대조는 군집 개수를 유지하면서 소속을 섞고 중심·전이·예측을 다시 계산한다. 레짐 이름만 바꾸는 것은 대조군이 아니다. 주 비교는 대조 seed별 성과 지표의 평균이며 수익을 먼저 평균한 포트폴리오와 다르다. 블록 재표집은 같은 달들을 양쪽 전략에 적용한다. Holm 보정과 정의 불가 사유를 기록하지만 짧은 진단의 유의성을 연구 결론으로 사용하지 않는다.', ''] + suite_sections
        has_context = any('T06' in ref['source'] for ref in copied) and any('outlier-exclusion-analysis' in ref['source'] for ref in copied)
        if has_context:
            lines += ['## 전표본 해석과 코로나 제외 사후 진단', '',
                '이 절은 거래 성과와 분리된 전체 과거표본의 해석이다. 기존 T06 자료를 재사용하며 GMM이나 elbow를 재적합하지 않는다. 고정 2023-02 자료의 1959-12~2023-01 758개월·원변수126개·선택102개·PCA52개(분산 약95.16%)이며 논문의746개월·127개·61성분과 다르다. 정상 군집은 r=5, GMM은6성분이며 elbow1~10과 전이/이탈조건부행렬(대각0)은 저장된 진단이다.', '',
                '작성자는 극단 월 하나가 R0를 독점하는 것을 막으려 제외했다고 설명했다. README에는4월, 과거 complete-case 실제 처리에는4·5월 삭제가 관측됐다. 제외 의도와 달력 압축 오류는 다른 문제다. 아래 재적합은 연속 원시 달력에서 t-code를 먼저 계산하고 fit행만 명시적으로 제외했으며 제외 사이를 한 달 전이로 연결하지 않았다.', '',
                '| 제외 거시월 | fit 월수 | R0 월수 | PCA | R0 중 NBER | NBER 중 R0 |',
                '| --- | --- | --- | --- | --- | --- |',
                '| 없음 |758|1|52|1/1|1/95|',
                '|2020-03·04|756|304|55|88/304|88/93|',
                '|2020-04·05|756|293|55|89/293|89/94|', '',
                '극단 월을 빼면 R0가 늘어나는 관측은 설명을 뒷받침한다. 다만 NBER 침체 포착 비율이 높아져도 R0의 약70%는 비침체월이다. 이 표는 전체표본 사후 분류이며 rolling 예측 개선을 측정하지 않았다. 기본 거래 실험은 모든 평가월과48연속개월을 유지한다.', '']
        for reference in copied:
            link = reference['path']
            lines += [f'[{reference["role"]}]({link})', '']
            if link.lower().endswith('.png'):
                lines += [f'![{reference["role"]}]({link})', '']
        unrun = data.get('unrun', []) if bundle else ['입력 실행 이외의 전체 논문 실험']
        lines += ['## 한계와 미실행 범위', '', '실행하지 않은 항목은 **NOT_RUN_FULL**로 남긴다.', '']
        lines += [f'- NOT_RUN_FULL: {item}' for item in unrun]
        lines += ['', '- WRDS 동일 가격과 실제 일별 공개시각은 확보하지 못했다. 공개 ETF 가격과 vintage/lag 계약을 쓴다.',
            '- 기본 비용0과 BL prior·공분산 shrinkage·fallback·LOO 표본 가정은 원문이 완전히 정하지 않은 구현 선택을 포함한다.',
            '- 새 Sortino는 전체월 downside RMS를 사용하며 과거 코드의 음수월 표준편차와 다르다. Sharpe는 월 초과수익의 표본표준편차에 √12를 적용한다.',
            '- 역사 감사 기록(78검사·46PASS·29FAIL·3SKIP)은 그 당시 코드의 기록이며 수정하거나 소급해 PASS로 바꾸지 않는다.', '',
            '## 원문 및 발견 대응', '', '[기계 판독 가능한 원문 대응](traceability.json) · [F01~F25](findings.json) · [증거 및 해시](evidence_index.json)', '',
            _table(trace, ['id', 'meaning', 'execution_scope', 'difference']), '',
            _table(findings, ['id', 'description', 'disposition', 'execution_scope']), '']
        (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
        manifest.update(status='succeeded', finished_at=datetime.now(timezone.utc).isoformat(),
            artifacts=[{'path':p.relative_to(output).as_posix(), 'sha256':sha256_file(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'report_manifest.json'])
        publish()
        return {'status': 'succeeded', 'path': str(output), 'manifest': str(output/'report_manifest.json')}
    except Exception as exc:
        manifest.update(status='failed', finished_at=datetime.now(timezone.utc).isoformat(), error=str(exc))
        publish()
        raise
