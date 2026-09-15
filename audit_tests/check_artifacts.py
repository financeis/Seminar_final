"""Validate the paper-audit contract. Exit 0: consistent; exit 1: missing/inconsistent.

This checks diagnostic integrity, not whether the original implementation is correct.
The integrated runner must be rerun after changes to audit_tests/*.py. Historical
standalone execution files are hashed as archived evidence, not as current runs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_tests.common import PAPER_SHA256, resolve_paper

BASELINE = '9439107b31f20c4b731ec1dc29f13c178beb311d'
DOMAINS = ('data_timing', 'regimes', 'forecasting', 'metrics_integration')
STATUSES = ('PASS', 'FAIL', 'ERROR', 'SKIP')
METHODS = {f'EQ{i:02d}' for i in range(1, 20)} | {
    'ALG1', 'DATA', 'REGIME_ANALYSIS', 'CONDITIONAL_TRANSITION', 'MATCHING',
    'MODELS', 'SIZING', 'ETF_DATA', 'ROLLING', 'METRICS', 'CONTROLS',
    'BENCHMARKS', 'VOL_SCALING'}
VERDICTS = {'일치', '부분일치', '불일치', '미구현', '논문모호', '검증불가'}
CATEGORIES = {'implementation_error', 'paper_deviation', 'missing_method',
              'paper_ambiguity', 'statistical_validity', 'reproducibility'}
SEVERITIES = {'critical', 'high', 'medium', 'low', 'info'}
CONFIDENCES = {'confirmed', 'supported', 'unresolved'}
FINDING_FIELDS = {'id', 'title', 'category', 'severity', 'confidence', 'paper_refs',
                  'code_refs', 'expected', 'observed', 'impact', 'evidence_ids',
                  'recommendation', 'limitations'}
RESULT_FIELDS = {'id', 'title', 'status', 'expected', 'observed', 'source_refs',
                 'inputs', 'oracle', 'tolerance', 'notes'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_rows(text):
    return [[s.strip() for s in line.strip().strip('|').split('|')]
            for line in text.splitlines() if line.strip().startswith('|')]


def counts(rows):
    c = Counter(row['status'] for row in rows)
    return {s: c[s] for s in STATUSES}


def validate(repo: Path, report_dir: Path):
    """Return errors and measured facts; read-only and reusable for corruption checks."""
    repo, report_dir = repo.resolve(), report_dir.resolve()
    errors, facts = [], {}

    def need(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)

    def read_json(path):
        try:
            return json.loads(path.read_text('utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f'{path.name}: cannot read JSON: {exc}')
            return None

    def read_text(path):
        try:
            return path.read_text('utf-8')
        except (OSError, UnicodeError) as exc:
            errors.append(f'{path.name}: cannot read text: {exc}')
            return ''

    evdir = report_dir / 'evidence'
    findings = read_json(report_dir / 'findings.json')
    env = read_json(report_dir / 'environment.json')
    inventory = read_json(evdir / 'inventory/manifest.json')
    coverage = read_json(evdir / 'paper/coverage.json')
    summary = read_json(evdir / 'run_summary.json')
    report = read_text(report_dir / 'audit_report.md')
    trace = read_text(report_dir / 'traceability.md')
    method = read_text(report_dir / 'methodology.md')
    index = read_text(report_dir / 'evidence_index.md')
    for key, value in [('findings', findings), ('environment', env), ('inventory', inventory),
                       ('coverage', coverage), ('summary', summary)]:
        need(isinstance(value, list if key == 'findings' else dict), f'{key}: wrong top-level schema')
    if errors:
        return errors, facts

    baseline_hashes = env.get('original_file_sha256', {})
    need(env.get('baseline_commit') == BASELINE, 'environment: baseline commit changed')
    need(isinstance(baseline_hashes, dict) and len(baseline_hashes) == 18,
         'environment: expected original 18-file hash manifest')
    actual_hashes = {}
    for rel, expected_hash in baseline_hashes.items():
        path = (repo / rel).resolve()
        if not need(path.is_relative_to(repo) and path.is_file(), f'baseline file missing/outside repo: {rel}'):
            continue
        actual_hashes[rel] = digest(path)
        need(actual_hashes[rel] == expected_hash, f'original file hash mismatch: {rel}')
    # Independent Git check prevents a modified file + modified manifest from
    # self-authenticating. LF/CRLF worktree conversion is handled by Git itself.
    try:
        proc = subprocess.run(['git', 'diff', '--name-only', BASELINE, '--', *baseline_hashes],
                              cwd=repo, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=30)
        need(proc.returncode == 0, 'baseline Git provenance unavailable: ' + proc.stderr.strip())
        need(not proc.stdout.strip(), 'Git content differs from fixed baseline: ' + proc.stdout.strip())
        facts['baseline_git_exit'] = proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f'baseline Git provenance unavailable: {exc}')
    facts['original_file_count'] = len(actual_hashes)

    try:
        paper_path = resolve_paper(repo)
        from pypdf import PdfReader
        pages = len(PdfReader(str(paper_path)).pages)
        need(pages == 26, 'designated paper must contain 26 pages')
        need(digest(paper_path) == PAPER_SHA256, 'designated paper hash mismatch')
        facts['designated_paper_sha256'] = digest(paper_path)
    except Exception as exc:
        errors.append(f'designated paper unavailable: {exc}')
        paper_path, pages = None, 26
    need(env.get('paper', {}).get('sha256') == PAPER_SHA256, 'environment: wrong paper hash')
    need(coverage.get('id') == 'PAPER-COVERAGE' and coverage.get('status') == 'PASS', 'paper coverage identity/status invalid')
    need(coverage.get('paper_sha256') == PAPER_SHA256, 'paper coverage: wrong paper hash')
    for field in ('read_pages', 'visually_scanned_pages'):
        need(coverage.get(field) == list(range(1, 27)), 'paper coverage: incomplete ' + field)
    need(coverage.get('equations') == list(range(1, 20)), 'paper coverage: equations 1–19 incomplete')
    need(coverage.get('algorithm') == 1, 'paper coverage: Algorithm 1 missing')
    for field, size in [('figure_method_map', 13), ('table_method_map', 6)]:
        mapping = coverage.get(field, {})
        need(set(mapping) == {str(i) for i in range(1, size + 1)}, f'paper coverage: incomplete {field}')
        for key, values in mapping.items():
            need(isinstance(values, list) and bool(values) and set(values) <= METHODS, f'{field} {key}: invalid method references')
    supplement = coverage.get('supplementary_appendix', {})
    need(supplement.get('sha256') == baseline_hashes.get('FRED-MD_updated_appendix.pdf'), 'appendix hash mismatch')
    tcode = supplement.get('csv_tcode_comparison', {})
    need(tcode.get('status') == 'PASS' and tcode.get('matched_variables') == 121 and tcode.get('mismatches') == {}, 'appendix t-code comparison incomplete')

    source_lines, source_defs = {}, {}
    for rel in baseline_hashes:
        if rel.endswith(('.py', '.md', '.csv', '.txt')):
            text = read_text(repo / rel)
            source_lines[rel] = len(text.splitlines())
            if rel.endswith('.py'):
                source_defs[rel] = [n for n in ast.walk(ast.parse(text))
                                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    python_files = inventory.get('python_files', [])
    need(inventory.get('id') == 'INVENTORY' and inventory.get('status') == 'PASS', 'inventory identity/status invalid')
    need(inventory.get('baseline_commit') == BASELINE, 'inventory baseline mismatch')
    need({x.get('path') for x in python_files} == {x for x in baseline_hashes if x.endswith('.py')}
         and len(python_files) == 11, 'inventory must cover original 11 Python files exactly')
    for entry in python_files:
        rel = entry.get('path')
        need(entry.get('sha256') == actual_hashes.get(rel), f'inventory hash mismatch: {rel}')
        need(entry.get('lines') == source_lines.get(rel), f'inventory line count mismatch: {rel}')

    def check_ref(ref, owner, require_hash):
        if not need(isinstance(ref, dict) and isinstance(ref.get('path'), str), f'{owner}: invalid source reference'):
            return
        rel = ref['path']
        # An archived absolute PDF source path can be relocated by the explicit
        # resolver; it is identified by content, never by a same-name substitute.
        if ref.get('sha256') == PAPER_SHA256:
            need(paper_path is not None, f'{owner}: designated PDF unresolved')
        else:
            need(rel in baseline_hashes, f'{owner}: reference outside original manifest: {rel}')
            if require_hash:
                need(ref.get('sha256') == actual_hashes.get(rel), f'{owner}: source hash mismatch {rel}')
        if 'line_start' in ref or 'line_end' in ref:
            a, b = ref.get('line_start'), ref.get('line_end')
            valid = isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and 1 <= a <= b <= source_lines.get(rel, 0)
            need(valid, f'{owner}: invalid source line range {rel}:{a}–{b}')
            # Descriptive labels such as original_*_AST are block names, not
            # Python symbols. Exact existing symbols must overlap their AST range.
            definitions = [n for n in source_defs.get(rel, []) if n.name == ref.get('symbol')]
            if definitions and valid:
                need(any(a <= n.end_lineno and b >= n.lineno for n in definitions), f'{owner}: source symbol outside range {rel}:{ref.get("symbol")}')
        for page in ref.get('pages', []):
            need(isinstance(page, int) and 1 <= page <= pages, f'{owner}: invalid PDF page')

    all_rows, domain_rows = [], {}
    for domain in DOMAINS:
        rows = read_json(evdir / domain / 'results.json')
        if not need(isinstance(rows, list) and bool(rows), f'{domain}: empty or invalid results'):
            continue
        domain_rows[domain] = rows
        for row in rows:
            if not need(isinstance(row, dict) and RESULT_FIELDS <= set(row), f'{domain}: incomplete result schema'):
                continue
            eid = row.get('id')
            need(isinstance(eid, str) and bool(eid), f'{domain}: invalid evidence ID')
            need(row['status'] in STATUSES, f'{eid}: invalid result status')
            need(bool(row['title']) and bool(row['oracle']), f'{eid}: title or independent oracle missing')
            tol = row['tolerance']
            if isinstance(tol, dict):
                for k in ('atol', 'rtol'):
                    if k in tol:
                        need(isinstance(tol[k], (float, int)) and math.isfinite(tol[k]) and tol[k] >= 0, f'{eid}: invalid {k}')
            elif tol is not None:
                need(isinstance(tol, (int, float)) and math.isfinite(tol) and tol >= 0, f'{eid}: invalid tolerance')
            need(isinstance(row['source_refs'], list), f'{eid}: source_refs must be a list')
            for ref in row['source_refs']:
                check_ref(ref, eid, True)
            if not row['inputs']:
                need(bool(row['source_refs']) or 'self-check' in row['title'].lower() or '자체' in row['title'], f'{eid}: neither inputs nor source/self-check basis')
            if row['status'] in ('FAIL', 'SKIP', 'ERROR'):
                need(bool(row['observed']) and bool(row['expected']), f'{eid}: non-PASS lacks explanation')
            all_rows.append(row)
        need(any(r.get('status') in ('PASS', 'FAIL') for r in rows), f'{domain}: no executed core checks')
    ids = [r['id'] for r in all_rows]
    need(len(ids) == len(set(ids)), 'duplicate evidence IDs')
    ev_ids = set(ids) | {'INVENTORY', 'PAPER-COVERAGE'}
    if len(domain_rows) != len(DOMAINS):
        return errors, facts
    actual_counts = counts(all_rows)
    need(summary.get('counts') == actual_counts and summary.get('total_records') == len(all_rows), 'summary counts differ from actual domain records')
    need(summary.get('exit_code') == 0 and actual_counts['ERROR'] == 0, 'integrated audit did not complete without execution ERROR')
    need(summary.get('unique_check_ids') is True, 'summary unique check assertion missing')
    need(summary.get('domain_order') == list(DOMAINS), 'summary domain order mismatch')
    for domain in DOMAINS:
        listed = [{'id': r['id'], 'status': r['status']} for r in domain_rows[domain]]
        need(summary.get('domain_records', {}).get(domain) == listed, f'summary {domain}: IDs/status differ')
        executions = [x for x in summary.get('domain_executions', []) if x.get('domain') == domain]
        if need(len(executions) == 1, f'summary {domain}: missing/duplicate execution'):
            execution = executions[0]
            need(execution.get('exit_code') == 0 and execution.get('counts') == counts(domain_rows[domain]), f'summary {domain}: execution counts/exit mismatch')
            need(execution.get('result_sha256') == digest(evdir/domain/'results.json'), f'summary {domain}: result hash mismatch')
    for phase in ('before', 'after'):
        hashes = summary.get('source_hashes_' + phase, {})
        need(hashes.get('baseline') == baseline_hashes and hashes.get('actual') == actual_hashes
             and hashes.get('matches_baseline') is True and hashes.get('file_count') == 18,
             f'summary source hashes {phase} mismatch')
    need(summary.get('source_unchanged') is True, 'summary: source preservation not established')
    current_audit = {str(p.relative_to(repo)).replace('\\', '/'): digest(p) for p in sorted((repo/'audit_tests').glob('*.py'))}
    recorded_audit = {k.replace('\\', '/'): v for k,v in summary.get('audit_source_sha256', {}).items()}
    need(recorded_audit == current_audit, 'audit source hashes changed or checker absent: rerun run_audit.py before checking final artifacts')
    archived = summary.get('evidence_files', [])
    listed_files = {x['path'].replace('\\', '/'): x['sha256'] for x in archived}
    actual_files = {str(p.relative_to(evdir)).replace('\\', '/'): digest(p) for p in evdir.rglob('*')
                    if p.is_file() and p.name != 'run_summary.json'}
    need(len(archived) == len(listed_files), 'summary: duplicate evidence paths')
    need(listed_files == actual_files, 'summary: archived evidence file set or hash changed; rerun audit')
    need(bool(summary.get('command')) and bool(summary.get('started_at_utc')) and bool(summary.get('finished_at_utc')),
         'summary execution provenance missing')
    for package, version in summary.get('environment', {}).get('packages', {}).items():
        need(env.get('packages', {}).get(package) == version, f'environment: integrated package version differs for {package}')
    for line in read_text(repo/'audit_tests/requirements-audit.txt').splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            package, version = line.strip().split('==', 1)
            need(env.get('packages', {}).get(package) == version, f'environment: pinned package differs for {package}')
            try:
                need(importlib.metadata.version(package) == version, f'current runtime package differs for {package}')
            except importlib.metadata.PackageNotFoundError:
                errors.append(f'current runtime package missing: {package}')

    finding_ids, used = [], set()
    for finding in findings:
        if not need(isinstance(finding, dict) and FINDING_FIELDS <= set(finding), 'finding: missing required fields'):
            continue
        fid = finding['id']
        finding_ids.append(fid)
        need(isinstance(fid, str) and re.fullmatch(r'F\d{2,}', fid), 'finding: invalid ID')
        for field in ('title', 'expected', 'observed', 'impact', 'recommendation'):
            need(isinstance(finding[field], str) and bool(finding[field].strip()), f'{fid}: empty {field}')
        need(isinstance(finding['limitations'], list) and all(isinstance(s, str) and bool(s.strip()) for s in finding['limitations']),
             f'{fid}: limitations must be a list of nonempty strings (empty list allowed)')
        need(finding['category'] in CATEGORIES, f'{fid}: invalid category')
        need(finding['severity'] in SEVERITIES, f'{fid}: invalid severity')
        need(finding['confidence'] in CONFIDENCES, f'{fid}: invalid confidence')
        need(isinstance(finding['paper_refs'], list) and bool(finding['paper_refs']), f'{fid}: paper refs missing')
        for ref in finding['paper_refs']:
            need(isinstance(ref, dict) and {'page','section','equation'} <= set(ref), f'{fid}: invalid paper ref schema')
            if not isinstance(ref, dict):
                continue
            need(ref.get('page') is None or isinstance(ref.get('page'), int) and 1 <= ref['page'] <= 26, f'{fid}: invalid paper page')
            need(ref.get('equation') is None or isinstance(ref.get('equation'), int) and 1 <= ref['equation'] <= 19, f'{fid}: invalid equation')
            need(ref.get('section') is None or isinstance(ref.get('section'), str), f'{fid}: invalid section')
        need(isinstance(finding['code_refs'], list), f'{fid}: invalid code refs')
        for ref in finding['code_refs']:
            need({'path','symbol','line_start','line_end'} <= set(ref), f'{fid}: incomplete code ref schema')
            check_ref(ref, fid, False)
        if finding['category'] == 'missing_method':
            need(finding['code_refs'] == [], f'{fid}: absent code must use []')
            need(all(p['path'] in finding['observed'] for p in python_files), f'{fid}: absence claim lacks full 11-file inspection scope')
        refs_used = finding['evidence_ids']
        need(isinstance(refs_used, list) and bool(refs_used) and len(refs_used) == len(set(refs_used)), f'{fid}: missing/duplicate evidence IDs')
        need(set(refs_used) <= ev_ids, f'{fid}: dangling evidence IDs: {set(refs_used)-ev_ids}')
        used.update(refs_used)
    need(len(finding_ids) == len(set(finding_ids)), 'duplicate finding IDs')
    for row in all_rows:
        if row['status'] in ('FAIL','SKIP','ERROR'):
            need(row['id'] in used, f'{row["id"]}: non-PASS not linked to a finding')
    facts.update(counts=actual_counts, domain_check_count=len(all_rows), finding_count=len(findings), evidence_id_count=len(ev_ids))
    need(env.get('audit_scope', {}).get('domain_check_count') == len(all_rows) and env.get('audit_scope', {}).get('finding_count') == len(findings), 'environment audit scope count mismatch')
    need(env.get('audit_scope', {}).get('actual_etf_replication') is False, 'environment: actual ETF replication scope changed')

    report_table = [row for row in table_rows(report) if row and re.fullmatch(r'F\d{2,}',row[0])]
    expected_table = [[f[k] for k in ('id','title','category','severity','confidence')] for f in findings]
    need(report_table == expected_table, 'report finding table differs from JSON IDs/title/category/severity/confidence')
    for finding in findings:
        need(f"### {finding['id']} — {finding['title']}" in report, f'{finding["id"]}: detailed report section missing')
        for key in ('expected','observed','impact','recommendation'):
            need(finding[key] in report, f'{finding["id"]}: report/JSON {key} differs')
        for limitation in finding['limitations']:
            need(limitation in report, f'{finding["id"]}: report/JSON limitation differs')
    for domain, expected in [(d,counts(domain_rows[d])) for d in DOMAINS]+[('TOTAL',actual_counts)]:
        candidates = [row for row in table_rows(report) if row and row[0] == domain]
        need(candidates == [[domain, *[str(expected[s]) for s in STATUSES]]], f'report count row mismatch: {domain}')

    traces = [row for row in table_rows(trace) if row and row[0] in METHODS]
    need(len(traces) == len(METHODS) and {r[0] for r in traces} == METHODS, 'traceability: equation/algorithm/method coverage incomplete or duplicated')
    for row in traces:
        if not need(len(row) == 7 and all(row), f'traceability {row[0]}: incomplete row schema'):
            continue
        need(row[5] in VERDICTS, f'traceability {row[0]}: invalid verdict')
        need(bool(re.search(r'p{1,2}\.', row[1])), f'traceability {row[0]}: original page missing')
        references = re.findall(r'(?<![A-Z0-9_])(?:F\d{2,}|DATA-[A-Z0-9_a-z-]+|REG-\d{3}|FOR-\d{3}|MET-\d{3}|INVENTORY|PAPER-COVERAGE)(?![A-Z0-9_])', row[6])
        need(bool(references) and set(references) <= ev_ids | set(finding_ids), f'traceability {row[0]}: invalid evidence/finding references')
        if row[0].startswith('EQ'):
            need(f'| {row[0]} |' in method, f'methodology missing {row[0]}')
    index_rows = [row for row in table_rows(index) if row and row[0] in ev_ids]
    need(len(index_rows) == len(ev_ids) and {r[0] for r in index_rows} == ev_ids, 'evidence index: orphan or duplicate evidence IDs')
    status_lookup = {r['id']:r['status'] for r in all_rows} | {'INVENTORY':'PASS','PAPER-COVERAGE':'PASS'}
    for row in index_rows:
        if not need(len(row) == 5, f'index {row[0]}: invalid schema'):
            continue
        need(row[1] == status_lookup[row[0]], f'index {row[0]}: status mismatch')
        named_findings = set(re.findall(r'\bF\d{2,}\b', row[4]))
        expected_findings = {f['id'] for f in findings if row[0] in f['evidence_ids']}
        need(named_findings == expected_findings and bool(row[4]), f'index {row[0]}: usage/finding mismatch')
        need((report_dir / row[2].strip('`')).is_file(), f'index {row[0]}: source document missing')
    return errors, facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/paper_audit'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        errors, facts = validate(args.repo, args.report_dir)
    except Exception as exc:
        errors, facts = [f'Artifact schema/check execution failure: {type(exc).__name__}: {exc}'], {}
    result = {'status':'FAIL' if errors else 'PASS', 'exit_code':int(bool(errors)), 'errors':errors, 'facts':facts}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
