"""Run all four audit domains sequentially, offline, preserving diagnostic FAILs.

Exit 0: completed diagnosis (FAIL findings are allowed).
Exit 2: harness/domain execution error, invalid result schema, or source hash drift.
Historical per-domain CLI execution files are not replaced by fabricated runs;
run_summary.json is the authoritative record for this integrated invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import io
from pathlib import Path
import sys
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits
from audit_tests.common import record, sha256, write_results
from audit_tests.metrics_support import environment, install_network_guard, original_hashes, write_json

DOMAINS = ('data_timing', 'regimes', 'forecasting', 'metrics_integration')
STATUSES = ('PASS', 'FAIL', 'ERROR', 'SKIP')


def counts_for(rows):
    counts = Counter(row['status'] for row in rows)
    return {status: counts[status] for status in STATUSES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reports/paper_audit/evidence'))
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    summary = {'started_at_utc': started, 'command': getattr(sys, 'orig_argv', [sys.executable, *sys.argv]),
               'cwd': str(Path.cwd()), 'repo': str(repo), 'output': str(output), 'environment': environment(),
               'domain_order': DOMAINS, 'domain_records': {}, 'domain_executions': [],
               'semantics': 'PASS/FAIL are property verdicts; ERROR is an execution/harness error; SKIP is unresolved/unavailable evidence. Exit 0 permits FAIL.',
               'historical_records': 'Per-domain execution.json and verification.json from earlier standalone runs remain historical. This summary records the actual current invocation.',
               'isolation': 'Sequential domains; networking denied; numerical BLAS thread limit 1; original scripts write only inside TemporaryDirectory.'}
    code, rows, seen, stdout, stderr = 0, [], set(), io.StringIO(), io.StringIO()
    install_network_guard()
    try:
        summary['source_hashes_before'] = original_hashes(repo)
        summary['audit_source_sha256'] = {str(path.relative_to(repo)): sha256(path) for path in sorted((repo / 'audit_tests').glob('*.py'))}
        if not summary['source_hashes_before']['matches_baseline']:
            raise RuntimeError('Original source/data files do not match the recorded 18-file baseline before execution')
        for domain in DOMAINS:
            domain_out, domain_err = io.StringIO(), io.StringIO()
            domain_started = datetime.now(timezone.utc).isoformat()
            domain_rows, domain_code = [], 0
            with redirect_stdout(domain_out), redirect_stderr(domain_err), threadpool_limits(limits=1):
                try:
                    module = importlib.import_module('audit_tests.test_' + domain)
                    returned = module.run(repo, output)
                    if not isinstance(returned, list):
                        raise TypeError(f'{domain}.run() returned {type(returned).__name__}, expected list')
                    # Validate before extending the aggregate, but retain valid prefix
                    # records if a later record fails validation.
                    for item in returned:
                        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or item.get('status') not in STATUSES:
                            raise ValueError('Invalid result record: ' + repr(item))
                        if item['id'] in seen:
                            raise ValueError('Duplicate global check ID: ' + item['id'])
                        seen.add(item['id'])
                        domain_rows.append(item)
                except Exception as exc:
                    domain_code, code = 2, 2
                    traceback.print_exc()
                    failure = record('RUN-' + domain.upper() + '-ERROR', 'Domain did not complete normally', False,
                                     'Domain run completes and returns unique valid records',
                                     {'exception': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()},
                                     inputs={'domain': domain, 'repo': str(repo)}, oracle='Runner call/return and result validation', status='ERROR')
                    domain_rows.append(failure)
                    # Keep all prior domain evidence and separately name the error;
                    # do not overwrite partial diagnostics with a fabricated success.
                    write_json(output / domain / 'runner_error.json', failure)
            rows.extend(domain_rows)
            domain_counts = counts_for(domain_rows)
            summary['domain_records'][domain] = [{'id': item['id'], 'status': item['status']} for item in domain_rows]
            summary['domain_executions'].append({'domain': domain, 'call': f'audit_tests.test_{domain}.run(repo, output)',
                'started_at_utc': domain_started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                'exit_code': domain_code, 'counts': domain_counts, 'stdout': domain_out.getvalue(), 'stderr': domain_err.getvalue(),
                'result_file': str(output / domain / 'results.json'),
                'result_sha256': sha256(output / domain / 'results.json') if (output / domain / 'results.json').exists() else None})
            stdout.write(f'{domain}: ' + ' '.join(f'{status}={domain_counts[status]}' for status in STATUSES) + '\n')
            if domain_code:
                stderr.write(f'{domain}: ERROR; see domain_executions stderr/traceback in run_summary.json\n')
        summary['source_hashes_after'] = original_hashes(repo)
        summary['source_unchanged'] = summary['source_hashes_before'] == summary['source_hashes_after']
        if not summary['source_unchanged'] or not summary['source_hashes_after']['matches_baseline']:
            raise RuntimeError('Original source/data files changed during audit execution')
    except Exception as exc:
        code = 2
        stderr.write(traceback.format_exc())
        failure = record('RUN-ERROR', 'Audit runner completion failure', False, 'Complete all four domains and preserve original files',
                         {'exception': type(exc).__name__, 'message': str(exc)}, oracle='Runner completion and original hashes', status='ERROR')
        rows.append(failure)
        summary['runner_error'] = failure
        try:
            summary['source_hashes_after'] = original_hashes(repo)
            summary['source_unchanged'] = summary.get('source_hashes_before') == summary['source_hashes_after']
        except Exception:
            summary['source_hash_after_error'] = traceback.format_exc()
    counts = counts_for(rows)
    if counts['ERROR']:
        code = 2
    stdout.write('TOTAL: ' + ' '.join(f'{status}={counts[status]}' for status in STATUSES) + f' EXIT={code}\n')
    summary.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), exit_code=code, counts=counts, total_records=len(rows),
                   stdout=stdout.getvalue(), stderr=stderr.getvalue(), unique_check_ids=len({row['id'] for row in rows}) == len(rows),
                   evidence_files=[{'path': str(path.relative_to(output)), 'sha256': sha256(path)}
                                   for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'run_summary.json'])
    write_json(output / 'run_summary.json', summary)
    print(stdout.getvalue(), end='')
    print(stderr.getvalue(), end='', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
