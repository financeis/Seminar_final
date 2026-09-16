import json
from pathlib import Path

import pytest

from regime_alloc.reporting.report import generate_report
from regime_alloc.contracts import ResearchError
from regime_alloc.contracts import sha256_file
from regime_alloc.reporting.verification import verify_run


def test_report_is_immutable_and_labels_unrun(tmp_path):
    bundle = tmp_path / 'bundle.json'
    bundle.write_text(json.dumps({'kind': 'research_evidence_bundle', 'runs': [],
        'references': [], 'unrun': ['100 controls over full period']}), encoding='utf-8')
    result = generate_report(bundle, tmp_path / 'report')
    out = Path(result['path'])
    assert result['status'] == 'succeeded'
    assert 'NOT_RUN_FULL' in (out / 'report.md').read_text(encoding='utf-8')
    assert len(json.loads((out / 'traceability.json').read_text(encoding='utf-8'))['items']) == 39
    assert len(json.loads((out / 'findings.json').read_text(encoding='utf-8'))['items']) == 25
    with pytest.raises(ResearchError):
        generate_report(bundle, out)
    assert json.loads(bundle.read_text())['runs'] == []
    trace = json.loads((out / 'traceability.json').read_text(encoding='utf-8'))
    trace['items'][0]['references'].append({'path': 'missing-evidence.json'})
    (out / 'traceability.json').write_text(json.dumps(trace), encoding='utf-8')
    manifest_path = out / 'report_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['artifacts']:
        if row['path'] == 'traceability.json':
            row['sha256'] = sha256_file(out / row['path'])
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    result = verify_run(bundle, report=out)
    assert next(c for c in result['checks'] if c['name'] == 'report_evidence')['status'] == 'fail'
    assert any('missing-evidence' in e['reason'] for e in result['errors'])
