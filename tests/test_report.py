import json
from pathlib import Path

import pytest

from regime_alloc.reporting.report import generate_report, _curves
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


def test_smoke_points_and_corrupt_input_rejection(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    run = tmp_path/'smoke'
    run.mkdir()
    (run/'effective_config.json').write_text('{"smoke": true}', encoding='utf-8')
    (run/'returns.csv').write_text('holding_month,strategy_id,net_return,wealth\n2003-02,spy,-0.1,0.9\n2022-12,spy,0.2,1.2\n', encoding='utf-8')
    points = []
    original = Axes.scatter
    def capture(self, x, y, **kwargs):
        points.append(list(y))
        return original(self, x, y, **kwargs)
    def no_line(*args, **kwargs):
        raise AssertionError('smoke must not connect independent NAV observations')
    monkeypatch.setattr(Axes, 'scatter', capture)
    monkeypatch.setattr(Axes, 'plot', no_line)
    _curves(run, tmp_path, 'smoke')
    assert points == [[-.1, .2]]
    (run/'run_manifest.json').write_text(json.dumps({'status':'succeeded', 'artifacts':[
        {'path':'effective_config.json','sha256':'incorrect'}]}), encoding='utf-8')
    with pytest.raises(ResearchError, match='hash mismatch'):
        generate_report(run, tmp_path/'rejected-report')
    assert json.loads((tmp_path/'rejected-report/report_manifest.json').read_text())['status'] == 'failed'
