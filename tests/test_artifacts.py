import json
import pytest
from regime_alloc.backtest.artifacts import RunWriter, COLUMNS
from regime_alloc.contracts import ResearchError


def test_atomic_run_conflict_and_failed_manifest(tmp_path):
    path = tmp_path / 'run'
    writer = RunWriter(path, {'run_id': 'test'})
    assert json.loads((path / 'run_manifest.json').read_text())['status'] == 'running'
    with pytest.raises(ResearchError, match='run_conflict'):
        RunWriter(path, {'run_id': 'other'})
    writer.fail(RuntimeError('injected failure'))
    record = json.loads((path / 'run_manifest.json').read_text())
    assert record['status'] == 'failed' and 'injected failure' in record['errors'][0]['reason']
    assert record['errors'][0]['error_code'] == 'numerical_failure'


def test_explicit_columns_duplicate_and_nonfinite(tmp_path):
    writer = RunWriter(tmp_path / 'run', {'run_id': 'test'})
    row = dict(run_id='test', decision_month='2003-02', partition_id='p', regime_id='R0', current_probability=.5, next_probability=.5)
    with pytest.raises(ResearchError, match='duplicate'):
        writer.csv('probabilities.csv', [row, row])
    with pytest.raises(ResearchError, match='nonfinite'):
        writer.csv('probabilities.csv', [{**row, 'current_probability': float('inf')}])
    writer.csv('probabilities.csv', [dict(reversed(list(row.items())))])
    assert (writer.path / 'probabilities.csv').read_text().splitlines()[0] == ','.join(COLUMNS['probabilities.csv'])
    writer.succeed()
    record = json.loads((writer.path / 'run_manifest.json').read_text())
    assert record['status'] == 'succeeded' and len(record['artifacts']) == 1
