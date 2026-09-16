"""Bounded synthetic checks; no market download, model fitting, or suite rerun."""
from dataclasses import asdict
import json

import pandas as pd
import pytest

from regime_alloc.backtest.accounting import account_month
from regime_alloc.backtest.artifacts import RunWriter
from regime_alloc.backtest.metrics import compute_metrics
from regime_alloc.config import PortfolioSettings
from regime_alloc.contracts import TICKERS, sha256_file
from regime_alloc.reporting.verification import verify_run


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.delenv('REGIME_DATA_ROOT', raising=False)
    root = tmp_path/'run'
    months = ['2020-01', '2020-02']
    settings = PortfolioSettings()
    config = {'portfolio': asdict(settings), 'evaluated_months': months, 'strategy_ids': ['spy'], 'smoke': False}
    writer = RunWriter(root, {'run_id': 'fixture', 'actual_period': {'evaluated_months': months}, 'strategy_ids': ['spy']})
    writer.json('effective_config.json', config)
    dates = ['2020-01-02T16:00:00-05:00', '2020-02-03T16:00:00-05:00', '2020-03-02T16:00:00-05:00']
    accounts = []
    for i, month in enumerate(months):
        r = pd.Series(0., index=TICKERS)
        r['SPY'] = [-.1, .2][i]
        target = pd.Series(0., index=TICKERS)
        target['SPY'] = 1.
        accounts.append(account_month(target, r, holding_month=month, decision_at=dates[i].replace('16:00', '09:00'), execution_at=dates[i], target_end=dates[i+1], previous=accounts[-1] if accounts else None))
    writer.jsonl('asset_returns.jsonl', [{'holding_month': a.holding_month, 'returns': a.asset_returns.to_dict()} for a in accounts])
    metrics = compute_metrics(months, [a.net_return for a in accounts], **{key: [getattr(a, key) for a in accounts] for key in ('turnover','transaction_cost','borrow_cost','financing_cost','gross_exposure')})
    for prefix in ('', 'scaled/'):
        writer.csv(prefix+'returns.csv', [a.to_record('fixture', 'spy') for a in accounts])
        writer.csv(prefix+'weights.csv', [r for a in accounts for r in a.weight_records('fixture', 'spy')])
        writer.csv(prefix+'metrics.csv', [metrics.to_record('fixture', 'spy')])
    writer.succeed()
    return root


def test_valid_bounded_run_and_relative_bundle(evidence, tmp_path):
    bundle = tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'kind': 'research_evidence_bundle', 'runs': [{'path': 'run', 'role': 'synthetic_validation', 'execution_scope': 'two months'}], 'references': [], 'unrun': ['full suite']}))
    result = verify_run(bundle)
    assert result['status'] == 'succeeded', result
    assert all(c['status'] == 'pass' for c in result['checks'])
    assert result['scope'][0]['unrun'] == ['full suite']
    json.dumps(result, allow_nan=False)


def test_corrupt_financial_value_even_after_rehash(evidence):
    path = evidence/'returns.csv'
    table = pd.read_csv(path)
    table.loc[0, 'gross_return'] = .5
    table.to_csv(path, index=False)
    manifest_path = evidence/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    next(x for x in manifest['artifacts'] if x['path'] == 'returns.csv')['sha256'] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    result = verify_run(evidence)
    assert result['status'] == 'failed'
    assert any(c['name'] == 'artifact_inventory' and c['status'] == 'pass' for c in result['checks'])
    assert any('gross_return' in e['reason'] for e in result['errors'])


def test_missing_inventory_fails(evidence):
    path = evidence/'run_manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['artifacts']
    path.write_text(json.dumps(manifest))
    result = verify_run(evidence)
    assert result['status'] == 'failed'
    assert any(e['error_code'] == 'missing_data' for e in result['errors'])
