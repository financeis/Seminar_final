"""Read-only evidence verification, independent of portfolio/accounting implementations.

Success covers the supplied evidence, not unrun experiments or model correctness.
Raw-price verification is available when REGIME_DATA_ROOT (or a suite data root)
is supplied; otherwise the saved asset panel is the declared accounting input.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..contracts import ErrorCode, ResearchError, TICKERS, sha256_file
from ..data.io import confined_path, read_json


def _require(ok, reason, code=ErrorCode.VERIFICATION_FAILED):
    if not ok:
        raise ResearchError(code, reason)


def _close(actual, expected, label):
    if expected is None or pd.isna(expected):
        _require(pd.isna(actual), f'{label}: expected undefined')
    else:
        _require(np.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10), f'{label}: numeric mismatch')


def _inventory(root, manifest, filename):
    rows = manifest.get('artifacts')
    _require(isinstance(rows, list) and bool(rows), 'missing artifact inventory', ErrorCode.MISSING_DATA)
    names = [r['path'] for r in rows]
    _require(len(names) == len(set(names)), 'duplicate artifact inventory')
    for row in rows:
        path = confined_path(root, row['path'])
        _require(path.is_file(), f'missing artifact: {path}', ErrorCode.MISSING_DATA)
        _require(sha256_file(path) == row['sha256'], f'artifact hash mismatch: {path}', ErrorCode.HASH_MISMATCH)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != filename and not p.name.startswith('.')}
    _require(set(names) == actual, 'artifact inventory differs from files')


def _references(root, rows):
    _require(isinstance(rows, list), 'references must be a list')
    for row in rows:
        path = (root / row['path']).resolve()
        _require(path.is_file(), f'missing reference: {path}', ErrorCode.MISSING_DATA)
        if 'sha256' in row:
            _require(sha256_file(path) == row['sha256'], f'reference hash mismatch: {path}', ErrorCode.HASH_MISMATCH)


def _financial(root, manifest, data_root, scope):
    config = read_json(root / 'effective_config.json')
    months = manifest['actual_period']['evaluated_months']
    strategies = manifest['strategy_ids']
    _require(months and len(set(months)) == len(months) and months == sorted(months), 'invalid evaluated months')
    _require(strategies and len(set(strategies)) == len(strategies), 'invalid strategies')
    _require(config['evaluated_months'] == months and config['strategy_ids'] == strategies, 'config/manifest panel mismatch')
    smoke = config.get('smoke', False)
    if not smoke:
        _require(months == list(pd.period_range(months[0], months[-1], freq='M').astype(str)), 'continuous run has month gaps')
    asset_rows = [json.loads(line) for line in (root / 'asset_returns.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assets = {row['holding_month']: row['returns'] for row in asset_rows}
    _require(len(assets) == len(asset_rows), 'duplicate asset-return months')
    prices = None
    if data_root:
        from ..data import load_prices
        prices = load_prices(data_root, manifest['dataset_ids']['yahoo']).set_index(['session', 'ticker'])
        scope.append({'path': str(root), 'raw_prices': 'verified against pinned dataset'})
    else:
        scope.append({'path': str(root), 'raw_prices': 'not checked: data root unavailable; accounting uses saved asset returns'})
    settings = config['portfolio']
    forecasts = pd.read_csv(root/'forecasts.csv') if any(s not in ('spy', 'ew') for s in strategies) else None
    probabilities = pd.read_csv(root/'probabilities.csv') if any('_mx_' in s for s in strategies) else None
    for prefix in ('', 'scaled/'):
        returns = pd.read_csv(root / f'{prefix}returns.csv')
        weights = pd.read_csv(root / f'{prefix}weights.csv')
        metrics = pd.read_csv(root / f'{prefix}metrics.csv')
        expected = {(m, s) for m in months for s in strategies}
        _require(len(returns) == len(expected) and set(zip(returns.holding_month, returns.strategy_id)) == expected, 'return month/strategy panel mismatch')
        expected_weights = {(m, s, t) for m, s in expected for t in (*TICKERS, 'CASH')}
        _require(len(weights) == len(expected_weights) and set(zip(weights.decision_month, weights.strategy_id, weights.ticker)) == expected_weights, 'weight panel mismatch')
        _require(len(metrics) == len(strategies) and set(metrics.strategy_id) == set(strategies), 'metric strategy panel mismatch')
        for frame in (returns, weights, metrics):
            _require(set(frame.run_id) == {manifest['run_id']}, 'table run identity mismatch')
        for strategy in strategies:
            series = returns[returns.strategy_id == strategy].sort_values('holding_month')
            previous = np.r_[np.zeros(len(TICKERS)), 1.]
            wealth = peak = 1.
            for row in series.itertuples():
                if smoke:
                    previous = np.r_[np.zeros(len(TICKERS)), 1.]
                    wealth = peak = 1.
                table = weights[(weights.strategy_id == strategy) & (weights.decision_month == row.holding_month)].set_index('ticker').loc[[*TICKERS, 'CASH']]
                target = table.target_weight.to_numpy()
                base = table.base_weight.to_numpy()
                expected_base = np.zeros(len(TICKERS))
                if strategy == 'spy':
                    expected_base[0] = 1.
                elif strategy == 'ew':
                    expected_base[:] = 1/len(TICKERS)
                else:
                    model, sizing, count = strategy.split('_')
                    count = int(count)
                    selected = forecasts[(forecasts.model == model) & (forecasts.decision_month == row.holding_month)]
                    _require(len(selected) == len(TICKERS) and set(selected.ticker) == set(TICKERS), 'forecast ticker panel mismatch')
                    scores = selected.set_index('ticker').loc[list(TICKERS), 'value'].to_numpy()
                    _require(np.isfinite(scores).all(), 'nonfinite forecast scores')
                    if sizing == 'mx':
                        p = probabilities[probabilities.decision_month == row.holding_month].sort_values('regime_id')
                        _require(len(p) == 6 and set(p.regime_id) == {f'R{i}' for i in range(6)}, 'mix regime panel mismatch')
                        sizing = 'los' if p.loc[p.next_probability.idxmax(), 'regime_id'] == 'R0' else 'lo'
                    positive = sorted((i for i, v in enumerate(scores) if v > 0), key=lambda i: (-scores[i], i))[:count]
                    negative = sorted((i for i, v in enumerate(scores) if v < 0), key=lambda i: (scores[i], i))[:count]
                    chosen = positive if sizing == 'lo' else positive + negative
                    if sizing == 'los':
                        chosen = sorted(range(len(TICKERS)), key=lambda i: (-abs(scores[i]), i))[:count]
                    denominator = sum(abs(scores[i]) for i in chosen)
                    if denominator:
                        expected_base[chosen] = scores[chosen]/denominator
                for actual, expected_weight in zip(base[:-1], expected_base):
                    _close(actual, expected_weight, 'forecast-derived base weight')
                if not prefix:
                    for actual, expected_weight in zip(target, base):
                        _close(actual, expected_weight, 'unscaled target equals base')
                _close(target.sum(), 1, 'target conservation')
                _close(base.sum(), 1, 'base conservation')
                _require(np.abs(base[:-1]).sum() <= 1+1e-10, 'base gross exceeds one')
                _require(np.abs(target[:-1]).sum() <= settings['gross_cap']+1e-10, 'gross cap exceeded')
                for a, b in zip(table.pretrade_weight, previous):
                    _close(a, b, 'pretrade drift')
                r = np.array([assets[row.holding_month][t] for t in TICKERS])
                _require(np.isfinite(r).all() and (r >= -1).all(), 'invalid asset returns')
                start, end, decision = map(pd.Timestamp, (row.execution_at, row.target_end, row.decision_at))
                _require(all(x.tzinfo is not None for x in (start, end, decision)), 'timestamps require explicit offsets')
                _require(decision < start < end, 'invalid accounting timestamps')
                if prices is not None:
                    days = [x.tz_convert('America/New_York').strftime('%Y-%m-%d') for x in (start, end)]
                    for i, ticker in enumerate(TICKERS):
                        raw = [float(prices.loc[(d, ticker), 'adj_close']) for d in days]
                        _close(r[i], raw[1]/raw[0]-1, 'raw adjusted-price return')
                days = (end.tz_convert('America/New_York').date() - start.tz_convert('America/New_York').date()).days
                turnover = np.abs(target[:-1]-previous[:-1]).sum()
                transaction = turnover * settings['transaction_cost_bps']/10000
                borrow = -np.minimum(target[:-1], 0).sum()*settings['borrow_rate']*days/365
                financing = max(-target[-1], 0)*settings['financing_rate']*days/365
                cash = max(target[-1], 0)*settings['cash_rate']*days/365
                gross = target[:-1] @ r
                net = gross + cash - transaction - borrow - financing
                wealth *= 1+net
                peak = max(peak, wealth)
                _require(wealth > 0, 'bankruptcy')
                for key, value in dict(turnover=turnover, transaction_cost=transaction, borrow_cost=borrow, financing_cost=financing, cash_return=cash, gross_return=gross, net_return=net, wealth=wealth, drawdown=wealth/peak-1, gross_exposure=np.abs(target[:-1]).sum(), net_exposure=target[:-1].sum()).items():
                    _close(getattr(row, key), value, f'{prefix}{strategy}/{row.holding_month}/{key}')
                previous = np.r_[target[:-1]*(1+r), target[-1]+cash-transaction-borrow-financing]/(1+net)
            metric = metrics[metrics.strategy_id == strategy].iloc[0]
            _close(metric.n_months, len(months), 'metric n_months')
            _require(metric.start_month == months[0] and metric.end_month == months[-1], 'metric period mismatch')
            fields = list(metrics.columns[5:])
            if smoke:
                _require(metric[fields].isna().all(), 'smoke aggregate metrics must be undefined')
                continue
            r = series.net_return.to_numpy()
            nav = np.cumprod(1+r)
            dd = nav/np.maximum.accumulate(np.r_[1., nav])[1:]-1
            vol = float(np.std(r, ddof=1)) if len(r) > 1 else None
            downside = float(np.sqrt(np.mean(np.minimum(r, 0)**2)))
            expected_metrics = dict(cagr=math.expm1(math.log(nav[-1])*12/len(r)), ann_vol=vol*math.sqrt(12) if vol is not None else None,
                sharpe=float(r.mean())/vol*math.sqrt(12) if vol else None, sortino=float(r.mean())/downside*math.sqrt(12) if len(r)>1 and downside else None,
                maxdd=dd.min(), avgdd=dd[dd<0].mean() if (dd<0).any() else 0., avgdd_all_months=dd.mean(), positive_ratio=(r>0).mean(),
                mean_turnover=series.turnover.mean(), total_cost=series[['transaction_cost','borrow_cost','financing_cost']].to_numpy().sum(), cash_ratio=(series.gross_exposure==0).mean())
            for key, value in expected_metrics.items():
                _close(metric[key], value, f'{prefix}{strategy}/metric/{key}')


def verify_run(run, report=None):
    """Return strict-JSON checks; never rewrite evidence or rerun research.

    Accept a single run, suite directory, or research_evidence_bundle JSON.
    A failed/missing child is a verification failure even if a bundle labels it
    diagnostic. Unrun entries remain scope disclosures, never executed evidence.
    """
    result = {'status': 'succeeded', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'verifier_source': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(__file__)},
        'checks': [], 'errors': [], 'scope': []}
    seen = set()

    def check(name, path, action):
        try:
            action()
            result['checks'].append({'name': name, 'path': str(path), 'status': 'pass'})
        except (ResearchError, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, ArithmeticError) as exc:
            code = exc.code.value if isinstance(exc, ResearchError) else ErrorCode.VERIFICATION_FAILED.value
            result['checks'].append({'name': name, 'path': str(path), 'status': 'fail'})
            result['errors'].append({'error_code': code, 'reason': str(exc), 'path': str(path)})
            result['status'] = 'failed'

    def visit(path, data_root=None):
        path = Path(path).resolve()
        _require(path not in seen, f'duplicate or cyclic run: {path}')
        seen.add(path)
        manifest_path = path/'run_manifest.json' if path.is_dir() else path
        manifest = read_json(manifest_path)
        root = manifest_path.parent
        if manifest.get('kind') == 'research_evidence_bundle':
            _require(isinstance(manifest.get('runs'), list) and bool(manifest['runs']), 'bundle has no runs')
            _references(root, manifest.get('references', []))
            result['scope'].append({'path': str(path), 'unrun': manifest.get('unrun', [])})
            for item in manifest['runs']:
                child = (root/item['path']).resolve()
                check('bundle_run', child, lambda child=child: visit(child, data_root))
            return
        _require(manifest.get('status') == 'succeeded', 'run did not succeed')
        result['scope'].append({'path': str(root), 'original_code_revision': manifest.get('code_revision', 'not_recorded'),
            'certification': 'supplied artifacts and independent arithmetic; not refit, scale-estimator or full-paper certification'})
        check('artifact_inventory', root, lambda: _inventory(root, manifest, 'run_manifest.json'))
        if manifest.get('kind') == 'reproduction_suite':
            planned = read_json(root/'planned_runs.json')
            execution = read_json(root/'execution_plan.json')['runs']
            completed = read_json(root/'suite_results.json')['completed']
            ids = [x['id'] for x in planned['runs']]
            _require(len(set(ids)) == len(ids) == planned['planned_count'] and bool(ids), 'invalid planned child inventory')
            _require(set(ids) == {x['id'] for x in execution} == {x['id'] for x in completed} and len(execution) == len(completed) == len(ids), 'suite children differ from plan')
            config = read_json(root/'effective_config.json')
            data_root = data_root or config.get('config', {}).get('data', {}).get('root')
            for item in execution:
                child = confined_path(root, item['path'])
                saved = read_json(child/'run_manifest.json')
                _require(saved['actual_period']['evaluated_months'] == item['actual_months'], 'child months differ from execution plan')
                selection = item['strategy_selection']
                _require(len(saved['strategy_ids']) == 50 if selection == 'all50' else saved['strategy_ids'] == selection, 'child strategies differ from execution plan')
                _require(next(x for x in completed if x['id'] == item['id'])['status'] == 'succeeded', 'planned child did not succeed')
                check('suite_child', child, lambda child=child: visit(child, data_root))
        else:
            check('independent_accounting_and_metrics', root, lambda: _financial(root, manifest, data_root, result['scope']))

    check('run_evidence', run, lambda: visit(run, os.environ.get('REGIME_DATA_ROOT')))
    if report is not None:
        def verify_report():
            path = Path(report).resolve()
            path = path/'report_manifest.json' if path.is_dir() else path
            manifest = read_json(path)
            _require(manifest.get('kind') == 'research_report' and manifest.get('status') == 'succeeded', 'report did not succeed')
            _inventory(path.parent, manifest, 'report_manifest.json')
            input_path = Path(run).resolve()
            input_path = input_path/'run_manifest.json' if input_path.is_dir() else input_path
            _require(input_path in {(path.parent/source['path']).resolve() for source in manifest['sources']},
                'report does not reference the supplied run or bundle')
            _references(path.parent, manifest['sources'])
            for filename in ('traceability.json', 'findings.json'):
                document = read_json(path.parent/filename)
                _references(path.parent, document['references'])
                items = document.get('items', [])
                expected = ({f'EQ{i:02d}' for i in range(1,20)} | {'ALG1'} | {f'FIG{i:02d}' for i in range(1,14)} | {f'TABLE{i:02d}' for i in range(1,7)}) if filename == 'traceability.json' else {f'F{i:02d}' for i in range(1,26)}
                _require({row['id'] for row in items} == expected and len(items) == len(expected), 'report trace/finding inventory mismatch')
                for row in items:
                    _references(path.parent, row['references'])
        check('report_evidence', report, verify_report)
    return result
