"""Load original function ASTs without executing script-level I/O.

The source is never rewritten. Each result records the exact original file hash.
Standalone numeric oracles belong in the individual tests, not in this loader.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm

TICKERS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_ref(repo, filename, symbol=None):
    path = Path(repo) / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    matches = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol]
    node = matches[-1] if matches else tree
    return {'path': filename, 'symbol': symbol, 'line_start': getattr(node, 'lineno', 1),
            'line_end': getattr(node, 'end_lineno', len(path.read_text(encoding='utf-8-sig').splitlines())),
            'sha256': sha256(path)}


def load_functions(repo, filename, names=None, extra=None):
    path = Path(repo) / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    scope = {'np': np, 'pd': pd, 'math': math, 'Path': Path, 'norm': norm, 'TICKERS': TICKERS}
    try:
        from scipy.optimize import linear_sum_assignment
        scope.update(linear_sum_assignment=linear_sum_assignment, SCIPY_OK=True)
    except ImportError:
        scope['SCIPY_OK'] = False
    scope.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def record(check_id, title, passed, expected, observed, *, source_refs=(), inputs=None,
           oracle='', tolerance=None, notes=(), status=None):
    return jsonable({'id': check_id, 'title': title, 'status': status or ('PASS' if passed else 'FAIL'),
                     'expected': expected, 'observed': observed, 'source_refs': list(source_refs),
                     'inputs': inputs, 'oracle': oracle, 'tolerance': tolerance, 'notes': list(notes)})


def write_results(output, domain, results):
    path = Path(output) / domain / 'results.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(results), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return path
