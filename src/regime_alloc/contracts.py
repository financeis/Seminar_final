"""Shared, side-effect-free identifiers, errors and artifact contracts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = '1.0'
TICKERS = ('SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')
PRICE_KEY = ('session', 'ticker')
MACRO_KEY = ('vintage_month', 'base_month', 'series')
WINDOW_KEY = ('run_id', 'decision_month')
FORECAST_KEY = (*WINDOW_KEY, 'model', 'ticker')


class ErrorCode(StrEnum):
    INVALID_CONFIG = 'invalid_config'
    MISSING_DATA = 'missing_data'
    DUPLICATE_KEY = 'duplicate_key'
    NONFINITE_VALUE = 'nonfinite_value'
    CALENDAR_GAP = 'calendar_gap'
    HASH_MISMATCH = 'hash_mismatch'
    PARTITION_MISMATCH = 'partition_mismatch'
    DEGENERATE_PARTITION = 'degenerate_partition'
    INSUFFICIENT_HISTORY = 'insufficient_history'
    NUMERICAL_FAILURE = 'numerical_failure'
    RUN_CONFLICT = 'run_conflict'
    VERIFICATION_FAILED = 'verification_failed'


class RunStatus(StrEnum):
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'


class CheckStatus(StrEnum):
    PASS = 'pass'
    FAIL = 'fail'
    NOT_APPLICABLE = 'not_applicable'


class ForecastStatus(StrEnum):
    OK = 'ok'
    FALLBACK = 'fallback'
    UNDEFINED = 'undefined'


class ScoreKind(StrEnum):
    EXPECTED_RETURN = 'expected_return'
    CONDITIONAL_SHARPE = 'conditional_sharpe'
    UTILITY_WEIGHT = 'utility_weight'


class FallbackKind(StrEnum):
    NONE = 'none'
    POOLED = 'pooled'
    REGIME_MEAN = 'regime_mean'
    POOLED_MEAN = 'pooled_mean'
    PRIOR_MEAN = 'prior_mean'


class FindingStatus(StrEnum):
    FIXED = 'fixed'
    IMPLEMENTED = 'implemented'
    ASSUMPTION_DOCUMENTED = 'assumption_documented'
    EXTERNAL_LIMITATION = 'external_limitation'


class ResearchError(ValueError):
    def __init__(self, code: ErrorCode | str, reason: str, *, details: dict | None = None):
        self.code = ErrorCode(code)
        self.reason = reason
        self.details = details or {}
        super().__init__(f'{self.code.value}: {reason}')

    @property
    def exit_code(self) -> int:
        return 2 if self.code == ErrorCode.INVALID_CONFIG else 4 if self.code == ErrorCode.VERIFICATION_FAILED else 3

    def to_dict(self) -> dict:
        return {'error_code': self.code.value, 'reason': self.reason, 'details': self.details}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (Path, date, datetime, pd.Period)):
        return value.isoformat() if hasattr(value, 'isoformat') else str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ResearchError(ErrorCode.NONFINITE_VALUE, 'JSON cannot contain NaN or infinity')
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _null_reasons(value: Any, location: str = '$') -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if item is None and not (value.get('reason') or value.get(f'{key}_reason') or value.get('null_reasons', {}).get(key)):
                raise ResearchError(ErrorCode.MISSING_DATA, f'null at {location}.{key} requires a reason')
            _null_reasons(item, f'{location}.{key}')
    elif isinstance(value, list):
        if any(item is None for item in value):
            raise ResearchError(ErrorCode.MISSING_DATA, f'bare null in array at {location}; use value/reason records')
        for i, item in enumerate(value):
            _null_reasons(item, f'{location}[{i}]')


def atomic_write_bytes(path: Path | str, content: bytes) -> None:
    """Commit a complete file without ever replacing a previous output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ResearchError(ErrorCode.RUN_CONFLICT, f'output already exists: {path}')
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails on an existing destination.
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise ResearchError(ErrorCode.RUN_CONFLICT, f'output already exists: {path}') from exc
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path | str, value: Any) -> None:
    ready = _jsonable(value)
    _null_reasons(ready)
    atomic_write_bytes(path, (canonical_json(ready) + '\n').encode('utf-8'))


def require_unique(frame: pd.DataFrame, keys: list[str] | tuple[str, ...]) -> None:
    if any(k not in frame for k in keys):
        raise ResearchError(ErrorCode.MISSING_DATA, f'missing key columns: {keys}')
    if frame[list(keys)].isna().any().any():
        raise ResearchError(ErrorCode.MISSING_DATA, f'null join key: {keys}')
    if frame.duplicated(list(keys)).any():
        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate key: {keys}')


def checked_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str] | tuple[str, ...], *, partition_column: str = 'partition_id') -> pd.DataFrame:
    """One-to-one complete join; mismatched partitions or dropped keys fail."""
    require_unique(left, keys)
    require_unique(right, keys)
    a = set(map(tuple, left[list(keys)].to_numpy()))
    b = set(map(tuple, right[list(keys)].to_numpy()))
    if a != b:
        raise ResearchError(ErrorCode.MISSING_DATA, 'join would omit keys')
    result = left.merge(right, on=list(keys), validate='one_to_one', suffixes=('', '_right'))
    if partition_column in left and partition_column in right and partition_column not in keys:
        if not result[partition_column].eq(result[f'{partition_column}_right']).all():
            raise ResearchError(ErrorCode.PARTITION_MISMATCH, 'joined rows belong to different partitions')
        result = result.drop(columns=f'{partition_column}_right')
    return result


def save_npz(path: Path | str, arrays: dict[str, np.ndarray]) -> dict:
    """Numeric-only model archive; returns a hash/shape/dtype manifest."""
    import io
    meta = {}
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
            raise ResearchError(ErrorCode.NONFINITE_VALUE, f'unsafe/nonfinite array: {name}')
        meta[name] = {'shape': list(a.shape), 'dtype': a.dtype.str, 'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_write_bytes(path, stream.getvalue())
    return {'sha256': sha256_file(path), 'arrays': meta}


def load_npz(path: Path | str, manifest: dict) -> dict[str, np.ndarray]:
    if sha256_file(path) != manifest['sha256']:
        raise ResearchError(ErrorCode.HASH_MISMATCH, f'archive changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest['arrays']):
            raise ResearchError(ErrorCode.VERIFICATION_FAILED, 'NPZ member mismatch')
        result = {}
        for name, spec in manifest['arrays'].items():
            a = archive[name]
            if list(a.shape) != spec['shape'] or a.dtype.str != spec['dtype'] or hashlib.sha256(a.tobytes(order='C')).hexdigest() != spec['sha256']:
                raise ResearchError(ErrorCode.HASH_MISMATCH, f'array changed: {name}')
            result[name] = a
        return result
