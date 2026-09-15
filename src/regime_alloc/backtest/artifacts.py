"""Versioned table fields and exclusive, atomic run publication.

The directory itself is the exclusive run reservation. Only its owning writer
can replace the lifecycle manifest; completed data files are never replaced.
Failures preserve every published file and explicitly mark the run failed.
"""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from ..contracts import (ErrorCode, ResearchError, atomic_write_bytes, canonical_json,
                         require_unique, sha256_file, write_json)

COLUMNS = {
    'probabilities.csv': 'run_id decision_month partition_id regime_id current_probability next_probability'.split(),
    'conditional_forecasts.csv': 'run_id decision_month partition_id model ticker regime_id value score_kind raw_n effective_n selected_lambda fallback status reason'.split(),
    'forecasts.csv': 'run_id decision_month partition_id model ticker value score_kind status reason'.split(),
    'weights.csv': 'run_id decision_month strategy_id ticker base_weight target_weight pretrade_weight'.split(),
    'returns.csv': 'run_id holding_month strategy_id decision_at execution_at target_end gross_return transaction_cost borrow_cost financing_cost cash_return net_return turnover gross_exposure net_exposure wealth drawdown'.split(),
    'metrics.csv': 'run_id strategy_id n_months start_month end_month cagr ann_vol sharpe sortino maxdd avgdd avgdd_all_months positive_ratio mean_turnover total_cost cash_ratio'.split(),
}
KEYS = {
    'probabilities.csv': ['run_id', 'decision_month', 'regime_id'],
    'conditional_forecasts.csv': ['run_id', 'decision_month', 'model', 'ticker', 'regime_id'],
    'forecasts.csv': ['run_id', 'decision_month', 'model', 'ticker'],
    'weights.csv': ['run_id', 'decision_month', 'strategy_id', 'ticker'],
    'returns.csv': ['run_id', 'holding_month', 'strategy_id'],
    'metrics.csv': ['run_id', 'strategy_id'],
}
WINDOW_COLUMNS = 'run_id window_id decision_month decision_at execution_at macro_vintage macro_cutoff train_months train_target_end_max feature_names feature_hash transform_hash partition_id regime_order raw_n effective_n pca_components explained_variance imputation_counts exclusions warnings'.split()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunWriter:
    def __init__(self, path, manifest):
        self.path = Path(path).resolve()
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ResearchError(ErrorCode.RUN_CONFLICT, f'run output already exists: {self.path}') from exc
        self.manifest = {**manifest, 'status': 'running', 'started_at': utc_now(),
                         'finished_at': '', 'artifacts': [], 'warnings': list(manifest.get('warnings', [])), 'errors': []}
        self._closed = False
        self._publish_manifest()

    def _publish_manifest(self):
        # Exclusive directory ownership permits lifecycle replacement only.
        content = (canonical_json(self.manifest) + '\n').encode('utf-8')
        fd, name = tempfile.mkstemp(prefix='.manifest-', dir=self.path)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path / 'run_manifest.json')
        finally:
            temporary.unlink(missing_ok=True)

    def _destination(self, name):
        destination = (self.path / name).resolve()
        if self._closed or not destination.is_relative_to(self.path) or destination == self.path or destination.name == 'run_manifest.json':
            raise ResearchError(ErrorCode.RUN_CONFLICT, 'closed run or invalid artifact path')
        return destination

    def json(self, name, value):
        write_json(self._destination(name), value)

    def jsonl(self, name, records, columns=None):
        if columns is not None and any(set(row) != set(columns) for row in records):
            raise ResearchError(ErrorCode.MISSING_DATA, 'JSONL field contract mismatch')
        ready = [{k: row[k] for k in columns} if columns else row for row in records]
        # Preserve declared field order, unlike the canonical hash representation.
        import json
        content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n' for row in ready)
        atomic_write_bytes(self._destination(name), content.encode('utf-8'))

    def csv(self, name, records):
        basename = Path(name).name
        columns = COLUMNS[basename]
        if any(set(row) != set(columns) for row in records):
            raise ResearchError(ErrorCode.MISSING_DATA, f'field contract mismatch: {name}')
        frame = pd.DataFrame(records, columns=columns)
        require_unique(frame, KEYS[basename])
        allowed_null = set(COLUMNS['metrics.csv'][3:]) if basename == 'metrics.csv' else {'selected_lambda'} if basename == 'conditional_forecasts.csv' else set()
        for column in columns:
            if column not in allowed_null and frame[column].isna().any():
                raise ResearchError(ErrorCode.MISSING_DATA, f'null values in {name}:{column}')
            for value in frame[column].dropna():
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ResearchError(ErrorCode.NONFINITE_VALUE, f'nonfinite values in {name}:{column}')
        if basename == 'conditional_forecasts.csv' and any(row['selected_lambda'] is None and not row['reason'] for row in records):
            raise ResearchError(ErrorCode.MISSING_DATA, 'null selected lambda requires reason')
        atomic_write_bytes(self._destination(name), frame.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode('utf-8'))

    def _inventory(self):
        self.manifest['artifacts'] = [{'path': p.relative_to(self.path).as_posix(), 'sha256': sha256_file(p)}
            for p in sorted(self.path.rglob('*')) if p.is_file() and p.name != 'run_manifest.json' and not p.name.startswith('.')]

    def succeed(self):
        if self._closed:
            raise ResearchError(ErrorCode.RUN_CONFLICT, 'run is already closed')
        self._inventory()
        self.manifest.update(status='succeeded', finished_at=utc_now())
        self._publish_manifest()
        self._closed = True
        return self.manifest.copy()

    def fail(self, error):
        if self._closed:
            return
        code = ErrorCode.MISSING_DATA if isinstance(error, OSError) else ErrorCode.NUMERICAL_FAILURE
        record = error.to_dict() if isinstance(error, ResearchError) else {'error_code': code.value, 'reason': f'{type(error).__name__}: {error}'}
        self._inventory()
        self.manifest.update(status='failed', finished_at=utc_now(), errors=[record])
        self._publish_manifest()
        self._closed = True
