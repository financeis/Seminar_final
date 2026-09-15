"""Strict TOML configuration shared by offline research and acquisition."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
import math
import re
import tomllib
from typing import get_type_hints, get_origin, get_args

from .contracts import TICKERS, ErrorCode, ResearchError


def month_string(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value) or int(value[:4]) < 1:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'invalid YYYY-MM month: {value!r}')
    return value


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path('../data')
    yahoo_source: Path = Path('../../work/etf-acquisition/20260915T173858Z')
    fred_source: Path = Path('../../work/fred-acquisition/20260915T174738Z')
    tickers: tuple[str, ...] = TICKERS
    start_date: str = '1993-01-01'
    end_date: str = '2026-09-16'
    fixed_vintage: str = '2023-02'
    yahoo_dataset_id: str = ''
    fred_dataset_id: str = ''
    nber_dataset_id: str = ''
    retry_delays: tuple[float, ...] = (2., 5., 10.)
    timeout_seconds: int = 30


@dataclass(frozen=True)
class ResearchSettings:
    profile: str = 'vintage_lagged'
    start_month: str = '2003-02'
    end_month: str = '2022-12'
    train_months: int = 48
    lag_months: int = 2
    missing_rate: float = .20
    ffill_limit: int = 2
    pca_variance: float = .95
    k_normal: int = 5
    n_init: int = 20
    max_iter: int = 300
    tol: float = 1e-6
    seed: int = 0
    lambda_min: float = 1e-4
    lambda_max: float = 1e4
    lambda_count: int = 41
    min_regime_samples: int = 6
    fallback: str = 'pooled'
    validation: str = 'loo'
    cov_shrink: float = .1
    cov_eps: float = 1e-8
    tau: float = .05
    omega_scale: float = 1.
    delta: float = 3.
    models: tuple[str, ...] = ('naive', 'ridge', 'bl', 'mvo')
    strategies: tuple[str, ...] = ('lo', 'lns', 'los', 'mx')
    selection_sizes: tuple[int, ...] = (2, 3, 4)

    @property
    def lambdas(self) -> tuple[float, ...]:
        import numpy as np
        return tuple(np.logspace(math.log10(self.lambda_min), math.log10(self.lambda_max), self.lambda_count))


@dataclass(frozen=True)
class PortfolioSettings:
    transaction_cost_bps: float = 0.
    borrow_rate: float = 0.
    financing_rate: float = 0.
    cash_rate: float = 0.
    vol_target: float = .10
    vol_lookback: int = 36
    gross_cap: float = 2.
    main_table_scaled: bool = False


@dataclass(frozen=True)
class SuiteSettings:
    control_repetitions: int = 100
    bootstrap_repetitions: int = 1000
    bootstrap_block_months: int = 12
    representative_strategy: str = 'lo'
    representative_size: int = 2
    sensitivity_lags: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    research: ResearchSettings = field(default_factory=ResearchSettings)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    suite: SuiteSettings = field(default_factory=SuiteSettings)
    output_root: Path = Path('../runs')


def _typed(value, annotation, label, base):
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must be a path')
        path = Path(value)
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    if get_origin(annotation) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} must be an array')
        return tuple(_typed(x, get_args(annotation)[0], label, base) for x in value)
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) if annotation is float else type(value) is annotation
    if not valid or isinstance(value, float) and not math.isfinite(value):
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{label} has incorrect type or nonfinite value')
    return float(value) if annotation is float else value


def _section(cls, raw, base):
    if not isinstance(raw, dict):
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'{cls.__name__} must be a table')
    allowed = {x.name for x in fields(cls)}
    if set(raw) - allowed:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'unknown {cls.__name__} keys: {sorted(set(raw)-allowed)}')
    defaults = cls()
    hints = get_type_hints(cls)
    return cls(**{k: _typed(raw.get(k, getattr(defaults, k)), hints[k], k, base) for k in allowed})


def validate_config(c: ResearchConfig) -> ResearchConfig:
    d, r, p, s = c.data, c.research, c.portfolio, c.suite
    def check(ok, reason):
        if not ok:
            raise ResearchError(ErrorCode.INVALID_CONFIG, reason)
    check(d.tickers == TICKERS, f'tickers must have fixed order {TICKERS}')
    for identifier in (d.yahoo_dataset_id, d.fred_dataset_id, d.nber_dataset_id):
        check(identifier == '' or bool(re.fullmatch('[0-9a-f]{64}', identifier)), 'dataset IDs must be lowercase SHA256 content identifiers')
    try:
        dates = [date.fromisoformat(x) for x in (d.start_date, d.end_date)]
    except (ValueError, TypeError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid acquisition date') from exc
    check(dates[0] < dates[1], 'end_date must follow start_date')
    for value in (d.fixed_vintage, r.start_month, r.end_month):
        month_string(value)
    check(d.fixed_vintage == '2023-02', 'fixed_snapshot requires the 2023-02 public release')
    check(r.start_month <= r.end_month, 'end_month precedes start_month')
    check(0 <= len(d.retry_delays) <= 3 and all(x >= 0 for x in d.retry_delays), 'at most three nonnegative retry delays')
    check(d.timeout_seconds > 0, 'timeout must be positive')
    check(r.profile in ('fixed_snapshot', 'vintage_lagged'), 'unknown profile')
    check(r.train_months == 48 and r.lag_months in (1, 2, 3), 'research requires a 48-month training window and lag 1, 2, or 3')
    check(0 <= r.missing_rate < 1 and 0 <= r.ffill_limit <= r.train_months, 'invalid missing-data policy')
    check(0 < r.pca_variance <= 1 and r.k_normal >= 2 and r.k_normal < r.train_months, 'invalid PCA/clustering parameters')
    check(r.n_init > 0 and r.max_iter > 0 and r.tol > 0 and r.seed >= 0, 'invalid numerical configuration')
    check(0 < r.lambda_min <= r.lambda_max and r.lambda_count >= 2, 'invalid lambda grid')
    check(r.min_regime_samples >= 2 and r.fallback in ('pooled', 'regime_mean'), 'invalid fallback')
    check(r.validation in ('loo', 'blocked'), 'invalid validation')
    check(0 <= r.cov_shrink <= 1 and all(x > 0 for x in (r.cov_eps, r.tau, r.omega_scale, r.delta)), 'invalid covariance/view parameters')
    for values, allowed, name in [(r.models, ('naive', 'ridge', 'bl', 'mvo'), 'models'), (r.strategies, ('lo', 'lns', 'los', 'mx'), 'strategies'), (r.selection_sizes, (2, 3, 4), 'selection_sizes')]:
        check(bool(values) and len(set(values)) == len(values) and set(values) <= set(allowed), f'invalid {name}')
    check(all(x >= 0 for x in (p.transaction_cost_bps, p.borrow_rate, p.financing_rate)), 'negative cost/rate')
    check(p.vol_target > 0 and p.vol_lookback >= 2 and p.gross_cap > 0, 'invalid volatility scaling')
    check(min(s.control_repetitions, s.bootstrap_repetitions, s.bootstrap_block_months) > 0, 'suite counts must be positive')
    check(s.representative_strategy in r.strategies and s.representative_size in r.selection_sizes, 'representative strategy not included')
    check(bool(s.sensitivity_lags) and len(set(s.sensitivity_lags)) == len(s.sensitivity_lags) and set(s.sensitivity_lags) <= {1, 2, 3}, 'invalid sensitivity lags')
    return c


def load_config(path: str | Path) -> ResearchConfig:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ResearchError(ErrorCode.INVALID_CONFIG, str(exc)) from exc
    allowed = {'data', 'research', 'portfolio', 'suite', 'output_root'}
    if set(raw) - allowed:
        raise ResearchError(ErrorCode.INVALID_CONFIG, f'unknown configuration keys: {sorted(set(raw)-allowed)}')
    c = ResearchConfig(
        data=_section(DataConfig, raw.get('data', {}), path.parent),
        research=_section(ResearchSettings, raw.get('research', {}), path.parent),
        portfolio=_section(PortfolioSettings, raw.get('portfolio', {}), path.parent),
        suite=_section(SuiteSettings, raw.get('suite', {}), path.parent),
        output_root=_typed(raw.get('output_root', '../runs'), Path, 'output_root', path.parent),
    )
    return validate_config(c)
