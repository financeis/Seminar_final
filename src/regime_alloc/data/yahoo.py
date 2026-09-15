"""Preserve original Yahoo observations; normalize only in memory."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..contracts import TICKERS, ErrorCode, ResearchError, require_unique, atomic_write_bytes
from .calendar import session_close, nyse_calendar
from .io import read_json, source_files, file_record, manifest_record, stage_provider, publish_provider, verify_manifest, confined_path, resolve_dataset

RAW_COLUMNS = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume', 'Dividends': 'dividends', 'Stock Splits': 'splits', 'Capital Gains': 'capital_gains'}
REQUIRED_PRICES = ('open', 'high', 'low', 'close', 'adj_close', 'volume')
OPTIONAL_EVENTS = ('dividends', 'splits', 'capital_gains')
YAHOO_PARAMETERS = dict(auto_adjust=False, back_adjust=False, actions=True, repair=False, keepna=True, interval='1d')


def validate_prices(prices: pd.DataFrame, *, tickers: tuple[str, ...] = TICKERS) -> dict:
    require_unique(prices, ('session', 'ticker'))
    required = {'session', 'ticker', 'finalized', *REQUIRED_PRICES}
    if not required <= set(prices.columns) or set(prices.ticker) != set(tickers):
        raise ResearchError(ErrorCode.MISSING_DATA, 'missing/extra required ticker or price column')
    if prices.empty or prices[list(REQUIRED_PRICES)].isna().any().any():
        raise ResearchError(ErrorCode.MISSING_DATA, 'required price observations contain nulls')
    if not np.isfinite(prices[list(REQUIRED_PRICES)].to_numpy(dtype=float)).all():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, 'price observations contain NaN/Inf')
    if (prices[['open','high','low','close','adj_close']] <= 0).any().any() or (prices.volume < 0).any():
        raise ResearchError(ErrorCode.MISSING_DATA, 'nonpositive price or negative volume')
    if (prices.high < prices[['open','close','low']].max(axis=1) - 1e-8).any() or (prices.low > prices[['open','close','high']].min(axis=1) + 1e-8).any():
        raise ResearchError(ErrorCode.MISSING_DATA, 'invalid OHLC ordering')
    if not prices.finalized.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ResearchError(ErrorCode.MISSING_DATA, 'finalized must be boolean')
    sessions = set(nyse_calendar().sessions.strftime('%Y-%m-%d'))
    if not set(prices.session) <= sessions:
        raise ResearchError(ErrorCode.CALENDAR_GAP, 'price includes a non-XNYS session')
    for col in OPTIONAL_EVENTS:
        if col in prices and np.isinf(prices[col].to_numpy(dtype=float)).any():
            raise ResearchError(ErrorCode.NONFINITE_VALUE, f'infinite optional event: {col}')
    return {'rows': len(prices), 'assets': len(tickers), 'unfinalized_rows': int((~prices.finalized).sum()), 'optional_missing': {col: int(prices[col].isna().sum()) if col in prices else len(prices) for col in OPTIONAL_EVENTS}}


def normalize_yahoo(files: list[tuple[str, Path]], retrieved_at: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(retrieved_at)
    if cutoff.tzinfo is None:
        raise ResearchError(ErrorCode.MISSING_DATA, 'retrieval timestamp must include timezone')
    frames = []
    calendar = nyse_calendar()
    close_map = {s.strftime('%Y-%m-%d'): c for s, c in zip(calendar.sessions, calendar.schedule['close'])}
    for ticker, path in files:
        frame = pd.read_csv(path, encoding='utf-8-sig').rename(columns=RAW_COLUMNS)
        if 'date' not in frame:
            raise ResearchError(ErrorCode.MISSING_DATA, f'raw price date column absent: {path}')
        frame['session'] = pd.to_datetime(frame.pop('date'), errors='raise').dt.strftime('%Y-%m-%d')
        frame['ticker'] = ticker
        for col in OPTIONAL_EVENTS:
            if col not in frame:
                frame[col] = np.nan
        frame['finalized'] = frame.session.map(lambda day: day in close_map and cutoff >= close_map[day])
        frames.append(frame[['session','ticker',*RAW_COLUMNS.values(),'finalized']])
    result = pd.concat(frames, ignore_index=True)
    validate_prices(result)
    order = {ticker: i for i, ticker in enumerate(TICKERS)}
    return result.assign(_order=result.ticker.map(order)).sort_values(['session','_order']).drop(columns='_order').reset_index(drop=True)


def accept_yahoo(data_root: Path | str, source: Path | str) -> dict:
    source = Path(source)
    original = read_json(source / 'manifest.json')
    prepared = source_files(source, original)
    if tuple(original.get('tickers', [])) != TICKERS or {x['ticker'] for x, _ in prepared} != set(TICKERS):
        raise ResearchError(ErrorCode.MISSING_DATA, 'prepared Yahoo snapshot must include the fixed ten assets')
    for key, value in YAHOO_PARAMETERS.items():
        if original.get('parameters', {}).get(key) != value:
            raise ResearchError(ErrorCode.MISSING_DATA, f'Yahoo request contract differs: {key}')
    retrieved_at = datetime.strptime(original['fetched_at_utc'], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
    prices = normalize_yahoo([(x['ticker'], p) for x, p in prepared], retrieved_at)
    quality = validate_prices(prices)
    stage, dest = stage_provider(data_root, 'yahoo')
    try:
        records = []
        for item, path in prepared:
            shutil.copyfile(path, stage / path.name)
            records.append(file_record(stage / path.name, f'raw/yahoo/{path.name}', 'raw_price'))
        for name in ('manifest.json', 'quality.json'):
            if (source / name).is_file():
                target = f'source_{name}'
                shutil.copyfile(source / name, stage / target)
                records.append(file_record(stage / target, f'raw/yahoo/{target}', 'source_metadata'))
        result = manifest_record(provider='yahoo', source_urls=[f'https://finance.yahoo.com/quote/{t}/history/' for t in TICKERS], retrieved_at=retrieved_at, request_parameters=original['parameters'], library_versions={'yfinance': original['yfinance_version'], 'python': original['python_version']}, files=records, tickers=TICKERS, coverage={'first_session': prices.session.min(), 'last_session': prices.session.max(), 'by_ticker': {t: {'first': g.session.min(), 'last': g.session.max(), 'rows': len(g)} for t, g in prices.groupby('ticker')}}, quality=quality, warnings=['Intraday observations are retained with finalized=false.', 'Missing optional corporate-action fields remain null; they are not observations of zero.'])
        return publish_provider(stage, dest, result)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_prices(data_root: Path | str, dataset_id: str | None = None) -> pd.DataFrame:
    _, manifest = resolve_dataset(data_root, 'yahoo', dataset_id)
    files = [(Path(x['path']).stem, confined_path(data_root, x['path'])) for x in manifest['files'] if x['kind'] == 'raw_price']
    return normalize_yahoo(files, manifest['retrieved_at'])


def acquire_yahoo(config: DataConfig, *, downloader=None, sleeper=time.sleep) -> dict:
    """Explicit network entrypoint; initial request plus up to three retries."""
    if config.tickers != TICKERS or len(config.retry_delays) > 3:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'invalid tickers/retry policy')
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    params = dict(YAHOO_PARAMETERS, start=config.start_date, end=config.end_date, threads=False, progress=False, group_by='ticker', timeout=config.timeout_seconds, multi_level_index=False)
    downloaded = {}
    for ticker in TICKERS:
        failure = ''
        for attempt in range(len(config.retry_delays)+1):
            try:
                # Per-asset downloads preserve each actual listing interval;
                # a union-index request inserts artificial pre-listing NaN rows.
                frame = downloader(ticker, **params)
                if frame is None or frame.empty:
                    raise ValueError('Yahoo returned no observations')
                downloaded[ticker] = frame
                break
            except Exception as exc:
                failure = str(exc)
                if attempt < len(config.retry_delays):
                    sleeper(config.retry_delays[attempt])
        if ticker not in downloaded:
            raise ResearchError(ErrorCode.MISSING_DATA, f'Yahoo acquisition failed for {ticker}: {failure}')
    # Prepare only real provider frames. No files are published until all ten validate.
    import tempfile
    import platform
    import importlib.metadata
    from ..contracts import write_json, sha256_file
    Path(config.root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.yahoo-source-', dir=config.root) as tmp:
        tmp = Path(tmp)
        entries = []
        now = datetime.now(timezone.utc)
        for ticker in TICKERS:
            try:
                frame = downloaded[ticker].copy()
            except KeyError as exc:
                raise ResearchError(ErrorCode.MISSING_DATA, f'Yahoo omitted asset: {ticker}') from exc
            # Provider all-null rows remain present (keepna=True); validation fails them.
            frame.index.name = 'date'
            path = tmp / f'{ticker}.csv'
            frame.to_csv(path, encoding='utf-8', index=True)
            entries.append({'ticker': ticker, 'path': path.name, 'sha256': sha256_file(path)})
        write_json(tmp / 'manifest.json', {'tickers': list(TICKERS), 'parameters': params, 'files': entries, 'fetched_at_utc': now.strftime('%Y%m%dT%H%M%SZ'), 'yfinance_version': importlib.metadata.version('yfinance'), 'python_version': platform.python_version()})
        return accept_yahoo(config.root, tmp)
