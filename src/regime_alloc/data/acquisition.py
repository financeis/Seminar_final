"""Explicit network boundary for immutable real-data acquisition."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import io
from pathlib import Path
import shutil
import tempfile
import time
import requests

import pandas as pd

from ..config import DataConfig, ResearchConfig
from ..contracts import ErrorCode, ResearchError, atomic_write_bytes, sha256_file, write_json
from .io import read_json, file_record, manifest_record, stage_provider, publish_provider, verify_manifest, resolve_dataset
from .yahoo import accept_yahoo, acquire_yahoo
from .fred import accept_fred, CHANGES_URL, FRED_PAGE

NBER_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC'
FRED_ARCHIVES = {
    'vintages-1999-2014.zip': 'https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip?hash=8A23C5FAF7A0D743A353D77DF4704028&sc_lang=en',
    'vintages-2015-2025.zip': 'https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip',
    'appendix-2024-03.zip': 'https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/fred-md_appendix0324.zip?hash=C8B84CF979313F21D9034FE68134E8EF&sc_lang=en',
}


def download_bytes(url: str, config: DataConfig, *, getter=requests.get, sleeper=time.sleep) -> bytes:
    if len(config.retry_delays) > 3:
        raise ResearchError(ErrorCode.INVALID_CONFIG, 'at most three retries are permitted')
    last = ''
    for attempt in range(len(config.retry_delays)+1):
        try:
            response = getter(url, timeout=config.timeout_seconds)
            response.raise_for_status()
            content = response.content
            if not content:
                raise OSError('empty response')
            return content
        except (OSError, ValueError) as exc:
            last = str(exc)
            if attempt < len(config.retry_delays):
                sleeper(config.retry_delays[attempt])
    raise ResearchError(ErrorCode.MISSING_DATA, f'required download failed: {url}: {last}')


def parse_nber(content: bytes) -> pd.DataFrame:
    try:
        table = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')
        if len(table.columns) != 2 or table.columns[-1] != 'USREC':
            raise ValueError('expected date and USREC columns')
        dates = pd.to_datetime(table.iloc[:,0], errors='raise')
        values = pd.to_numeric(table.USREC, errors='raise')
        if dates.isna().any() or not (dates.dt.day == 1).all() or not values.isin([0,1]).all():
            raise ValueError('non-monthly date or nonbinary USREC indicator')
        result = pd.DataFrame({'month': dates.dt.strftime('%Y-%m'), 'usrec': values.astype(int)})
        from ..contracts import require_unique
        from .calendar import month_range
        require_unique(result, ('month',))
        if result.month.tolist() != month_range(result.month.iloc[0], result.month.iloc[-1]):
            raise ValueError('missing NBER indicator month')
        return result
    except (ValueError, KeyError, IndexError) as exc:
        raise ResearchError(ErrorCode.MISSING_DATA, f'invalid official NBER/USREC response: {exc}') from exc


def acquire_nber(config: DataConfig, *, fetcher=download_bytes) -> dict:
    content = fetcher(NBER_URL, config)
    table = parse_nber(content)
    stage, dest = stage_provider(config.root, 'nber')
    try:
        atomic_write_bytes(stage / 'USREC.csv', content)
        records = [file_record(stage / 'USREC.csv', 'raw/nber/USREC.csv', 'raw_indicator')]
        manifest = manifest_record(provider='fred_usrec_nber', source_urls=[NBER_URL, 'https://fred.stlouisfed.org/series/USREC', 'https://www.nber.org/research/business-cycle-dating'], retrieved_at=datetime.now(timezone.utc).isoformat(), request_parameters={'series': 'USREC', 'purpose': 'ex_post_interpretation_only'}, library_versions={'parser': 'pandas'}, files=records, coverage={'first_month': table.month.iloc[0], 'last_month': table.month.iloc[-1]}, quality={'rows': len(table), 'binary': True}, warnings=['Ex-post business-cycle interpretation only; prohibited as a trading feature.', 'USREC covers the month after the NBER peak through the trough month.'])
        return publish_provider(stage, dest, manifest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_nber(data_root: Path | str, dataset_id: str | None = None) -> pd.DataFrame:
    directory, _ = resolve_dataset(data_root, 'nber', dataset_id)
    result = parse_nber((directory / 'USREC.csv').read_bytes())
    result.attrs['purpose'] = 'ex_post_interpretation_only'
    return result


def acquire_fred(config: DataConfig, *, fetcher=download_bytes) -> dict:
    changes = fetcher(CHANGES_URL, config)
    if (config.fred_source / 'manifest.json').is_file():
        return accept_fred(config.root, config.fred_source, changes_pdf=changes)
    Path(config.root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fred-source-', dir=config.root) as tmp:
        source = Path(tmp)
        records = []
        for name, url in FRED_ARCHIVES.items():
            atomic_write_bytes(source / name, fetcher(url, config))
            records.append({'path': name, 'url': url, 'sha256': sha256_file(source/name), 'bytes': (source/name).stat().st_size})
        write_json(source / 'manifest.json', {'source_page': FRED_PAGE, 'files': records})
        return accept_fred(config.root, source, changes_pdf=changes)


def acquire(config: ResearchConfig | DataConfig) -> dict:
    """Accept prepared snapshots or acquire official data. Mandatory failure raises.

    Existing complete providers are verified and reused, never overwritten.
    A corrupt or incomplete existing provider fails and requires a new data root.
    """
    c = config.data if isinstance(config, ResearchConfig) else config
    manifests = {}
    for name in ('yahoo', 'fred', 'nber'):
        identifier = getattr(c, f'{name}_dataset_id')
        candidates = list((c.root / 'raw' / name).glob('*/dataset_manifest.json'))
        if identifier and (c.root/'raw'/name/identifier/'dataset_manifest.json').is_file() or not identifier and candidates:
            _, manifests[name] = resolve_dataset(c.root, name, identifier or None)
        elif name == 'yahoo':
            manifests[name] = accept_yahoo(c.root, c.yahoo_source) if (c.yahoo_source / 'manifest.json').is_file() else acquire_yahoo(c)
        elif name == 'fred':
            manifests[name] = acquire_fred(c)
        else:
            manifests[name] = acquire_nber(c)
        if identifier and manifests[name]['dataset_id'] != identifier:
            raise ResearchError(ErrorCode.HASH_MISMATCH, f'acquired {name} differs from pinned dataset ID')
    from .validation import validate
    pinned = replace(c, **{f'{name}_dataset_id': manifest['dataset_id'] for name, manifest in manifests.items()})
    summary = validate(replace(config, data=pinned) if isinstance(config, ResearchConfig) else pinned)
    return {'status': 'succeeded', 'datasets': {name: m['dataset_id'] for name, m in manifests.items()}, 'validation': summary}
