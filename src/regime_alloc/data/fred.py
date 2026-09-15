"""FRED-MD snapshots, vintage-local t-codes and provenance-backed groups."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import pandas as pd

from ..contracts import ErrorCode, ResearchError, require_unique, write_json
from .calendar import assumed_available_at, month_range
from .io import safe_zip_members, read_json, source_files, file_record, manifest_record, stage_provider, publish_provider, verify_manifest, confined_path, resolve_dataset, write_parser_provenance

FRED_PAGE = 'https://www.stlouisfed.org/research/economists/mccracken/fred-databases'
CHANGES_URL = 'https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/fredmdchanges0324.pdf'
# These connect metadata only. A historic name is NEVER renamed or populated
# with the successor's observations. Dates are the official change vintages.
ALIASES = {
    'PPIFGS': ('WPSFD49207', '2016-03', 4),
    'PPIFCG': ('WPSFD49502', '2016-03', 4),
    'PPIITM': ('WPSID61', '2016-03', 4),
    'PPICRM': ('WPSID62', '2016-03', 4),
    'CUUR0000SAD': ('CUSR0000SAD', '2017-04', 8),
    'CUUR0000SA0L2': ('CUSR0000SA0L2', '2017-04', 8),
    'AMBSL': ('BOGMBASE', '2020-01', 10),
    'TWEXMMTH': ('TWEXAFEGSMTHx', '2020-04', 11),
    'VXOCLSx': ('VIXCLSx', '2021-12', 13),
}


@dataclass(frozen=True)
class MacroVintage:
    vintage_month: str
    values: pd.DataFrame  # index YYYY-MM, columns original series names; raw levels
    tcodes: pd.Series    # int, index exactly values.columns
    groups: pd.Series   # int, index exactly values.columns; includes group 6
    metadata: dict
    assumed_available_at: str
    dataset_id: str

    def long_frame(self) -> pd.DataFrame:
        frame = self.values.rename_axis('base_month').reset_index().melt(id_vars='base_month', var_name='series', value_name='value')
        frame['vintage_month'] = self.vintage_month
        frame['tcode'] = frame.series.map(self.tcodes)
        frame['group'] = frame.series.map(self.groups)
        frame['assumed_available_at'] = self.assumed_available_at
        frame['value_kind'] = 'raw'
        frame['value_reason'] = np.where(frame.value.isna(), 'provider_missing', '')
        return frame[['vintage_month','base_month','series','value','tcode','group','assumed_available_at','value_kind','value_reason']]


def vintage_from_filename(name: str) -> str | None:
    stem = Path(name).stem
    match = re.fullmatch(r'(\d{4})-(\d{2})', stem) or re.fullmatch(r'FRED-MD[-_](\d{4})m(\d{2})', stem)
    return f'{match[1]}-{match[2]}' if match else None


def build_catalog(archives: list[Path]) -> dict:
    groups = {}
    appendix_sources = {}
    vintages = {}
    observed = {}
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = safe_zip_members(archive)
            for name in members:
                if name.endswith(('_historic_appendix.csv', '_updated_appendix.csv')):
                    # Official appendix source bytes use Windows-1252. Derived
                    # metadata is written as UTF-8 while original ZIP is unchanged.
                    table = pd.read_csv(io.BytesIO(archive.read(name)), encoding='cp1252')
                    for row in table.itertuples():
                        key, group = str(row.fred).upper(), int(row.group)
                        if key in groups and groups[key] != group:
                            raise ResearchError(ErrorCode.MISSING_DATA, f'conflicting official group for {row.fred}')
                        groups[key] = group
                        appendix_sources.setdefault(key, []).append({'archive': path.name, 'member': name, 'id': int(row.id)})
                month = vintage_from_filename(name) if name.lower().endswith('.csv') else None
                if month:
                    if month in vintages:
                        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate macro vintage {month}')
                    columns = list(pd.read_csv(io.BytesIO(archive.read(name)), nrows=0, encoding='cp1252').columns)[1:]
                    if len(set(columns)) != len(columns):
                        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate macro series in {month}')
                    vintages[month] = {'archive': path.name, 'member': name, 'series_count': len(columns)}
                    for series in columns:
                        observed.setdefault(series, []).append(month)
    metadata = {}
    for series, months in observed.items():
        reference = series
        alias = ALIASES.get(series)
        if alias:
            reference = alias[0]
        key = reference.upper()
        if key not in groups:
            raise ResearchError(ErrorCode.MISSING_DATA, f'unknown FRED-MD group: {series}')
        metadata[series] = {
            'series': series, 'group': groups[key], 'group_reference_series': reference,
            'group_sources': appendix_sources[key],
            'observed_first_vintage': min(months), 'observed_last_vintage': max(months),
            'observed_vintages': sorted(months),
            'metadata_only_alias': bool(alias), 'values_replaced': False,
        }
        if alias:
            metadata[series].update({'successor_series': alias[0], 'official_change_vintage': alias[1], 'change_document_url': CHANGES_URL, 'change_item': alias[2], 'applicability': 'group classification of original historic series; preserves vintage-specific raw values and t-code'})
    if not vintages or not groups:
        raise ResearchError(ErrorCode.MISSING_DATA, 'FRED archive missing vintages or appendix')
    expected = set(month_range('1999-08', '2025-12'))
    if set(vintages) != expected:
        raise ResearchError(ErrorCode.CALENDAR_GAP, f'FRED vintage coverage mismatch: missing={sorted(expected-set(vintages))}, extra={sorted(set(vintages)-expected)}')
    return {'vintages': dict(sorted(vintages.items())), 'series': metadata, 'source_page': FRED_PAGE, 'changes_url': CHANGES_URL, 'value_policy': 'raw original series; no successor-value substitution', 'original_encoding': 'cp1252', 'derived_encoding': 'utf-8'}


def parse_macro_csv(content: bytes, vintage_month: str, catalog: dict, *, lag_months: int = 2, dataset_id: str = '') -> MacroVintage:
    # Inspect unmodified header before pandas can mangle duplicate names.
    import csv
    lines = content.decode('cp1252').splitlines()
    header = next(csv.reader([line for line in lines if line.strip()][:1]))
    if len(set(header)) != len(header):
        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate macro header in {vintage_month}')
    table = pd.read_csv(io.BytesIO(content), encoding='cp1252')
    if len(table) < 2 or not str(table.iloc[0,0]).lower().startswith('transform'):
        raise ResearchError(ErrorCode.MISSING_DATA, f'no transformation row in vintage {vintage_month}')
    raw_codes = pd.to_numeric(table.iloc[0,1:], errors='raise')
    if not raw_codes.isin(range(1,8)).all():
        raise ResearchError(ErrorCode.MISSING_DATA, f'invalid vintage-local t-code in {vintage_month}')
    tcodes = raw_codes.astype(int)
    values = table.iloc[1:,1:].apply(pd.to_numeric, errors='raise')
    dates = pd.to_datetime(table.iloc[1:,0], format='mixed', errors='raise')
    if dates.isna().any() or not (dates.dt.day == 1).all():
        raise ResearchError(ErrorCode.CALENDAR_GAP, f'invalid macro base date in {vintage_month}')
    values.index = pd.Index(dates.dt.strftime('%Y-%m').to_numpy(), name='base_month')
    if values.index.has_duplicates:
        raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate macro base month in {vintage_month}')
    if values.index.tolist() != month_range(values.index[0], values.index[-1]):
        raise ResearchError(ErrorCode.CALENDAR_GAP, f'noncontiguous macro base months in {vintage_month}')
    if np.isinf(values.to_numpy()).any():
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'infinite macro value in {vintage_month}')
    group_values = []
    metadata = {}
    for series in values:
        if series not in catalog['series']:
            raise ResearchError(ErrorCode.MISSING_DATA, f'unknown macro series/group: {series}')
        spec = catalog['series'][series]
        if not 1 <= spec['group'] <= 8 or vintage_month not in spec['observed_vintages']:
            raise ResearchError(ErrorCode.MISSING_DATA, f'group metadata not applicable: {series} {vintage_month}')
        group_values.append(spec['group'])
        metadata[series] = spec
    return MacroVintage(vintage_month, values, tcodes, pd.Series(group_values, index=values.columns, name='group', dtype=int), metadata, assumed_available_at(vintage_month, lag_months).isoformat(), dataset_id)


def accept_fred(data_root: Path | str, source: Path | str, *, changes_pdf: bytes) -> dict:
    source = Path(source)
    original = read_json(source / 'manifest.json')
    prepared = source_files(source, original)
    archives = [path for _, path in prepared if path.suffix.lower() == '.zip']
    if len(archives) != 3 or not changes_pdf.startswith(b'%PDF-'):
        raise ResearchError(ErrorCode.MISSING_DATA, 'three official ZIPs and Changes PDF are required')
    catalog = build_catalog(archives)
    fixed_spec = catalog['vintages']['2023-02']
    with zipfile.ZipFile(source / fixed_spec['archive']) as archive:
        fixed = parse_macro_csv(archive.read(fixed_spec['member']), '2023-02', catalog)
    if fixed.values.shape[1] != 126 or fixed.values.index[-1] != '2023-01':
        raise ResearchError(ErrorCode.MISSING_DATA, 'fixed 2023-02 snapshot differs from acquired specification')
    stage, dest = stage_provider(data_root, 'fred')
    try:
        records = []
        for _, path in prepared:
            shutil.copyfile(path, stage / path.name)
            records.append(file_record(stage / path.name, f'raw/fred/{path.name}', 'raw_archive'))
        shutil.copyfile(source / 'manifest.json', stage / 'source_manifest.json')
        records.append(file_record(stage / 'source_manifest.json', 'raw/fred/source_manifest.json', 'source_metadata'))
        from ..contracts import atomic_write_bytes
        atomic_write_bytes(stage / 'Changes-to-FRED-MD.pdf', changes_pdf)
        records.append(file_record(stage / 'Changes-to-FRED-MD.pdf', 'raw/fred/Changes-to-FRED-MD.pdf', 'official_metadata'))
        write_json(stage / 'catalog.json', catalog)
        records.append(file_record(stage / 'catalog.json', 'raw/fred/catalog.json', 'derived_metadata'))
        versions, provenance = write_parser_provenance(stage,'fred')
        records.append(provenance)
        acquired_at = datetime.strptime(source.name, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat() if re.fullmatch(r'\d{8}T\d{6}Z', source.name) else datetime.now(timezone.utc).isoformat()
        manifest = manifest_record(provider='fred_md', source_urls=[FRED_PAGE, *(x['url'] for x, _ in prepared), CHANGES_URL], retrieved_at=acquired_at, request_parameters={'vintages_start': '1999-08', 'vintages_end': '2025-12', 'fixed_vintage': '2023-02', 'transformations': 'raw; per-vintage t-code retained'}, library_versions=versions, files=records, coverage={'first_vintage': '1999-08', 'last_vintage': '2025-12', 'vintage_count': len(catalog['vintages']), 'fixed_last_base_month': '2023-01', 'fixed_series_count': 126}, quality={'unknown_groups': [], 'metadata_only_alias_count': len(ALIASES), 'changes_document_retrieved_at': datetime.now(timezone.utc).isoformat()}, warnings=['Historical ZIP vintages can incorporate publisher corrections; this is a retrieved public archive, not proof of original publication bytes.', 'Group 6 is retained for feature-stage exclusion.', 'Series counts differ by vintage; the fixed file has 126 variables versus 127 described in the paper.', 'Assumed availability is a research convention, not an observed release timestamp.'])
        return publish_provider(stage, dest, manifest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def available_vintages(data_root: Path | str, dataset_id: str | None = None) -> list[str]:
    directory, _ = resolve_dataset(data_root, 'fred', dataset_id)
    return sorted(read_json(directory / 'catalog.json')['vintages'])


def load_macro_vintage(data_root: Path | str, vintage_month: str, lag_months: int = 2, dataset_id: str | None = None) -> MacroVintage:
    directory, manifest = resolve_dataset(data_root, 'fred', dataset_id)
    catalog = read_json(directory / 'catalog.json')
    if vintage_month not in catalog['vintages']:
        raise ResearchError(ErrorCode.MISSING_DATA, f'macro vintage absent: {vintage_month}')
    spec = catalog['vintages'][vintage_month]
    with zipfile.ZipFile(confined_path(directory, spec['archive'])) as archive:
        safe_zip_members(archive)
        return parse_macro_csv(archive.read(spec['member']), vintage_month, catalog, lag_months=lag_months, dataset_id=manifest['dataset_id'])
