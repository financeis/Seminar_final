"""Offline manifest verification and confined archive reads."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile
from urllib.parse import urlsplit

from ..contracts import SCHEMA_VERSION, ErrorCode, ResearchError, canonical_id, sha256_file, write_json


def confined_path(root: Path | str, relative: str) -> Path:
    rel = PurePosixPath(relative.replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts or ':' in relative:
        raise ResearchError(ErrorCode.MISSING_DATA, f'unsafe relative path: {relative}')
    root = Path(root).resolve()
    path = (root / str(rel)).resolve()
    if not path.is_relative_to(root):
        raise ResearchError(ErrorCode.MISSING_DATA, f'path outside data root: {relative}')
    return path


def safe_zip_members(archive: zipfile.ZipFile) -> list[str]:
    """Validate every member, including unused ones; never extract arbitrarily."""
    names = []
    for item in archive.infolist():
        name = item.filename.replace('\\', '/')
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or ':' in name or stat.S_ISLNK(item.external_attr >> 16):
            raise ResearchError(ErrorCode.MISSING_DATA, f'unsafe ZIP member: {item.filename}')
        if name in names:
            raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate ZIP member: {name}')
        if item.file_size > 100_000_000 or item.file_size / max(item.compress_size, 1) > 1000:
            raise ResearchError(ErrorCode.MISSING_DATA, f'oversized ZIP member: {name}')
        names.append(name)
    return names


def read_json(path: Path | str) -> dict:
    def bad_constant(value):
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'JSON constant: {value}')
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad_constant)
        require_finite_json(value)
        return value
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchError(ErrorCode.MISSING_DATA, f'cannot read {path}: {exc}') from exc


def require_finite_json(value, location: str = '$') -> None:
    """JSON exponent overflow also becomes infinity without parse_constant."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ResearchError(ErrorCode.NONFINITE_VALUE, f'nonfinite JSON number at {location}')
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f'{location}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, f'{location}[{index}]')


def validate_manifest_contract(manifest: dict) -> None:
    """Validate R12 structure before any file reads or content hashing."""
    def check(ok, reason):
        if not ok:
            raise ResearchError(ErrorCode.MISSING_DATA, f'manifest contract: {reason}')
    check(isinstance(manifest, dict), 'must be a JSON object')
    required = {'schema_version','dataset_id','provider','source_urls','retrieved_at','request_parameters','library_versions','files','tickers','coverage','quality','warnings'}
    check(required <= set(manifest), f'missing required fields: {sorted(required-set(manifest))}')
    require_finite_json(manifest)
    check(manifest['schema_version'] == SCHEMA_VERSION, f'unsupported schema_version: {manifest["schema_version"]!r}')
    check(isinstance(manifest['dataset_id'], str) and re.fullmatch('[0-9a-f]{64}', manifest['dataset_id']), 'dataset_id must be a SHA256 content ID')
    check(isinstance(manifest['provider'], str) and bool(manifest['provider'].strip()), 'provider must be a nonempty string')
    urls = manifest['source_urls']
    check(isinstance(urls, list) and bool(urls), 'source_urls must be a nonempty string array')
    for url in urls:
        check(isinstance(url, str), 'source_urls members must be strings')
        try:
            parsed_url = urlsplit(url)
            valid_url = parsed_url.scheme in ('http','https') and bool(parsed_url.hostname)
        except ValueError:
            valid_url = False
        check(valid_url, 'source_urls members must be absolute HTTP(S) URLs')
    stamp = manifest['retrieved_at']
    check(isinstance(stamp, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})', stamp), 'retrieved_at must be an offset ISO timestamp')
    try:
        timestamp = datetime.fromisoformat(stamp)
        check(timestamp.tzinfo is not None and timestamp.utcoffset() is not None, 'retrieved_at requires a timezone offset')
    except ValueError as exc:
        raise ResearchError(ErrorCode.MISSING_DATA, 'manifest contract: invalid retrieved_at date/time') from exc
    for name in ('request_parameters','coverage','quality'):
        check(isinstance(manifest[name], dict), f'{name} must be an object')
    versions = manifest['library_versions']
    check(isinstance(versions, dict) and bool(versions), 'library_versions must be a nonempty object')
    check(all(isinstance(k,str) and k.strip() and isinstance(v,str) and re.fullmatch(r'\d+(?:\.\d+)*(?:[A-Za-z0-9.+_-]*)',v) for k,v in versions.items()), 'library_versions must map library names to actual version numbers')
    for name in ('tickers','warnings'):
        check(isinstance(manifest[name], list) and all(isinstance(x,str) and x.strip() for x in manifest[name]), f'{name} must be a string array')
    check(len(set(manifest['tickers'])) == len(manifest['tickers']), 'tickers must not contain duplicates')
    records = manifest['files']
    check(isinstance(records,list) and bool(records), 'files must be a nonempty array')
    for index, entry in enumerate(records):
        check(isinstance(entry,dict), f'files[{index}] must be an object')
        check({'path','sha256','bytes','kind'} <= set(entry), f'files[{index}] lacks required file fields')
        check(isinstance(entry['path'],str) and bool(entry['path'].strip()), f'files[{index}].path must be a nonempty relative path')
        rel = PurePosixPath(entry['path'].replace('\\','/'))
        check(not rel.is_absolute() and '..' not in rel.parts and ':' not in entry['path'], f'files[{index}].path must stay within the data root')
        check(isinstance(entry['sha256'],str) and re.fullmatch('[0-9a-f]{64}',entry['sha256']), f'files[{index}].sha256 must be a SHA256 digest')
        check(type(entry['bytes']) is int and entry['bytes'] >= 0, f'files[{index}].bytes must be a nonnegative integer')
        check(isinstance(entry['kind'],str) and bool(entry['kind'].strip()), f'files[{index}].kind must be a nonempty string')


def verify_manifest(data_root: Path | str, manifest: dict | Path | str) -> dict:
    if not isinstance(manifest, dict):
        manifest = read_json(manifest)
    validate_manifest_contract(manifest)
    seen = set()
    for entry in manifest['files']:
        if entry['path'] in seen:
            raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate manifest file: {entry["path"]}')
        seen.add(entry['path'])
        path = confined_path(data_root, entry['path'])
        if not path.is_file():
            raise ResearchError(ErrorCode.MISSING_DATA, f'required snapshot file absent: {path}')
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ResearchError(ErrorCode.HASH_MISMATCH, f'snapshot changed: {path}')
    for entry in manifest['files']:
        if entry['kind'] == 'derived_metadata' and Path(entry['path']).name == 'parser_provenance.json':
            provenance = read_json(confined_path(data_root,entry['path']))
            if not isinstance(provenance,dict) or provenance.get('library_versions') != manifest['library_versions']:
                raise ResearchError(ErrorCode.HASH_MISMATCH, 'manifest library versions differ from hashed parser provenance')
    expected = canonical_id({'source_hashes': sorted(x['sha256'] for x in manifest['files']), 'request_parameters': manifest['request_parameters']})
    if manifest['dataset_id'] != expected:
        raise ResearchError(ErrorCode.HASH_MISMATCH, 'dataset content ID mismatch')
    return manifest


def file_record(path: Path, relative: str, kind: str) -> dict:
    return {'path': relative, 'sha256': sha256_file(path), 'bytes': path.stat().st_size, 'kind': kind}


def write_parser_provenance(stage: Path, provider: str, *, previous_dataset_id: str = '', correction_reason: str = '') -> tuple[dict, dict]:
    """Make parser environment metadata part of the content-addressed dataset."""
    import platform
    import numpy as np
    import pandas as pd
    versions = {'python':platform.python_version(), 'pandas':pd.__version__, 'numpy':np.__version__}
    metadata = {'schema_version':SCHEMA_VERSION, 'kind':'parser_provenance', 'library_versions':versions}
    if previous_dataset_id:
        if not re.fullmatch('[0-9a-f]{64}',previous_dataset_id) or not correction_reason.strip():
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'metadata revision requires previous dataset ID and correction reason')
        metadata.update(previous_dataset_id=previous_dataset_id, correction_reason=correction_reason)
    path = stage/'parser_provenance.json'
    write_json(path, metadata)
    return versions, file_record(path,f'raw/{provider}/parser_provenance.json','derived_metadata')


def require_complete_provider(data_root: Path | str, provider: str) -> None:
    """Detect unfinished snapshots before acquisition; never repair their files.

    Completion, byte counts and hashes are inspected for every snapshot.
    The selected snapshot also undergoes the full current metadata contract,
    allowing preserved older metadata-schema snapshots to coexist.
    """
    directory = Path(data_root)/'raw'/provider
    if not directory.exists():
        return
    def incomplete(reason):
        raise ResearchError(ErrorCode.MISSING_DATA, f'incomplete {provider} provider: {reason}; use a new data root')
    if not directory.is_dir():
        incomplete('provider path is not a directory')
    entries = list(directory.iterdir())
    if not entries:
        incomplete('empty provider directory')
    for snapshot in entries:
        if not snapshot.is_dir() or not re.fullmatch('[0-9a-f]{64}',snapshot.name):
            incomplete(f'unrecognized snapshot entry {snapshot.name}')
        path = snapshot/'dataset_manifest.json'
        if not path.is_file():
            incomplete(f'manifest absent for {snapshot.name}')
        manifest = read_json(path)
        if not isinstance(manifest,dict) or not isinstance(manifest.get('files'),list) or not manifest['files']:
            incomplete(f'file inventory absent for {snapshot.name}')
        for entry in manifest['files']:
            if not isinstance(entry,dict) or not isinstance(entry.get('path'),str):
                incomplete(f'invalid file inventory for {snapshot.name}')
            target = confined_path(data_root,entry['path'])
            if not target.is_relative_to(snapshot.resolve()) or not target.is_file():
                incomplete(f'file absent/outside snapshot: {entry["path"]}')
            if type(entry.get('bytes')) is not int or entry['bytes'] < 0 or not isinstance(entry.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256']):
                incomplete(f'byte count/hash absent or invalid: {entry["path"]}')
            if target.stat().st_size != entry['bytes'] or sha256_file(target) != entry['sha256']:
                raise ResearchError(ErrorCode.HASH_MISMATCH, f'existing snapshot changed: {target}; acquisition refused')


def manifest_record(*, provider: str, source_urls: list[str], retrieved_at: str, request_parameters: dict, library_versions: dict, files: list[dict], tickers: tuple | list = (), coverage: dict, quality: dict, warnings: list) -> dict:
    return {
        'schema_version': SCHEMA_VERSION,
        'dataset_id': canonical_id({'source_hashes': sorted(x['sha256'] for x in files), 'request_parameters': request_parameters}),
        'provider': provider, 'source_urls': source_urls, 'retrieved_at': retrieved_at,
        'request_parameters': request_parameters, 'library_versions': library_versions,
        'files': files, 'tickers': list(tickers), 'coverage': coverage,
        'quality': quality, 'warnings': warnings,
    }


def stage_provider(root: Path | str, provider: str) -> tuple[Path, Path]:
    dest = Path(root) / 'raw' / provider
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.parent / f'.{provider}-{uuid.uuid4().hex}'
    stage.mkdir()
    return stage, dest


def publish_provider(stage: Path, dest: Path, manifest: dict) -> dict:
    identifier = manifest['dataset_id']
    manifest['files'] = [{**item, 'path': f'raw/{dest.name}/{identifier}/{Path(item["path"]).name}'} for item in manifest['files']]
    validate_manifest_contract(manifest)
    dest = dest / identifier
    write_json(stage / 'dataset_manifest.json', manifest)
    if dest.exists():
        raise ResearchError(ErrorCode.RUN_CONFLICT, f'provider snapshot already exists: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(dest)
    return manifest


def resolve_dataset(data_root: Path | str, provider: str, dataset_id: str | None = None) -> tuple[Path, dict]:
    """Resolve an explicit content ID, or the sole dataset; never select latest."""
    root = Path(data_root)
    provider_root = root / 'raw' / provider
    if dataset_id:
        if not re.fullmatch('[0-9a-f]{64}', dataset_id):
            raise ResearchError(ErrorCode.INVALID_CONFIG, 'dataset ID must be a SHA256 content identifier')
        path = provider_root / dataset_id / 'dataset_manifest.json'
    else:
        candidates = list(provider_root.glob('*/dataset_manifest.json'))
        if len(candidates) != 1:
            raise ResearchError(ErrorCode.MISSING_DATA, f'{provider}: expected one snapshot or explicit dataset ID, found {len(candidates)}')
        path = candidates[0]
    manifest = verify_manifest(root, path)
    if manifest['dataset_id'] != path.parent.name or dataset_id and manifest['dataset_id'] != dataset_id:
        raise ResearchError(ErrorCode.HASH_MISMATCH, 'requested dataset ID differs from manifest/location')
    return path.parent, manifest


def source_files(source: Path, source_manifest: dict) -> list[tuple[dict, Path]]:
    result = []
    seen = set()
    for item in source_manifest['files']:
        name = Path(item['path']).name
        if name in seen:
            raise ResearchError(ErrorCode.DUPLICATE_KEY, f'duplicate source filename: {name}')
        seen.add(name)
        path = source / name
        if not path.is_file():
            raise ResearchError(ErrorCode.MISSING_DATA, f'prepared file absent: {path}')
        if sha256_file(path) != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
            raise ResearchError(ErrorCode.HASH_MISMATCH, f'prepared file changed: {path}')
        result.append((item, path))
    return result
