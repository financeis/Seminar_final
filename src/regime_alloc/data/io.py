"""Offline manifest verification and confined archive reads."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

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
        return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchError(ErrorCode.MISSING_DATA, f'cannot read {path}: {exc}') from exc


def verify_manifest(data_root: Path | str, manifest: dict | Path | str) -> dict:
    if not isinstance(manifest, dict):
        manifest = read_json(manifest)
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
    if 'dataset_id' in manifest:
        expected = canonical_id({'source_hashes': sorted(x['sha256'] for x in manifest['files']), 'request_parameters': manifest['request_parameters']})
        if manifest['dataset_id'] != expected:
            raise ResearchError(ErrorCode.HASH_MISMATCH, 'dataset content ID mismatch')
    return manifest


def file_record(path: Path, relative: str, kind: str) -> dict:
    return {'path': relative, 'sha256': sha256_file(path), 'bytes': path.stat().st_size, 'kind': kind}


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
