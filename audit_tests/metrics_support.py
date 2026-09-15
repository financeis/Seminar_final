"""Isolated execution and evidence helpers used only by the audit harness."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_tests.common import jsonable, sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def environment():
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': packages}


def original_hashes(repo):
    baseline = json.loads((Path(repo) / 'reports/paper_audit/environment.json').read_text(encoding='utf-8'))['original_file_sha256']
    actual = {name: sha256(Path(repo) / name) if (Path(repo) / name).is_file() else None for name in baseline}
    return {'baseline': baseline, 'actual': actual, 'matches_baseline': actual == baseline, 'file_count': len(baseline)}


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def install_network_guard():
    def guard(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.sendto'}:
            raise RuntimeError('Audit isolation prohibits network access: ' + event)
    sys.addaudithook(guard)


def run_original_script(repo, filename, cwd, timeout=60):
    """Run the unchanged complete script in a temporary directory, with networking denied."""
    helper = Path(__file__).resolve()
    command = [sys.executable, '-X', 'utf8', str(helper), '--execute', str(Path(repo) / filename)]
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = datetime.now(timezone.utc).isoformat()
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding='utf-8', env=env, timeout=timeout)
    return {'command': command, 'cwd': str(cwd), 'started_at_utc': started,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': done.returncode,
            'stdout': done.stdout, 'stderr': done.stderr, 'source_sha256': sha256(Path(repo) / filename),
            'execution': 'runpy.run_path of unchanged complete source; socket operations denied; matplotlib Agg'}


if __name__ == '__main__':
    install_network_guard()
    if len(sys.argv) != 3 or sys.argv[1] != '--execute':
        raise SystemExit('Internal helper expects --execute ORIGINAL_SCRIPT')
    runpy.run_path(sys.argv[2], run_name='__main__')
