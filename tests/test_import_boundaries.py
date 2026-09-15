"""Importing the research library must not start a research run."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_package_imports_without_io_or_historical_code(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    script = textwrap.dedent(
        """
        import importlib
        import json
        import os
        import pkgutil
        import sys

        def guard(event, arguments):
            if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
                raise AssertionError(f"network access during import: {event}")
            if event in {"os.mkdir", "os.remove", "os.rename", "os.rmdir"}:
                raise AssertionError(f"filesystem mutation during import: {event}")
            if event == "open":
                path, mode, flags = arguments
                writes = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
                if flags & writes:
                    raise AssertionError(f"file creation/write during import: {path}")

        sys.addaudithook(guard)
        import regime_alloc
        names = sorted(module.name for module in pkgutil.walk_packages(
            regime_alloc.__path__, regime_alloc.__name__ + "."
        ))
        for name in names:
            importlib.import_module(name)
        forbidden = [name for name in sys.modules if
                     name.split(".")[0] in {"audit_tests", "legacy"}
                     or name.split(".")[0].startswith("Step")]
        assert not forbidden, forbidden
        assert "regime_alloc.data.acquisition" in names
        print(json.dumps({"imported": names, "status": "pass"}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "pass"' in result.stdout
    assert list(tmp_path.iterdir()) == []
