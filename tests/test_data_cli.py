"""Command boundaries: errors stay errors and relative paths stay reproducible."""
import json

from regime_alloc.cli import main


def test_unknown_configuration_returns_two(tmp_path, capsys):
    config = tmp_path / "invalid.toml"
    config.write_text("unexpected = true\n", encoding="utf-8")
    assert main(["data", "validate", "--config", str(config)]) == 2
    assert json.loads(capsys.readouterr().err)["error_code"] == "invalid_config"


def test_missing_data_is_a_failed_offline_command(tmp_path, capsys):
    config = tmp_path / "research.toml"
    config.write_text(
        '[data]\nroot = "absent"\n'
        + '\n'.join(f'{provider}_dataset_id = "' + "0" * 64 + '"' for provider in ("yahoo", "fred", "nber"))
        + '\n',
        encoding="utf-8",
    )
    assert main(["data", "validate", "--config", str(config)]) == 3
    assert json.loads(capsys.readouterr().err)["error_code"] == "missing_data"
    assert not (tmp_path / "absent").exists()


def test_valid_command_uses_config_directory(tmp_path, monkeypatch, capsys):
    from regime_alloc import data

    config = tmp_path / "research.toml"
    config.write_text('[data]\nroot = "prices"\n', encoding="utf-8")
    seen = []

    def validate(c):
        seen.append(c.data.root)
        return {"status": "succeeded"}

    monkeypatch.setattr(data, "validate", validate)
    assert main(["data", "validate", "--config", str(config)]) == 0
    assert seen == [tmp_path / "prices"]
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_provider_exception_never_reports_success(tmp_path, monkeypatch, capsys):
    from regime_alloc import data

    config = tmp_path / "research.toml"
    config.write_text("", encoding="utf-8")

    def acquire(c):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(data, "acquire", acquire)
    assert main(["data", "acquire", "--config", str(config)]) == 3
    assert "provider unavailable" in json.loads(capsys.readouterr().err)["reason"]
