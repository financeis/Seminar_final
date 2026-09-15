from regime_alloc.cli import main


def test_run_bad_configuration_has_config_exit(tmp_path, capsys):
    assert main(['run', '--config', str(tmp_path / 'missing.toml'), '--output', str(tmp_path / 'run')]) == 2
    assert 'invalid_config' in capsys.readouterr().err
