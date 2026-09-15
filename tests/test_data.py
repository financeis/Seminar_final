import io
import json
import zipfile
import pandas as pd
import pytest
from regime_alloc.contracts import TICKERS, ResearchError
from regime_alloc.data import validate_prices, verify_manifest, safe_zip_members, acquire_yahoo
from regime_alloc.config import DataConfig


def prices():
    return pd.DataFrame([dict(session='2023-01-03', ticker=t, open=100., high=101., low=99., close=100., adj_close=100., volume=1., dividends=0., splits=0., capital_gains=0., finalized=True) for t in TICKERS])


def test_price_keys_and_required_assets():
    p = prices()
    validate_prices(p)
    with pytest.raises(ResearchError, match='duplicate_key'):
        validate_prices(pd.concat([p,p.iloc[:1]]))
    with pytest.raises(ResearchError, match='missing_data'):
        validate_prices(p.iloc[:-1])
    p.loc[0,'adj_close'] = float('inf')
    with pytest.raises(ResearchError, match='nonfinite_value'):
        validate_prices(p)


def test_manifest_hash_change(tmp_path):
    (tmp_path/'x.csv').write_text('data')
    manifest = {'files': [{'path': 'x.csv', 'sha256': '0'*64, 'bytes': 4}]}
    with pytest.raises(ResearchError, match='hash_mismatch'):
        verify_manifest(tmp_path, manifest)


@pytest.mark.parametrize('name', ['../evil.csv', '/absolute.csv', 'C:/evil.csv', 'safe/../../evil.csv', '..\\evil.csv'])
def test_zip_escape_rejected(name):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        z.writestr(name, 'bad')
    b.seek(0)
    with zipfile.ZipFile(b) as z, pytest.raises(ResearchError):
        safe_zip_members(z)


def test_download_failure_writes_no_synthetic_data(tmp_path):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise OSError('provider unavailable')
    with pytest.raises(ResearchError, match='missing_data'):
        acquire_yahoo(DataConfig(root=tmp_path, retry_delays=(0,0,0)), downloader=fail)
    assert len(calls) == 4  # initial attempt plus at most three retries
    assert not list(tmp_path.rglob('*.csv'))
    assert calls[0]['auto_adjust'] is False
    assert calls[0]['keepna'] is True


def test_macro_raw_values_tcode_and_group_remain_vintage_local():
    from regime_alloc.data.fred import parse_macro_csv
    # Synthetic unit fixture, deliberately distinct observations/t-codes across vintages.
    catalog = {'series': {'AMBSL': {'group': 5, 'observed_vintages': ['2002-12']}}}
    content = b'sasdate,AMBSL\nTransform:,6\n1/1/2000,10\n2/1/2000,12\n'
    result = parse_macro_csv(content, '2002-12', catalog)
    assert result.values.loc['2000-02','AMBSL'] == 12
    assert result.tcodes['AMBSL'] == 6
    assert result.groups['AMBSL'] == 5
    assert result.assumed_available_at == '2003-02-01T00:00:00-05:00'
    assert result.long_frame()['value_kind'].unique().tolist() == ['raw']
    with pytest.raises(ResearchError, match='missing_data'):
        parse_macro_csv(content, '2002-12', {'series': {}})
    with pytest.raises(ResearchError, match='calendar_gap'):
        parse_macro_csv(content.replace(b'2/1/2000',b'3/1/2000'), '2002-12', catalog)
    with pytest.raises(ResearchError, match='duplicate_key'):
        parse_macro_csv(content.replace(b'2/1/2000',b'1/1/2000'), '2002-12', catalog)


def test_optional_events_are_null_and_intraday_never_final(tmp_path):
    from regime_alloc.data.yahoo import normalize_yahoo
    files = []
    for ticker in TICKERS:
        path = tmp_path/f'{ticker}.csv'
        path.write_text('date,Open,High,Low,Close,Adj Close,Volume\n2026-09-15,100,101,99,100,100,1\n')
        files.append((ticker,path))
    p = normalize_yahoo(files, '2026-09-15T17:38:58+00:00')
    assert not p.finalized.any()
    assert p[['dividends','splits','capital_gains']].isna().all().all()


def test_partial_provider_success_is_still_failure_without_published_files(tmp_path):
    def partial(ticker, **kwargs):
        return pd.DataFrame({'Open':[1.]}) if ticker == 'SPY' else pd.DataFrame()
    with pytest.raises(ResearchError, match='missing_data'):
        acquire_yahoo(DataConfig(root=tmp_path, retry_delays=()), downloader=partial)
    assert not list(tmp_path.rglob('*.csv'))


def test_content_id_resolution_never_selects_latest(tmp_path):
    from regime_alloc.data.io import stage_provider, publish_provider, manifest_record, file_record, resolve_dataset
    from regime_alloc.contracts import atomic_write_bytes
    ids = []
    for data in (b'first', b'second'):
        stage, dest = stage_provider(tmp_path, 'fixture')
        atomic_write_bytes(stage/'raw.txt', data)
        m = manifest_record(provider='test_fixture', source_urls=[], retrieved_at='2020-01-01T00:00:00+00:00', request_parameters={}, library_versions={}, files=[file_record(stage/'raw.txt','raw/fixture/raw.txt','test_fixture')], coverage={}, quality={}, warnings=[])
        publish_provider(stage,dest,m)
        ids.append(m['dataset_id'])
    with pytest.raises(ResearchError, match='missing_data'):
        resolve_dataset(tmp_path,'fixture')
    directory, manifest = resolve_dataset(tmp_path,'fixture',ids[0])
    assert directory.name == ids[0]
    assert manifest['files'][0]['path'] == f'raw/fixture/{ids[0]}/raw.txt'
    with pytest.raises(ResearchError, match='missing_data'):
        resolve_dataset(tmp_path,'fixture','0'*64)


@pytest.mark.real_data
def test_acquired_ten_assets_and_fixed_snapshot():
    import os
    from pathlib import Path
    from regime_alloc.data import load_prices, load_macro_vintage, available_vintages, load_nber
    value = os.environ.get('REGIME_DATA_ROOT')
    if not value:
        pytest.skip('set REGIME_DATA_ROOT to verify acquired immutable real data')
    root = Path(value)
    p = load_prices(root)
    assert len(p) == 71230
    assert p.ticker.unique().tolist() == list(TICKERS)
    assert p.session.min() == '1993-01-29'
    assert p.session.max() == '2026-09-15'
    assert (~p.finalized).sum() == 10
    from regime_alloc.data.io import resolve_dataset
    _, m = resolve_dataset(root, 'yahoo')
    assert len([x for x in m['files'] if x['kind'] == 'raw_price']) == 10
    assert next(x['sha256'] for x in m['files'] if x['path'].endswith('/SPY.csv')) == 'edd58ea6d8432511b02735e6823e0327956ae1b21b06f1c1299378ab2aa565ee'
    fixed = load_macro_vintage(root,'2023-02')
    assert fixed.values.shape == (769,126)
    assert fixed.values.index[-1] == '2023-01'
    old = load_macro_vintage(root,'2002-12')
    assert old.values.shape[1] == 124
    assert 'AMBSL' in old.values and 'BOGMBASE' not in old.values
    assert old.groups['AMBSL'] == 5
    assert len(available_vintages(root)) == 317
    rec = load_nber(root).set_index('month').usrec
    assert rec.loc['2020-03'] == 1 and rec.loc['2020-05'] == 0
