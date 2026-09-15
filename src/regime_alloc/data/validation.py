"""Offline complete snapshot checks. Validation does not download or write."""
from pathlib import Path
import zipfile

from ..config import DataConfig, ResearchConfig
from ..contracts import ErrorCode, ResearchError
from .io import read_json, verify_manifest, safe_zip_members, resolve_dataset
from .yahoo import load_prices, validate_prices
from .fred import parse_macro_csv
from .acquisition import load_nber
from .calendar import monthly_returns, decision_ledger, month_range


def validate(config: ResearchConfig | DataConfig | Path | str, *, all_vintages: bool = True) -> dict:
    root = config.data.root if isinstance(config, ResearchConfig) else config.root if isinstance(config, DataConfig) else Path(config)
    research = config.research if isinstance(config, ResearchConfig) else ResearchConfig().research
    data_config = config.data if isinstance(config, ResearchConfig) else config if isinstance(config, DataConfig) else DataConfig(root=root)
    resolved = {name: resolve_dataset(root, name, getattr(data_config,f'{name}_dataset_id') or None) for name in ('yahoo','fred','nber')}
    manifests = {name: result[1] for name,result in resolved.items()}
    prices = load_prices(root, manifests['yahoo']['dataset_id'])
    quality = validate_prices(prices)
    returns = monthly_returns(prices, research.start_month, research.end_month)
    catalog = read_json(resolved['fred'][0]/'catalog.json')
    checked = 0
    specs = catalog['vintages'] if all_vintages else {data_config.fixed_vintage: catalog['vintages'][data_config.fixed_vintage]}
    by_archive = {}
    for month, spec in specs.items():
        by_archive.setdefault(spec['archive'], []).append((month,spec))
    for archive_name, rows in by_archive.items():
        with zipfile.ZipFile(resolved['fred'][0]/archive_name) as archive:
            safe_zip_members(archive)
            for month, spec in rows:
                parsed = parse_macro_csv(archive.read(spec['member']), month, catalog, lag_months=research.lag_months)
                if len(parsed.values.columns) != spec['series_count']:
                    raise ResearchError(ErrorCode.MISSING_DATA, f'catalog series count mismatch in {month}')
                checked += 1
    ledgers = [decision_ledger(month, catalog['vintages'], profile=research.profile, lag_months=research.lag_months, train_months=research.train_months, fixed_vintage=data_config.fixed_vintage) for month in month_range(research.start_month, research.end_month)]
    # Training targets, as well as evaluated targets, require exact first prices.
    monthly_returns(prices, ledgers[0]['training_rows'][0]['target_month'], research.end_month)
    nber = load_nber(root, manifests['nber']['dataset_id'])
    return {'status': 'succeeded', 'offline': True, 'datasets': {name: item['dataset_id'] for name, item in manifests.items()}, 'prices': quality, 'checked_macro_vintages': checked, 'return_months': len(returns), 'first_decision_month': research.start_month, 'last_holding_month': research.end_month, 'required_last_price_session': ledgers[-1]['target_return_end_at'][:10], 'nber_rows': len(nber), 'ledger_checks': sum(len(x['checks']) for x in ledgers)}
