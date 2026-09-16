"""data components for reproducible research."""
"""Public data API. Importing this package performs no IO or acquisition."""
from .io import verify_manifest, safe_zip_members
from .yahoo import accept_yahoo, acquire_yahoo, load_prices, validate_prices
from .fred import MacroVintage, accept_fred, load_macro_vintage, available_vintages
from .acquisition import acquire, acquire_fred, acquire_nber, load_nber
from .validation import validate

__all__ = ['verify_manifest', 'safe_zip_members', 'accept_yahoo', 'acquire_yahoo', 'load_prices', 'validate_prices', 'MacroVintage', 'accept_fred', 'load_macro_vintage', 'available_vintages', 'acquire', 'acquire_fred', 'acquire_nber', 'load_nber', 'validate']
