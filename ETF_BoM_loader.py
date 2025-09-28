# ETF Beginning-of-Month (BoM) monthly returns loader + alignment to macro calendar (per Oliveira et al., Sec. 5–6)
# - Uses the 10 ETFs in Table 2: SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY.  :contentReference[oaicite:3]{index=3}
# - Computes BoM returns (first trading day to next first trading day).  :contentReference[oaicite:4]{index=4}
# - Aligns the return matrix to the Section 3 monthly macro calendar (first-of-month stamps).

import pandas as pd, numpy as np
from pathlib import Path

TICKERS = ["SPY","XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]

# ---------- (A) Macro calendar (prefers Section 3 artifact) ----------
def load_macro_calendar_from_section3(
    labels_csv="section3_1_regime_labels.csv",
    fred_md_csv="FRED-MD_2024m12.csv",
    window=("1978-07-01","2024-09-01")
) -> pd.DatetimeIndex:
    """
    Load the monthly macro calendar (first-of-month timestamps) used in Section 3.
    Priority:
      (1) Read 'sasdate' from section3_1_regime_labels.csv (already complete-case months).
      (2) Fallback: read FRED-MD dates within the window (only for calendar stamps).
    """
    p = Path(labels_csv)
    if p.exists():
        lab = pd.read_csv(p, parse_dates=["sasdate"])
        return pd.DatetimeIndex(pd.to_datetime(lab["sasdate"].unique())).sort_values()
    # Fallback calendar (no transforms here; only need first-of-month stamps)
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    raw = pd.read_csv(fred_md_csv, header=0, skiprows=[1])
    raw["sasdate"] = pd.to_datetime(raw["sasdate"], errors="coerce")
    raw = raw.dropna(subset=["sasdate"]).sort_values("sasdate")
    raw = raw[(raw["sasdate"] >= start) & (raw["sasdate"] <= end)]
    return pd.DatetimeIndex(pd.to_datetime(raw["sasdate"].unique())).sort_values()

# ---------- (B) Compute BoM returns from DAILY adjusted-close prices ----------
def compute_bom_returns_from_daily(
    prices: pd.DataFrame,
    tickers=TICKERS,
    date_col="date",
    ticker_col="ticker",
    price_col="adj_close"
) -> pd.DataFrame:
    """
    BoM returns (per paper): for each (ticker, month), take the FIRST TRADING DAY price (min(date) within month).
    Monthly return at month t indexed by month-start date:
        R_t = P(first_day_{t+1}) / P(first_day_{t}) - 1
    Returns: DataFrame (index = month first-of-month; columns = tickers).  :contentReference[oaicite:5]{index=5}
    """
    df = prices[[date_col, ticker_col, price_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([ticker_col, date_col])
    df = df[df[ticker_col].isin(tickers)].copy()

    # Month key = first day of that month (first-of-month stamp)
    df["month"] = df[date_col].values.astype("datetime64[M]")
    # Get FIRST trading day per (ticker, month)
    idx = df.groupby([ticker_col, "month"])[date_col].idxmin()
    firsts = df.loc[idx, [date_col, "month", ticker_col, price_col]].sort_values([ticker_col, "month"])

    # Wide matrix of first-day prices
    P = firsts.pivot(index="month", columns=ticker_col, values=price_col).sort_index()

    # BoM returns: next month's first price over current month's first price minus 1
    R = P.shift(-1) / P - 1.0
    R = R.iloc[:-1]                    # last row has no forward month
    R = R.reindex(columns=tickers)     # enforce column order
    return R

# ---------- (C) Align to macro calendar (require complete 10-ETF panel per month) ----------
def align_to_macro_calendar(
    R_bom: pd.DataFrame,
    macro_calendar: pd.DatetimeIndex,
    require_all_tickers=True
) -> pd.DataFrame:
    """
    Intersect the BoM return months with the Section 3 macro calendar (first-of-month stamps).
    If require_all_tickers=True, drop months with any NaN across the 10 ETFs
    (paper uses a fixed, complete set of ETFs in the backtest). :contentReference[oaicite:6]{index=6}
    """
    R = R_bom.copy()
    R.index = pd.DatetimeIndex(R.index)    # ensure plain timestamps
    idx = pd.DatetimeIndex(sorted(set(R.index).intersection(set(macro_calendar))))
    R_aligned = R.loc[idx].sort_index()
    if require_all_tickers:
        R_aligned = R_aligned.dropna(axis=0, how="any")
    return R_aligned

# ====================== REAL DATA using yfinance ======================
# Import yfinance for fetching real ETF data
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(["pip", "install", "yfinance"])
    import yfinance as yf

def _fetch_real_prices():
    """
    Fetch real daily adjusted close prices for the 10 ETFs using yfinance.
    Returns DataFrame with columns: date, ticker, adj_close
    """
    # Define start date (earliest ETF inception is SPY in 1993, but sectors started in 1998)
    start_date = "1993-01-01"
    end_date = "2024-10-31"

    rows = []
    for tkr in TICKERS:
        try:
            # Download data for each ticker
            print(f"Downloading {tkr}...")
            ticker = yf.Ticker(tkr)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if not data.empty:
                # Reset index to get date as column
                data = data.reset_index()
                # Create dataframe in expected format
                # Use 'Close' column since auto_adjust=True gives adjusted prices
                ticker_df = pd.DataFrame({
                    "date": data["Date"],
                    "ticker": tkr,
                    "adj_close": data["Close"]
                })
                rows.append(ticker_df)
                print(f"  Downloaded {len(ticker_df)} days of data for {tkr}")
            else:
                print(f"Warning: No data retrieved for {tkr}")

        except Exception as e:
            print(f"Error downloading {tkr}: {e}")
            # Skip this ticker if download fails
            continue

    if rows:
        print(f"\nTotal ETF data rows: {sum(len(r) for r in rows)}")
        return pd.concat(rows, ignore_index=True)
    else:
        raise RuntimeError("Failed to download any ETF data")

# For backward compatibility, keep the old function name but use real data
def _make_synth_prices():
    """
    This function now fetches real prices instead of synthetic data.
    Keeping the name for compatibility with existing code.
    """
    return _fetch_real_prices()

# Demo run
macro_calendar = load_macro_calendar_from_section3()
synthetic_prices = _make_synth_prices()
R_bom = compute_bom_returns_from_daily(synthetic_prices)           # BoM returns
R_aligned = align_to_macro_calendar(R_bom, macro_calendar, True)   # align & require full 10-ETF panel

# Save artifacts
R_bom.to_csv("etf_bom_returns_raw_demo.csv")
R_aligned.to_csv("etf_bom_returns_aligned_demo.csv")

print("=== Summary ===")
print(f"Macro calendar: {len(macro_calendar)} months | {macro_calendar.min().date()} .. {macro_calendar.max().date()}")
print(f"BoM returns (raw):   shape={R_bom.shape} | range={R_bom.index.min().date()} .. {R_bom.index.max().date()}")
print(f"BoM returns (aligned, complete 10 ETFs/month): shape={R_aligned.shape} | "
      f"range={R_aligned.index.min().date()} .. {R_aligned.index.max().date()}")

print("\nHead (aligned):")
print(R_aligned.head(5).to_string(index=True))
print("\nTail (aligned):")
print(R_aligned.tail(5).to_string(index=True))


print("ETF BoM raw:", R_bom.index.min(), "→", R_bom.index.max(), R_bom.shape)
print("ETF BoM aligned (10 tickers only):", R_aligned.index.min(), "→", R_aligned.index.max(), R_aligned.shape)

