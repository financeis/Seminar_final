# Backtest driver (long-only) using the single Step-4 file
# Inputs: section5_step4_longonly_weights.csv + etf_bom_returns_aligned_demo.csv
# Outputs: section5_step5_longonly_backtest_returns.csv + ..._metrics.csv
# Faithful to Eq. (18) sizing and Sec. 5.4 metrics.
#
# ============================================================================
# BACKTESTING PERIOD CONFIGURATION:
# The backtesting period is determined by Section5_step2.py LIMIT_WINDOWS setting:
#   - LIMIT_WINDOWS = None: Full period (260+ months, 2002-2024)
#   - LIMIT_WINDOWS = 60: Last 60 months only (2019-2024)
# To change the period, modify LIMIT_WINDOWS in Section5_step2.py and rerun steps 2-4
# ============================================================================

import pandas as pd, numpy as np, math
from pathlib import Path

W_PATH = Path("section5_step4_longonly_weights.csv")
ETF_RET_PATH = Path("etf_bom_returns_aligned_demo.csv")
RET_OUT = Path("section5_step5_longonly_backtest_returns.csv")
MET_OUT = Path("section5_step5_longonly_backtest_metrics.csv")
TICKERS = ["SPY","XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
L_LIST = [2,3,4]

w = pd.read_csv(W_PATH, parse_dates=["sasdate_t","sasdate_tp1"])
R = pd.read_csv(ETF_RET_PATH, index_col=0, parse_dates=True)

def weight_cols_in_df(df, l):
    cols = [f"w_lo_{l}_{t}" for t in TICKERS]
    if not all(c in df.columns for c in cols):
        raise RuntimeError(f"Missing expected weight columns for l={l}: {cols}")
    return cols

rows=[]
for _, row in w.iterrows():
    t, tp1 = row["sasdate_t"], row["sasdate_tp1"]
    if tp1 not in R.index:
        continue
    ret_tp1 = R.loc[tp1, TICKERS].astype(float).to_numpy()
    rec = {"sasdate_t": t, "sasdate_tp1": tp1}
    for l in L_LIST:
        wvec = row[weight_cols_in_df(w, l)].astype(float).to_numpy()
        rec[f"ret_lo_{l}"] = float(np.nansum(wvec * ret_tp1))
    rec["ret_spy"] = float(R.loc[tp1,"SPY"])
    rec["ret_ew"]  = float(R.loc[tp1, TICKERS].mean())
    rows.append(rec)

rets = pd.DataFrame(rows).sort_values("sasdate_t").set_index("sasdate_tp1")
rets.index.name="Date"
rets.to_csv(RET_OUT)

def compute_drawdowns(r):
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    dd = wealth/wealth.cummax() - 1.0
    return (dd[dd<0].mean() if (dd<0).any() else 0.0), dd.min(), wealth

def perf_metrics(monthly):
    monthly = monthly.dropna()
    mu, sd = monthly.mean(), monthly.std(ddof=1)
    ann = np.sqrt(12.0)
    sharpe = (mu/sd)*ann if sd and np.isfinite(sd) and sd>0 else np.nan
    ddn = monthly[monthly<0.0].std(ddof=1)
    sortino = (mu/ddn)*ann if ddn and np.isfinite(ddn) and ddn>0 else np.nan
    avgdd, maxdd, wealth = compute_drawdowns(monthly)
    ppos = (monthly>0).mean() if len(monthly) else np.nan
    return {"Sharpe":float(sharpe),"Sortino":float(sortino),
            "AvgDD":float(avgdd),"MaxDD":float(maxdd),
            "% Positive Ret.":float(ppos)}

metrics=[]
for l in L_LIST:
    m = perf_metrics(rets[f"ret_lo_{l}"]); m.update({"Model":f"ridge_lo_{l}"}); metrics.append(m)
m_spy = perf_metrics(rets["ret_spy"]); m_spy.update({"Model":"spy"}); metrics.append(m_spy)
m_ew  = perf_metrics(rets["ret_ew"]);  m_ew.update({"Model":"ew"});  metrics.append(m_ew)

pd.DataFrame(metrics).sort_values("Sharpe", ascending=False).to_csv(MET_OUT, index=False)

# (Optional) save simple 10% vol-target cumulative log-return figures for each series...
