import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# File paths
RET_PATH = Path("backtest_returns_full_period.csv")
MET_PATH = Path("backtest_metrics_full_period.csv")

# Read data
returns = pd.read_csv(RET_PATH, index_col=0, parse_dates=True)
metrics = pd.read_csv(MET_PATH)

# Strategy columns mapping
strategy_cols = {
    'Long-only (l=2)': 'ret_lo_2',
    'Long-only (l=3)': 'ret_lo_3',
    'Long-only (l=4)': 'ret_lo_4',
    'SPY Buy&Hold': 'ret_spy',
    'Equal Weight': 'ret_ew'
}

# Colors for each strategy
colors = {
    'Long-only (l=2)': '#2E7D32',
    'Long-only (l=3)': '#1565C0',
    'Long-only (l=4)': '#C62828',
    'SPY Buy&Hold': '#6A1B9A',
    'Equal Weight': '#E65100'
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Tactical Asset Allocation Strategy Backtest Results (2002-2024)', fontsize=16, y=1.02)

# 1. Cumulative Returns
ax1 = plt.subplot(2, 3, 1)
for strategy, col in strategy_cols.items():
    if col in returns.columns:
        cum_ret = (1 + returns[col]).cumprod()
        ax1.plot(returns.index, cum_ret, label=strategy, linewidth=2, color=colors[strategy])
ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 2. Drawdown
ax2 = plt.subplot(2, 3, 2)
for strategy, col in strategy_cols.items():
    if col in returns.columns:
        cum_ret = (1 + returns[col]).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        ax2.fill_between(returns.index, drawdown * 100, 0,
                         alpha=0.3, label=strategy, color=colors[strategy])
ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Drawdown (%)')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 3. Rolling 12-month Sharpe Ratio
ax3 = plt.subplot(2, 3, 3)
window = 12
for strategy, col in strategy_cols.items():
    if col in returns.columns:
        rolling_mean = returns[col].rolling(window=window).mean()
        rolling_std = returns[col].rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
        ax3.plot(returns.index[window-1:], rolling_sharpe[window-1:],
                label=strategy, linewidth=1.5, color=colors[strategy])
ax3.set_title('Rolling 12-Month Sharpe Ratio', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 4. Monthly Return Distribution (Box Plot)
ax4 = plt.subplot(2, 3, 4)
box_data = []
box_labels = []
box_colors_list = []
for strategy, col in strategy_cols.items():
    if col in returns.columns:
        box_data.append(returns[col].dropna() * 100)
        box_labels.append(strategy.replace(' ', '\n'))
        box_colors_list.append(colors[strategy])

bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], box_colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_title('Monthly Return Distribution', fontsize=12, fontweight='bold')
ax4.set_ylabel('Monthly Return (%)')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 5. Risk-Return Scatter
ax5 = plt.subplot(2, 3, 5)
for strategy in strategy_cols.keys():
    metric_row = metrics[metrics['Strategy'] == strategy]
    if not metric_row.empty:
        ann_ret = metric_row['Ann.Return'].values[0]
        ann_vol = metric_row['Ann.Vol'].values[0]
        sharpe = metric_row['Sharpe'].values[0]
        ax5.scatter(ann_vol, ann_ret, s=200, alpha=0.7,
                   color=colors[strategy], edgecolors='black', linewidth=2)
        ax5.annotate(f'{strategy}\n(SR: {sharpe:.2f})',
                    xy=(ann_vol, ann_ret),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')

ax5.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
ax5.set_xlabel('Annualized Volatility (%)')
ax5.set_ylabel('Annualized Return (%)')
ax5.grid(True, alpha=0.3)

# Add capital allocation line (CAL) from risk-free rate (assumed 2%)
rf_rate = 2.0
vol_range = np.linspace(0, 20, 100)
for strategy in strategy_cols.keys():
    metric_row = metrics[metrics['Strategy'] == strategy]
    if not metric_row.empty:
        sharpe = metric_row['Sharpe'].values[0]
        cal_returns = rf_rate + sharpe * vol_range
        ax5.plot(vol_range, cal_returns, '--', alpha=0.3,
                color=colors[strategy], linewidth=1)

# 6. Performance Metrics Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Prepare table data
table_data = []
for strategy in strategy_cols.keys():
    metric_row = metrics[metrics['Strategy'] == strategy]
    if not metric_row.empty:
        row_data = [
            strategy,
            f"{metric_row['Sharpe'].values[0]:.3f}",
            f"{metric_row['Ann.Return'].values[0]:.1f}%",
            f"{metric_row['Ann.Vol'].values[0]:.1f}%",
            f"{metric_row['MaxDD'].values[0]:.1f}%"
        ]
        table_data.append(row_data)

# Create table
table = ax6.table(cellText=table_data,
                 colLabels=['Strategy', 'Sharpe', 'Ann. Return', 'Ann. Vol', 'Max DD'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Style the table
for i in range(len(table_data) + 1):
    for j in range(5):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold')
        else:
            if j == 0:
                cell.set_facecolor(colors[table_data[i-1][0]])
                cell.set_alpha(0.3)
            else:
                cell.set_facecolor('#F5F5F5')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

ax6.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()

# Save figure
plt.savefig('backtest_visualization.png', dpi=300, bbox_inches='tight')
print("Backtest visualization saved as 'backtest_visualization.png'")

# Show plot
plt.show()

# Additional analysis: Crisis period performance
print("\n" + "="*60)
print("Crisis Period Performance Analysis")
print("="*60)

# Define crisis periods
crisis_periods = [
    ('2008 Financial Crisis', '2007-12-01', '2009-06-30'),
    ('COVID-19 Pandemic', '2020-02-01', '2020-12-31'),
    ('2022 Market Correction', '2022-01-01', '2022-12-31')
]

for period_name, start, end in crisis_periods:
    print(f"\n{period_name} ({start} to {end}):")
    print("-" * 40)

    period_mask = (returns.index >= start) & (returns.index <= end)
    period_returns = returns[period_mask]

    if len(period_returns) > 0:
        for strategy, col in strategy_cols.items():
            if col in period_returns.columns:
                total_return = (1 + period_returns[col]).prod() - 1
                avg_return = period_returns[col].mean()
                volatility = period_returns[col].std() * np.sqrt(12)
                print(f"{strategy:20} | Total: {total_return*100:6.2f}% | Avg Monthly: {avg_return*100:5.2f}% | Ann Vol: {volatility:5.2f}%")

print("\n" + "="*60)
print("Strategy Correlation Matrix")
print("="*60)

# Calculate correlation matrix
correlation_matrix = returns[[strategy_cols[s] for s in strategy_cols.keys() if strategy_cols[s] in returns.columns]].corr()
correlation_matrix.columns = [s for s in strategy_cols.keys() if strategy_cols[s] in returns.columns]
correlation_matrix.index = correlation_matrix.columns
print("\n", correlation_matrix.round(3))