import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Regime mapping based on sequential thinking analysis
regime_mapping = {
    0: "Economic Difficulty",
    1: "Expansionary Growth",
    2: "Stagflationary Pressure",
    3: "Economic Recovery",
    4: "Reflationary Boom",
    5: "Pre-Recession Transition"
}

# Key statistics from our analysis
regime_stats = {
    'Regime': [0, 1, 2, 3, 4, 5],
    'Economic Label': [
        'Economic Difficulty',
        'Expansionary Growth',
        'Stagflationary Pressure',
        'Economic Recovery',
        'Reflationary Boom',
        'Pre-Recession Transition'
    ],
    'Unemployment': [8.33, 4.70, 4.53, 5.17, 5.66, 6.17],
    'Fed Funds Rate': [0.48, 3.32, 4.84, 3.14, 0.49, 2.13],
    'Consumer Sentiment': [69.29, 86.14, 92.60, 87.52, 87.87, 87.58],
    'S&P 500': [1202, 2416, 1872, 1877, 2083, 851]
}

df = pd.DataFrame(regime_stats)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Economic Regime Characteristics Mapping', fontsize=16, fontweight='bold', y=1.02)

# Color palette for regimes
colors = ['#8B0000', '#2E7D32', '#FF8C00', '#1565C0', '#9C27B0', '#795548']

# 1. Unemployment Rate
ax = axes[0, 0]
bars = ax.bar(df['Regime'], df['Unemployment'], color=colors)
ax.set_title('Unemployment Rate by Regime', fontsize=12, fontweight='bold')
ax.set_xlabel('Regime')
ax.set_ylabel('Unemployment Rate (%)')
ax.set_ylim(0, 10)
for i, (regime, label) in enumerate(zip(df['Regime'], df['Economic Label'])):
    ax.text(regime, -0.5, label.replace(' ', '\n'), ha='center', fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 2. Federal Funds Rate
ax = axes[0, 1]
ax.bar(df['Regime'], df['Fed Funds Rate'], color=colors)
ax.set_title('Federal Funds Rate by Regime', fontsize=12, fontweight='bold')
ax.set_xlabel('Regime')
ax.set_ylabel('Fed Funds Rate (%)')
ax.set_ylim(0, 6)
for i, (regime, label) in enumerate(zip(df['Regime'], df['Economic Label'])):
    ax.text(regime, -0.3, label.replace(' ', '\n'), ha='center', fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 3. Consumer Sentiment
ax = axes[0, 2]
ax.bar(df['Regime'], df['Consumer Sentiment'], color=colors)
ax.set_title('Consumer Sentiment by Regime', fontsize=12, fontweight='bold')
ax.set_xlabel('Regime')
ax.set_ylabel('Consumer Sentiment Index')
ax.set_ylim(60, 100)
for i, (regime, label) in enumerate(zip(df['Regime'], df['Economic Label'])):
    ax.text(regime, 58, label.replace(' ', '\n'), ha='center', fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 4. S&P 500 Index
ax = axes[1, 0]
ax.bar(df['Regime'], df['S&P 500'], color=colors)
ax.set_title('S&P 500 Index by Regime', fontsize=12, fontweight='bold')
ax.set_xlabel('Regime')
ax.set_ylabel('S&P 500 Index')
ax.set_ylim(0, 3000)
for i, (regime, label) in enumerate(zip(df['Regime'], df['Economic Label'])):
    ax.text(regime, -150, label.replace(' ', '\n'), ha='center', fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 5. Combined Metrics Heatmap
ax = axes[1, 1]
# Normalize data for heatmap
metrics_norm = df[['Unemployment', 'Fed Funds Rate', 'Consumer Sentiment', 'S&P 500']].T
# Min-max normalize each metric
for idx in metrics_norm.index:
    row = metrics_norm.loc[idx]
    min_val, max_val = row.min(), row.max()
    if max_val > min_val:
        metrics_norm.loc[idx] = (row - min_val) / (max_val - min_val)

sns.heatmap(metrics_norm, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=df['Regime'], yticklabels=['Unemployment', 'Fed Rate', 'Sentiment', 'S&P 500'],
            cbar_kws={'label': 'Normalized Value'}, ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Economic Indicators Heatmap', fontsize=12, fontweight='bold')

# 6. Summary Table
ax = axes[1, 2]
ax.axis('tight')
ax.axis('off')

table_data = []
for _, row in df.iterrows():
    table_data.append([
        f"Regime {row['Regime']}",
        row['Economic Label'],
        f"{row['Unemployment']:.1f}%",
        f"{row['Fed Funds Rate']:.1f}%",
        f"{row['Consumer Sentiment']:.0f}",
        f"{row['S&P 500']:.0f}"
    ])

table = ax.table(cellText=table_data,
                colLabels=['Regime', 'Economic Phase', 'Unemp.', 'Fed Rate', 'Sentiment', 'S&P 500'],
                cellLoc='center',
                loc='center',
                colWidths=[0.1, 0.35, 0.1, 0.1, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Color code the table
for i in range(len(table_data) + 1):
    for j in range(6):
        cell = table[i, j]
        if i == 0:  # Header
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold')
        else:  # Data rows
            if j == 0 or j == 1:  # Regime number and label
                cell.set_facecolor(colors[i-1])
                cell.set_alpha(0.3)
            else:
                cell.set_facecolor('#F5F5F5')

ax.set_title('Regime Mapping Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('regime_economic_mapping.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*60)
print("ECONOMIC REGIME MAPPING")
print("="*60)
for regime, label in regime_mapping.items():
    print(f"Regime {regime} → {label}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("""
1. Regime 0 (Economic Difficulty): Crisis periods with 8.3% unemployment
   - Includes 2008 Financial Crisis, COVID-19 pandemic
   - Zero interest rate policy (0.48%)

2. Regime 1 (Expansionary Growth): Strong bull markets
   - Lowest unemployment (4.7%)
   - Highest stock market (S&P 2416)

3. Regime 2 (Stagflationary Pressure): Inflation fighting mode
   - Highest interest rates (4.84%)
   - Despite low unemployment, stocks underperform

4. Regime 3 (Economic Recovery): Post-crisis recovery
   - Balanced indicators
   - Building consumer confidence

5. Regime 4 (Reflationary Boom): QE-driven growth
   - Near-zero rates (0.49%) with strong stocks
   - Accommodative monetary policy

6. Regime 5 (Pre-Recession Transition): Warning signals
   - Lowest stock market (S&P 851)
   - Rising unemployment trend
""")