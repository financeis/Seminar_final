import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("RECALCULATING WITH ACTUAL REGIME LABELS")
print("="*60)

# Read the ACTUAL regime labels from section3_1
regime_labels = pd.read_csv('section3_1_regime_labels.csv')
regime_labels['sasdate'] = pd.to_datetime(regime_labels['sasdate'])

# Read FRED-MD data
fred_df = pd.read_csv('FRED-MD_2024m12.csv', skiprows=[1])
fred_df['sasdate'] = pd.to_datetime(fred_df['sasdate'])

# Merge by date
merged = pd.merge(regime_labels, fred_df, on='sasdate', how='inner')

# Key indicators
indicators = ['RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL', 'S&P 500']
fred_cols = ['RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL']

# Check S&P 500 column name
if 'S&P 500' not in fred_df.columns:
    if 'S&P500' in fred_df.columns:
        fred_cols.append('S&P500')
        indicators[-1] = 'S&P500'
    elif 'SP500' in fred_df.columns:
        fred_cols.append('SP500')
        indicators[-1] = 'SP500'
else:
    fred_cols.append('S&P 500')

print("\n1. REGIME DISTRIBUTION:")
print("-"*40)
regime_counts = merged['final_regime_label'].value_counts().sort_index()
for regime, count in regime_counts.items():
    print(f"Regime {regime}: {count} months")

# Calculate statistics for each regime
print("\n2. ACTUAL STATISTICS BY REGIME:")
print("-"*40)

regime_stats = pd.DataFrame(index=indicators[:len(fred_cols)], columns=range(6))

for regime in range(6):
    regime_data = merged[merged['final_regime_label'] == regime]
    if len(regime_data) > 0:
        print(f"\nRegime {regime} (n={len(regime_data)}):")
        for ind, col in zip(indicators[:len(fred_cols)], fred_cols):
            if col in regime_data.columns:
                data = regime_data[col].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    regime_stats.loc[ind, regime] = mean_val
                    print(f"  {ind:12s}: Mean={mean_val:8.2f}, Std={std_val:8.2f}")

# Convert to numeric
regime_stats = regime_stats.astype(float)

print("\n3. MIN-MAX NORMALIZATION:")
print("-"*40)

# Normalize each row
normalized_stats = regime_stats.copy()
for idx in regime_stats.index:
    row = regime_stats.loc[idx]
    if row.notna().any():
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:
            normalized_stats.loc[idx] = (row - min_val) / (max_val - min_val)
        else:
            normalized_stats.loc[idx] = 0.5

print("\nRaw Statistics:")
print(regime_stats.round(2))
print("\nNormalized Statistics (0=min, 1=max):")
print(normalized_stats.round(3))

# Special focus on unemployment
print("\n4. UNEMPLOYMENT RATE ANALYSIS:")
print("-"*40)
for regime in range(6):
    regime_data = merged[merged['final_regime_label'] == regime]
    if len(regime_data) > 0 and 'UNRATE' in regime_data.columns:
        unrate = regime_data['UNRATE'].dropna()
        if len(unrate) > 0:
            print(f"Regime {regime}: {unrate.mean():.2f}% (normalized: {normalized_stats.loc['UNRATE', regime]:.3f})")

# Check which regime is ACTUALLY the outlier (Regime 0)
print("\n5. REGIME 0 (OUTLIER) CHARACTERISTICS:")
print("-"*40)
outlier_months = merged[merged['is_regime0_outlier'] == True]
print(f"Total outlier months (Regime 0): {len(outlier_months)}")
if len(outlier_months) > 0:
    for col in ['UNRATE', 'FEDFUNDS', 'UMCSENTx']:
        if col in outlier_months.columns:
            data = outlier_months[col].dropna()
            if len(data) > 0:
                print(f"{col}: Mean={data.mean():.2f}, Std={data.std():.2f}")

# Create corrected heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_stats, annot=True, fmt='.2f', cmap='YlGnBu',
            vmin=0, vmax=1, square=True, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'Normalized Value'})

plt.title('CORRECTED Min-Max Normalized Regime Statistics\n(Using actual regime labels)',
          fontsize=14, fontweight='bold')
plt.xlabel('Regime', fontsize=12)
plt.ylabel('FRED Statistic', fontsize=12)
plt.tight_layout()
plt.savefig('corrected_regime_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("CORRECTED heatmap saved as 'corrected_regime_heatmap.png'")
print("="*60)