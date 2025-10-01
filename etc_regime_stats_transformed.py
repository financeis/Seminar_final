import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for Korean text
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("RECALCULATING WITH TRANSFORMED DATA")
print("="*60)

# Read the ACTUAL regime labels from section3_1
regime_labels = pd.read_csv('section3_1_regime_labels.csv')
regime_labels['sasdate'] = pd.to_datetime(regime_labels['sasdate'])

# Read FRED-MD data WITH transformation codes
fred_df = pd.read_csv('FRED-MD_2024m12.csv', header=0)

# Extract transformation codes (second row)
tcode_row = fred_df.iloc[0]
tcodes = {}
for col in fred_df.columns:
    if col != 'sasdate':
        try:
            tcodes[col] = int(float(tcode_row[col]))
        except:
            tcodes[col] = None

# Remove the transformation code row
fred_df = fred_df.iloc[1:].copy()
fred_df['sasdate'] = pd.to_datetime(fred_df['sasdate'])

# Define transformation functions
def transform_series(x, tcode):
    """Apply FRED-MD transformation codes"""
    x = pd.to_numeric(x, errors='coerce')

    if tcode == 1:  # Levels (no transformation)
        return x
    elif tcode == 2:  # First difference
        return x.diff()
    elif tcode == 3:  # Second difference
        return x.diff().diff()
    elif tcode == 4:  # Natural log
        return np.log(x.replace(0, np.nan))
    elif tcode == 5:  # First difference of natural log
        return np.log(x.replace(0, np.nan)).diff()
    elif tcode == 6:  # Second difference of natural log
        return np.log(x.replace(0, np.nan)).diff().diff()
    elif tcode == 7:  # First difference of percent change
        pct = x.pct_change()
        return pct.diff()
    else:
        return x

# Key indicators and their transformation codes
indicators_mapping = {
    'RPI': 'RPI',           # Real Personal Income
    'UNRATE': 'UNRATE',     # Unemployment Rate
    'UMCSENTx': 'UMCSENTx', # U. of Michigan Consumer Sentiment
    'FEDFUNDS': 'FEDFUNDS', # Federal Funds Rate
    'CPIAUCSL': 'CPIAUCSL', # CPI All Urban Consumers
    'S&P 500': 'S&P 500'    # S&P 500
}

# Check S&P 500 column name
if 'S&P 500' not in fred_df.columns:
    if 'S&P500' in fred_df.columns:
        indicators_mapping['S&P 500'] = 'S&P500'
    elif 'SP500' in fred_df.columns:
        indicators_mapping['S&P 500'] = 'SP500'

# Korean labels for display
korean_labels = {
    'RPI': '실질개인소득',
    'UNRATE': '실업률',
    'UMCSENTx': '소비자심리지수',
    'FEDFUNDS': '시장금리',
    'CPIAUCSL': 'CPI',
    'S&P 500': 'S&P500 지수'
}

# Apply transformations
print("\n1. APPLYING TRANSFORMATIONS:")
print("-"*40)
transformed_data = fred_df[['sasdate']].copy()

for display_name, col_name in indicators_mapping.items():
    if col_name in fred_df.columns and col_name in tcodes:
        tcode = tcodes[col_name]
        print(f"{display_name:12s} ({col_name}): Transformation code = {tcode}")
        transformed_data[display_name] = transform_series(fred_df[col_name], tcode)
    else:
        print(f"{display_name:12s}: Column not found or no tcode")

# Merge with regime labels
merged = pd.merge(regime_labels, transformed_data, on='sasdate', how='inner')

print("\n2. REGIME DISTRIBUTION:")
print("-"*40)
regime_counts = merged['final_regime_label'].value_counts().sort_index()
for regime, count in regime_counts.items():
    print(f"Regime {regime}: {count} months")

# Calculate statistics for each regime (with transformed data)
print("\n3. STATISTICS BY REGIME (TRANSFORMED DATA):")
print("-"*40)

regime_stats = pd.DataFrame(index=list(indicators_mapping.keys()), columns=range(6))

for regime in range(6):
    regime_data = merged[merged['final_regime_label'] == regime]
    if len(regime_data) > 0:
        print(f"\nRegime {regime} (n={len(regime_data)}):")
        for ind in indicators_mapping.keys():
            if ind in regime_data.columns:
                data = regime_data[ind].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    regime_stats.loc[ind, regime] = mean_val
                    print(f"  {ind:12s}: Mean={mean_val:8.4f}, Std={std_val:8.4f}")

# Convert to numeric
regime_stats = regime_stats.astype(float)

print("\n4. MIN-MAX NORMALIZATION:")
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

print("\nTransformed Statistics:")
print(regime_stats.round(4))
print("\nNormalized Statistics (0=min, 1=max):")
print(normalized_stats.round(3))

# Special focus on unemployment (transformed)
print("\n5. UNEMPLOYMENT RATE ANALYSIS (TRANSFORMED):")
print("-"*40)
for regime in range(6):
    regime_data = merged[merged['final_regime_label'] == regime]
    if len(regime_data) > 0 and 'UNRATE' in regime_data.columns:
        unrate = regime_data['UNRATE'].dropna()
        if len(unrate) > 0:
            print(f"Regime {regime}: Mean={unrate.mean():.4f}, Normalized={normalized_stats.loc['UNRATE', regime]:.3f}")

# Check transformation effects
print("\n6. TRANSFORMATION EFFECTS:")
print("-"*40)
print("Transformation codes used:")
for display_name, col_name in indicators_mapping.items():
    if col_name in tcodes:
        tcode = tcodes[col_name]
        tcode_desc = {
            1: "Levels",
            2: "First difference",
            3: "Second difference",
            4: "Natural log",
            5: "First diff of log",
            6: "Second diff of log",
            7: "First diff of percent change"
        }
        print(f"  {display_name:12s}: {tcode_desc.get(tcode, 'Unknown')}")

# Create corrected heatmap with Korean labels
plt.figure(figsize=(10, 8))

# Create DataFrame with Korean labels for plotting
normalized_stats_korean = normalized_stats.copy()
normalized_stats_korean.index = [korean_labels.get(idx, idx) for idx in normalized_stats.index]

sns.heatmap(normalized_stats_korean, annot=True, fmt='.2f', cmap='YlGnBu',
            vmin=0, vmax=1, square=True, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': '정규화 값'})

plt.title('국면별 주요 경제지표(최소-최대 정규화, FRED 권장 변환 적용)',
          fontsize=14, fontweight='bold')
plt.xlabel('Regime', fontsize=12)
plt.ylabel('FRED 지표', fontsize=12)
plt.tight_layout()
plt.savefig('corrected_regime_heatmap_transformed.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Heatmap with TRANSFORMED data saved as 'corrected_regime_heatmap_transformed.png'")
print("="*60)