import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set font for Korean text
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Read the data
df = pd.read_csv('section3_1_regime_labels.csv')
df['sasdate'] = pd.to_datetime(df['sasdate'])
df['year'] = df['sasdate'].dt.year
df['month'] = df['sasdate'].dt.month

# Define regime colors and labels
regime_colors = {
    0: '#FF0000',  # Red for outliers (Regime 0)
    1: '#2E7D32',  # Green
    2: '#1976D2',  # Blue
    3: '#F57C00',  # Orange
    4: '#7B1FA2',  # Purple
    5: '#795548'   # Brown
}

regime_labels = {
    0: 'Regime 0 (Outliers/Crisis)',
    1: 'Regime 1',
    2: 'Regime 2',
    3: 'Regime 3',
    4: 'Regime 4',
    5: 'Regime 5'
}

# Create figure with single plot
fig = plt.figure(figsize=(16, 6))

# Timeline view - Regime over time with NBER recessions
ax1 = plt.subplot(1, 1, 1)

for regime in sorted(df['final_regime_label'].unique()):
    regime_data = df[df['final_regime_label'] == regime]
    ax1.scatter(regime_data['sasdate'], [regime]*len(regime_data),
               color=regime_colors[regime], label=regime_labels[regime], s=20, alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Regime', fontsize=12)
ax1.set_title('구현된 경기 국면 분류', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yticks(range(6))

plt.tight_layout()
plt.savefig('regime_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("REGIME ANALYSIS SUMMARY (1992-2024)")
print("="*60)

regime_counts = df['final_regime_label'].value_counts().sort_index()

print("\n1. Regime Frequencies:")
for regime in sorted(regime_counts.index):
    count = regime_counts[regime]
    pct = (count / len(df)) * 100
    print(f"   {regime_labels[regime]}: {count} months ({pct:.1f}%)")

print("\n2. Crisis Periods (Regime 0):")
crisis_months = df[df['final_regime_label'] == 0][['sasdate', 'final_regime_label']]
crisis_periods_list = []
if len(crisis_months) > 0:
    start = crisis_months.iloc[0]['sasdate']
    for i in range(1, len(crisis_months)):
        if (crisis_months.iloc[i]['sasdate'] - crisis_months.iloc[i-1]['sasdate']).days > 35:
            crisis_periods_list.append((start, crisis_months.iloc[i-1]['sasdate']))
            start = crisis_months.iloc[i]['sasdate']
    crisis_periods_list.append((start, crisis_months.iloc[-1]['sasdate']))

    for start, end in crisis_periods_list:
        duration = (end - start).days / 30
        print(f"   {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')} ({duration:.0f} months)")

print("\n" + "="*60)