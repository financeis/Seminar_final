import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

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

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14))

# 1. Timeline view - Regime over time with NBER recessions
ax1 = plt.subplot(4, 1, 1)

# Add NBER recession shading (peak to trough)
nber_recessions = [
    ('1990-07-01', '1991-03-01'),  # July 1990 to March 1991
    ('2001-03-01', '2001-11-01'),  # March 2001 to November 2001
    ('2007-12-01', '2009-06-01'),  # December 2007 to June 2009
    ('2020-02-01', '2020-04-01'),  # February 2020 to April 2020
]

for peak, trough in nber_recessions:
    peak_date = pd.to_datetime(peak)
    trough_date = pd.to_datetime(trough)
    # Only show recessions within our data range
    if trough_date >= df['sasdate'].min() and peak_date <= df['sasdate'].max():
        ax1.axvspan(peak_date, trough_date, alpha=0.3, color='grey', label='NBER Recession' if peak == '1990-07-01' else '')

for regime in sorted(df['final_regime_label'].unique()):
    regime_data = df[df['final_regime_label'] == regime]
    ax1.scatter(regime_data['sasdate'], [regime]*len(regime_data),
               color=regime_colors[regime], label=regime_labels[regime], s=20, alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Regime', fontsize=12)
ax1.set_title('Economic Regimes Over Time with NBER Recessions (1992-2024)', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yticks(range(6))

# 2. Heatmap - Year vs Month with regime colors
ax2 = plt.subplot(4, 2, 3)
pivot_data = df.pivot_table(index='year', columns='month', values='final_regime_label', aggfunc='first')
im = ax2.imshow(pivot_data, cmap='tab10', aspect='auto', interpolation='nearest')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Year', fontsize=12)
ax2.set_title('Regime Heatmap by Year and Month', fontsize=14, fontweight='bold')
ax2.set_xticks(range(12))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax2.set_yticks(range(len(pivot_data.index)))
ax2.set_yticklabels(pivot_data.index, fontsize=8)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Regime', rotation=270, labelpad=20)

# 3. Yearly regime distribution - Stacked bar chart
ax3 = plt.subplot(4, 2, 4)
yearly_regime = df.groupby(['year', 'final_regime_label']).size().unstack(fill_value=0)
yearly_regime_pct = yearly_regime.div(yearly_regime.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(yearly_regime_pct))
for regime in sorted(df['final_regime_label'].unique()):
    if regime in yearly_regime_pct.columns:
        ax3.bar(yearly_regime_pct.index, yearly_regime_pct[regime], bottom=bottom,
               color=regime_colors[regime], label=regime_labels[regime], width=0.8)
        bottom += yearly_regime_pct[regime]
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Percentage (%)', fontsize=12)
ax3.set_title('Yearly Regime Distribution (Percentage)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# 4. Regime duration analysis
ax4 = plt.subplot(4, 2, 5)
regime_changes = []
current_regime = df.iloc[0]['final_regime_label']
start_date = df.iloc[0]['sasdate']
for i in range(1, len(df)):
    if df.iloc[i]['final_regime_label'] != current_regime:
        end_date = df.iloc[i-1]['sasdate']
        duration = (end_date - start_date).days / 30  # Convert to months
        regime_changes.append({
            'regime': current_regime,
            'start': start_date,
            'end': end_date,
            'duration_months': duration
        })
        current_regime = df.iloc[i]['final_regime_label']
        start_date = df.iloc[i]['sasdate']
# Add the last regime
end_date = df.iloc[-1]['sasdate']
duration = (end_date - start_date).days / 30
regime_changes.append({
    'regime': current_regime,
    'start': start_date,
    'end': end_date,
    'duration_months': duration
})
regime_duration_df = pd.DataFrame(regime_changes)
for regime in sorted(regime_duration_df['regime'].unique()):
    regime_durations = regime_duration_df[regime_duration_df['regime'] == regime]['duration_months']
    ax4.hist(regime_durations, alpha=0.6, label=regime_labels[regime],
            color=regime_colors[regime], bins=20, edgecolor='black')
ax4.set_xlabel('Duration (months)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Distribution of Regime Durations', fontsize=14, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Overall regime composition
ax5 = plt.subplot(4, 2, 6)
regime_counts = df['final_regime_label'].value_counts().sort_index()
colors = [regime_colors[i] for i in regime_counts.index]
wedges, texts, autotexts = ax5.pie(regime_counts, labels=[regime_labels[i] for i in regime_counts.index],
                                    colors=colors, autopct='%1.1f%%', startangle=90)
ax5.set_title('Overall Regime Composition (1992-2024)', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 6. Crisis periods highlighting with NBER recessions
ax6 = plt.subplot(4, 1, 4)

# Add NBER recession shading first
for peak, trough in nber_recessions:
    peak_date = pd.to_datetime(peak)
    trough_date = pd.to_datetime(trough)
    # Only show recessions within our data range
    if trough_date >= df['sasdate'].min() and peak_date <= df['sasdate'].max():
        ax6.axvspan(peak_date, trough_date, alpha=0.2, color='grey')
        # Add NBER label at the top
        ax6.text(peak_date + (trough_date - peak_date)/2, 0.95, 'NBER',
                transform=ax6.get_xaxis_transform(), ha='center', fontsize=8, color='grey')

# Plot all regimes with emphasis on Regime 0 (crisis)
for i, row in df.iterrows():
    color = '#FF0000' if row['final_regime_label'] == 0 else '#E0E0E0'
    alpha = 1.0 if row['final_regime_label'] == 0 else 0.3
    ax6.axvline(row['sasdate'], color=color, alpha=alpha, linewidth=1.5 if row['final_regime_label'] == 0 else 0.5)

# Highlight major crisis periods from our model
model_crisis_periods = [
    ('2001-09', '2001-09', 'Model: 9/11'),
    ('2008-02', '2012-08', 'Model: GFC'),
    ('2014-01', '2014-01', 'Model: Jan 2014'),
    ('2020-03', '2020-03', 'Model: COVID')
]
for start, end, label in model_crisis_periods:
    start_date = pd.to_datetime(start + '-01')
    end_date = pd.to_datetime(end + '-01')
    ax6.text(start_date + (end_date - start_date)/2, 0.85, label,
            transform=ax6.get_xaxis_transform(), ha='center', fontsize=9, fontweight='bold', color='red')

ax6.set_xlabel('Date', fontsize=12)
ax6.set_title('Crisis Periods: Model Regime 0 (Red Lines) vs NBER Recessions (Grey Shading)', fontsize=14, fontweight='bold')
ax6.set_ylim(0, 1)
ax6.set_yticks([])
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('regime_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("REGIME ANALYSIS SUMMARY (1992-2024)")
print("="*60)

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

print("\n3. Average Regime Duration:")
regime_avg_duration = regime_duration_df.groupby('regime')['duration_months'].agg(['mean', 'std', 'max', 'min'])
for regime in regime_avg_duration.index:
    stats = regime_avg_duration.loc[regime]
    print(f"   {regime_labels[regime]}:")
    print(f"      Mean: {stats['mean']:.1f} months")
    print(f"      Std:  {stats['std']:.1f} months")
    print(f"      Max:  {stats['max']:.1f} months")
    print(f"      Min:  {stats['min']:.1f} months")

print("\n4. Regime Transitions:")
transitions = {}
for i in range(len(df)-1):
    from_regime = df.iloc[i]['final_regime_label']
    to_regime = df.iloc[i+1]['final_regime_label']
    if from_regime != to_regime:
        key = f"R{from_regime} -> R{to_regime}"
        transitions[key] = transitions.get(key, 0) + 1

print("   Most common transitions:")
sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
for transition, count in sorted_transitions:
    print(f"      {transition}: {count} times")

print("\n" + "="*60)