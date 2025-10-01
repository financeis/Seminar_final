import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for Korean text
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Read transition matrix
E_df = pd.read_csv('section3_3_transition_matrix.csv', index_col=0)
E = E_df.values

# Read transition counts for additional info
counts_df = pd.read_csv('section3_3_transition_counts.csv', index_col=0)
counts = counts_df.values

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Transition Probability Matrix Heatmap (Transposed)
ax1 = axes[0]
# Transpose the matrix to swap axes
E_transposed = E.T
im1 = ax1.imshow(E_transposed, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

# Add text annotations (using transposed matrix)
for i in range(E_transposed.shape[0]):
    for j in range(E_transposed.shape[1]):
        text = ax1.text(j, i, f'{E_transposed[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

ax1.set_xticks(range(6))
ax1.set_yticks(range(6))
ax1.set_xticklabels([f'R{i}' for i in range(6)])
ax1.set_yticklabels([f'R{i}' for i in range(6)])
ax1.set_xlabel('From Regime (현재 국면)', fontsize=12)
ax1.set_ylabel('To Regime (다음 국면)', fontsize=12)
ax1.set_title('전이 확률 행렬\n(Transition Probability Matrix)', fontsize=14, fontweight='bold')

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('전이 확률', rotation=270, labelpad=20)

# Plot 2: Network-style visualization with self-loops emphasized
ax2 = axes[1]

# Create a more interpretable visualization
# Diagonal (self-transitions) as bars
regimes = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']
self_trans = np.diag(E)
x_pos = np.arange(len(regimes))

bars = ax2.bar(x_pos, self_trans, color='steelblue', alpha=0.7)
ax2.set_xlabel('Regime', fontsize=12)
ax2.set_ylabel('자기 전이 확률 (Self-transition probability)', fontsize=12)
ax2.set_title('국면 지속성\n(Regime Persistence)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(regimes)
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, self_trans)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('regime_transition_matrix_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("REGIME TRANSITION ANALYSIS")
print("="*60)

print("\n1. TRANSITION PROBABILITY MATRIX:")
print("-"*40)
print(E_df.round(3))

print("\n2. REGIME PERSISTENCE (Diagonal values):")
print("-"*40)
for i, regime in enumerate(regimes):
    print(f"{regime}: {self_trans[i]:.3f} ({self_trans[i]*100:.1f}% stay in same regime)")

print("\n3. MOST LIKELY TRANSITIONS (excluding self-transitions):")
print("-"*40)
for i, from_regime in enumerate(regimes):
    # Set diagonal to -1 to exclude
    row = E[i].copy()
    row[i] = -1
    max_idx = np.argmax(row)
    if row[max_idx] > 0:
        print(f"{from_regime} → R{max_idx}: {row[max_idx]:.3f}")

print("\n4. TRANSITION COUNTS (actual occurrences):")
print("-"*40)
print(counts_df)

# Additional analysis: Expected duration in each regime
print("\n5. EXPECTED DURATION IN EACH REGIME:")
print("-"*40)
for i, regime in enumerate(regimes):
    if self_trans[i] < 1.0:
        expected_duration = 1 / (1 - self_trans[i])
        print(f"{regime}: {expected_duration:.1f} months")
    else:
        print(f"{regime}: Absorbing state (infinite duration)")

print("\n" + "="*60)