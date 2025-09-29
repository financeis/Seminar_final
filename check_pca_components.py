import numpy as np
import pandas as pd
from pathlib import Path

# Replicate section3.py PCA process
CSV = Path("FRED-MD_2024m12.csv")
assert CSV.exists(), "FRED-MD CSV not found"

# Read tcode row
trow = pd.read_csv(CSV, header=0, nrows=1)
tcode_map = {}
for col in trow.columns:
    if col == "sasdate": continue
    try: tcode_map[col] = int(trow.iloc[0][col])
    except: pass

# Read data and filter window
df_raw = pd.read_csv(CSV, header=0, skiprows=[1])
df_raw["sasdate"] = pd.to_datetime(df_raw["sasdate"], errors="coerce")
df_raw = df_raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
start_date = pd.Timestamp("1978-07-01")
end_date = pd.Timestamp("2024-09-01")
df = df_raw[(df_raw["sasdate"] >= start_date) & (df_raw["sasdate"] <= end_date)].reset_index(drop=True)

# Group-6 handling (same as section3.py)
group6_all = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
              "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM",
              "AAAFFM","BAAFFM","TWEXAFEGSMTHx","EXSZUSx","EXJPUSx","EXUSUKx","EXCAUSx"]
must_keep = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
             "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM","AAAFFM","BAAFFM",
             "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx"]
available_vars = [c for c in df.columns if c != "sasdate"]
kept_vars = [v for v in available_vars if (v not in group6_all) or (v in must_keep)]

# Transform series
def transform_series(x: pd.Series, tcode: int) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if tcode == 1:  y = x.astype(float)
    elif tcode == 2: y = x.diff(1)
    elif tcode == 3: y = x.diff(1).diff(1)
    elif tcode == 4: y = np.log(x.where(x > 0))
    elif tcode == 5: y = np.log(x.where(x > 0)).diff(1)
    elif tcode == 6: y = np.log(x.where(x > 0)).diff(1).diff(1)
    elif tcode == 7:
        r1 = x / x.shift(1) - 1.0
        r2 = x.shift(1) / x.shift(2) - 1.0
        y = r1 - r2
    else: y = pd.Series(index=x.index, dtype=float)
    return y.astype(float)

work = df[["sasdate"] + kept_vars].copy()
transformed = {"sasdate": work["sasdate"]}
for v in kept_vars:
    tc = tcode_map.get(v, None)
    transformed[v] = transform_series(work[v], tc) if tc is not None else np.nan
df_t = pd.DataFrame(transformed)

# Series screening for internal gaps
protected = set(must_keep)
kept_vars2 = []
for v in kept_vars:
    y = df_t[v]
    fi, li = y.first_valid_index(), y.last_valid_index()
    if (fi is None) or (li is None):
        internal_ratio = 1.0
    else:
        inner = y.loc[fi:li]
        internal_ratio = inner.isna().mean()
    if (v in protected) or (internal_ratio <= 0.02):
        kept_vars2.append(v)
df_t = df_t[["sasdate"] + kept_vars2]

# Complete-case months
df_cc = df_t.dropna(axis=0, how="any").reset_index(drop=True)

# Standardize
Z = df_cc.drop(columns=["sasdate"]).astype(float)
means = Z.mean(axis=0)
stds = Z.std(axis=0, ddof=1).replace(0.0, np.nan)
Z_std = (Z - means) / stds
Z_std = Z_std.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

# PCA
X = Z_std.to_numpy()
U, S, Vt = np.linalg.svd(X, full_matrices=False)
n = X.shape[0]
eigvals = (S**2) / (n - 1)
evr = eigvals / eigvals.sum()
cum = np.cumsum(evr)

# Find 95% variance threshold
p95 = int(np.searchsorted(cum, 0.95) + 1)

print("="*60)
print("PCA ANALYSIS RESULTS")
print("="*60)
print(f"\nOriginal variables: {len(kept_vars)} → {len(kept_vars2)} (after screening)")
print(f"Final variables after standardization: {X.shape[1]}")
print(f"Number of observations: {X.shape[0]}")
print(f"\nPCA Components needed for 95% variance: {p95}")
print(f"Dimension reduction: {X.shape[1]} → {p95}")
print(f"Compression ratio: {p95/X.shape[1]*100:.1f}%")

print("\n" + "-"*40)
print("Variance Explained by Component:")
print("-"*40)
for i in range(min(10, len(evr))):
    print(f"Component {i+1}: {evr[i]*100:5.2f}% (Cumulative: {cum[i]*100:5.2f}%)")

print(f"\n...")
print(f"Component {p95}: {evr[p95-1]*100:5.2f}% (Cumulative: {cum[p95-1]*100:5.2f}%)")

# Show the elbow
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
ax1.plot(range(1, min(31, len(evr)+1)), evr[:30]*100, 'o-')
ax1.axvline(p95, color='red', linestyle='--', label=f'95% threshold (n={p95})')
ax1.set_xlabel('Component Number')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('PCA Scree Plot')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Cumulative variance
ax2.plot(range(1, min(31, len(cum)+1)), cum[:30]*100, 'o-')
ax2.axhline(95, color='red', linestyle='--', label='95% threshold')
ax2.axvline(p95, color='red', linestyle='--')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Cumulative Variance Explained')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('pca_components_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'pca_components_analysis.png'")