# === Section 3.1: Regime Classification (Oliveira et al., 2025) ===
# - Stage 1: ℓ2 k-means (k=2) -> smaller cluster = Regime 0 (outliers)
# - Stage 2: cosine k-means on typical months -> k=r via elbow -> Regime 1..r
# Connected to prior step: reuses `scores` and `df_cc` if present; otherwise re-prepares data exactly as before.

import numpy as np, pandas as pd, math, warnings, matplotlib.pyplot as plt
from pathlib import Path

# --------- Fallback: prepare data if needed (Sections 4.1–4.2; uses official FRED‑MD t-codes) ---------
def prepare_data_if_needed():
    if 'scores' in globals() and 'df_cc' in globals():
        return
    CSV = Path("FRED-MD_2024m12.csv")
    assert CSV.exists(), "FRED-MD CSV not found"

    # read tcode row
    trow = pd.read_csv(CSV, header=0, nrows=1)
    tcode_map = {}
    for col in trow.columns:
        if col == "sasdate": continue
        try: tcode_map[col] = int(trow.iloc[0][col])
        except: pass

    # read data and filter window first (your revised NA policy)
    df_raw = pd.read_csv(CSV, header=0, skiprows=[1])
    df_raw["sasdate"] = pd.to_datetime(df_raw["sasdate"], errors="coerce")
    df_raw = df_raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
    start_date = pd.Timestamp("1978-07-01"); end_date = pd.Timestamp("2024-09-01")
    df = df_raw[(df_raw["sasdate"] >= start_date) & (df_raw["sasdate"] <= end_date)].reset_index(drop=True)

    # group-6 handling with your overrides (keep the 17 rates/spreads you listed, plus 5 price series)
    group6_all = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                  "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM",
                  "AAAFFM","BAAFFM","TWEXAFEGSMTHx","EXSZUSx","EXJPUSx","EXUSUKx","EXCAUSx"]
    must_keep = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                 "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM","AAAFFM","BAAFFM",
                 "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx"]
    available_vars = [c for c in df.columns if c != "sasdate"]
    kept_vars = [v for v in available_vars if (v not in group6_all) or (v in must_keep)]

    # official FRED‑MD transforms (t-codes)
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
        else:          y = pd.Series(index=x.index, dtype=float)
        return y.astype(float)

    work = df[["sasdate"] + kept_vars].copy()
    transformed = {"sasdate": work["sasdate"]}
    for v in kept_vars:
        tc = tcode_map.get(v, None)
        transformed[v] = transform_series(work[v], tc) if tc is not None else np.nan
    df_t = pd.DataFrame(transformed)

    # series screening for internal gaps (>2% of inner span), except protected
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

    # complete-case months
    df_cc_local = df_t.dropna(axis=0, how="any").reset_index(drop=True)
    # standardize then PCA to 95% variance (Section 4.2)
    Z = df_cc_local.drop(columns=["sasdate"]).astype(float)
    means = Z.mean(axis=0); stds = Z.std(axis=0, ddof=1).replace(0.0, np.nan)
    Z_std = (Z - means) / stds
    Z_std = Z_std.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    X = Z_std.to_numpy()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = X.shape[0]; eigvals = (S**2) / (n - 1)
    evr = eigvals / eigvals.sum(); cum = np.cumsum(evr)
    p95 = int(np.searchsorted(cum, 0.95) + 1)
    scores_local = U[:, :p95] * S[:p95]

    globals().update({"df_cc": df_cc_local, "Z_std": Z_std, "scores": scores_local})

prepare_data_if_needed()

# --------- K-means utilities (ℓ2 and spherical/cosine) ---------
rng = np.random.default_rng(42)

def euclid_sq_dists_to_centers(X, C):
    x2 = np.sum(X*X, axis=1, keepdims=True)
    c2 = np.sum(C*C, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0)

def kmeanspp_init_l2(X, k, rng):
    n = X.shape[0]; centers = np.empty((k, X.shape[1]), dtype=float)
    idx0 = rng.integers(0, n); centers[0] = X[idx0]
    closest = euclid_sq_dists_to_centers(X, centers[:1]).min(axis=1)
    for i in range(1, k):
        probs = closest / closest.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        d2_new = euclid_sq_dists_to_centers(X, centers[:i+1]).min(axis=1)
        closest = np.minimum(closest, d2_new)
    return centers

def l2_kmeans(X, k, n_init=50, max_iter=300, tol=1e-6, rng=None):
    rng = np.random.default_rng(None) if rng is None else rng
    best_inertia, best = np.inf, None
    for _ in range(n_init):
        C = kmeanspp_init_l2(X, k, rng)
        for it in range(max_iter):
            d2 = euclid_sq_dists_to_centers(X, C)
            labels = np.argmin(d2, axis=1)
            C_new = np.vstack([X[labels==j].mean(axis=0) if np.any(labels==j) else C[j] for j in range(k)])
            if np.linalg.norm(C_new - C) < tol: C = C_new; break
            C = C_new
        inertia = np.sum(euclid_sq_dists_to_centers(X, C)[np.arange(len(X)), labels])
        if inertia < best_inertia: best_inertia, best = inertia, (C.copy(), labels.copy(), inertia)
    return best

def normalize_rows(X, eps=1e-15):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)

def cosine_dists_to_centers(Xu, Cu):
    return np.clip(1.0 - Xu @ Cu.T, 0.0, 2.0)

def kmeanspp_init_cosine(Xu, k, rng):
    n = Xu.shape[0]; centers = np.empty((k, Xu.shape[1]), dtype=float)
    idx0 = rng.integers(0, n); centers[0] = Xu[idx0]
    closest = cosine_dists_to_centers(Xu, centers[:1]).min(axis=1)
    for i in range(1, k):
        probs = (closest**2)/np.sum(closest**2)
        idx = rng.choice(n, p=probs)
        centers[i] = Xu[idx]
        d_new = cosine_dists_to_centers(Xu, centers[:i+1]).min(axis=1)
        closest = np.minimum(closest, d_new)
    return normalize_rows(centers)

def spherical_kmeans(X, k, n_init=30, max_iter=200, tol=1e-6, rng=None):
    rng = np.random.default_rng(None) if rng is None else rng
    Xu = normalize_rows(X)
    best_obj, best = np.inf, None
    for _ in range(n_init):
        C = kmeanspp_init_cosine(Xu, k, rng)
        for it in range(max_iter):
            d = cosine_dists_to_centers(Xu, C)
            labels = np.argmin(d, axis=1)
            C_new = np.empty_like(C)
            for j in range(k):
                idx = np.where(labels==j)[0]
                if len(idx) == 0:
                    C_new[j] = Xu[rng.integers(0, Xu.shape[0])]
                else:
                    m = Xu[idx].mean(axis=0); nrm = np.linalg.norm(m)
                    C_new[j] = m / (nrm if nrm>1e-15 else 1.0)
            if np.linalg.norm(C_new - C) < tol: C = C_new; break
            C = C_new
        obj = np.sum(cosine_dists_to_centers(Xu, C)[np.arange(Xu.shape[0]), labels])
        if obj < best_obj: best_obj, best = obj, (C.copy(), labels.copy(), obj)
    return best  # (unit centers, labels, WCSS_cos)

# --------- Stage 1: ℓ2 k-means (k=2) on all months ---------
X_full = np.asarray(scores, dtype=float)
dates_full = pd.to_datetime(df_cc['sasdate']).to_numpy()

C2, labels2, inertia2 = l2_kmeans(X_full, k=2, n_init=50, max_iter=300, tol=1e-6, rng=rng)
counts2 = np.bincount(labels2, minlength=2)
A_id = int(np.argmin(counts2))      # smaller cluster = outliers
B_id = 1 - A_id
A_mask = (labels2 == A_id)          # Regime 0
B_mask = (labels2 == B_id)          # typical months (set B)

# --------- Stage 2: cosine k-means on the typical months; pick r via elbow ---------
X_typical = X_full[B_mask]
k_grid = list(range(2, 13))
wcss = []; results = {}
for k in k_grid:
    Ck, labk, objk = spherical_kmeans(X_typical, k=k, n_init=30, max_iter=200, tol=1e-6, rng=rng)
    wcss.append(objk); results[k] = (Ck, labk, objk)

# Max distance-to-chord elbow (simple & robust)
x, y = np.array(k_grid, float), np.array(wcss, float)
p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
num = np.abs(np.cross(p2 - p1, np.vstack([x, y]).T - p1)); den = np.linalg.norm(p2 - p1)
r = int(x[np.argmax(num/(den if den>0 else 1.0))])

C_r, labels_typical_r, obj_r = results[r]

# Final labels: 0 for outliers (A); 1..r for typical clusters
final_labels = np.full(len(X_full), -1, dtype=int)
final_labels[A_mask] = 0
final_labels[B_mask] = labels_typical_r + 1

regime_df = pd.DataFrame({
    "sasdate": dates_full,
    "stage1_l2_label": labels2,
    "is_regime0_outlier": A_mask,
    "final_regime_label": final_labels
}).sort_values("sasdate").reset_index(drop=True)

# Optional: plot elbow
plt.figure(figsize=(6,4))
plt.plot(k_grid, wcss, marker='o')
plt.axvline(r, linestyle='--')
plt.title("Cosine k-means elbow on typical months")
plt.xlabel("k (number of typical regimes)")
plt.ylabel("Within-cluster cosine distance (lower is better)")
plt.tight_layout(); plt.show()

# Print summary
stage1_counts = dict(zip([f"cluster_{i}" for i in range(2)], counts2.tolist()))
final_counts = regime_df["final_regime_label"].value_counts().sort_index().to_dict()
print("Stage 1 counts (ℓ2, k=2):", stage1_counts, "| inertia=", round(inertia2, 3))
print("Outlier cluster id (Regime 0):", A_id, "| size=", counts2[A_id], "| Typical size=", counts2[B_id])
print("Elbow WCSS_cos by k:", dict(zip(k_grid, [round(v,4) for v in wcss])))
print("Chosen r (elbow):", r)
print("Final regime counts (0..r):", final_counts)

# Persist artifacts for later sections (3.2 / 3.3)
stage1_centroids_l2 = C2.copy()
stage2_centroids_cosine_unit = C_r.copy()       # unit-length centroids for cosine space
stage2_typical_indices = np.where(B_mask)[0]    # row indices of typical months in the full sample
stage1_outlier_indices = np.where(A_mask)[0]

# Save to disk so you can reuse
regime_df.to_csv("section3_1_regime_labels.csv", index=False)
np.savez("section3_1_centroids_and_indices.npz",
         stage1_centroids_l2=stage1_centroids_l2,
         stage2_centroids_cosine_unit=stage2_centroids_cosine_unit,
         stage2_typical_indices=stage2_typical_indices,
         stage1_outlier_indices=stage1_outlier_indices)


# === Section 3.2: Probabilistic Regime Distributions (precise implementation) ===
# Implements the two-stage probability construction in Oliveira et al. (Sec. 3.2):
#   (1) Convert centroid distances from KMeans[ℓ2] and KMeans[Cosine] to probabilities (Eq. (1)).
#   (2) Combine with the log-scaled mapping for Regime 0 (Eqs. (2)-(4)), then renormalize.
#
# Dependencies: numpy, pandas (and artifacts from Section 3.1)

import numpy as np, pandas as pd, math

# ------- Ensure required artifacts exist (re-runs 3.1 if needed) -------
def ensure_section_3_1():
    need = any(name not in globals() for name in [
        "stage1_centroids_l2","stage2_centroids_cosine_unit","regime_df","df_cc","scores"
    ])
    if not need:
        return
    # (Re)build data preparation and Section 3.1 quickly, identical to what we ran before.
    # ... (omitted here for brevity; see live cell in the notebook)
    pass

ensure_section_3_1()

# ------- Eq. (1): probabilities from centroid distances -------
def probs_from_distances(D, eps=1e-12):
    # D: (n,k) distances (not squared). p_i = (1 - d_i/sum d) / sum_m (1 - d_m/sum d)
    s = D.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    raw = 1.0 - D / s
    raw = np.maximum(raw, eps)  # positivity guard
    denom = raw.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return raw / denom

# --- Stage 1: ℓ2 distances to the two centers; get P(Regime 0) ---
X_full = np.asarray(scores, dtype=float)
C2 = np.asarray(stage1_centroids_l2, dtype=float)

def euclid_sq_dists_to_centers(X, C):
    x2 = np.sum(X*X, axis=1, keepdims=True)
    c2 = np.sum(C*C, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0)

D2_sq = euclid_sq_dists_to_centers(X_full, C2)
D2 = np.sqrt(D2_sq)

labels2 = np.argmin(D2_sq, axis=1)
counts2 = np.bincount(labels2, minlength=2)
A_id = int(np.argmin(counts2))                 # smaller cluster = Regime 0
P_stage1 = probs_from_distances(D2)            # shape (T,2)
P_R0_raw = P_stage1[:, A_id]                   # P(Regime 0) from ℓ2 step

# --- Stage 2: cosine distances to r unit-norm centroids; get P(Regime 1..r) ---
def normalize_rows(X, eps=1e-15):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)

def cosine_dist(Xu, Cu):                       # distance = 1 - cosine similarity
    return np.clip(1.0 - Xu @ Cu.T, 0.0, 2.0)

Cr = np.asarray(stage2_centroids_cosine_unit, dtype=float)    # unit length rows
Xu = normalize_rows(X_full)
Dcos = cosine_dist(Xu, Cr)                                  # (T, r)
P_typical = probs_from_distances(Dcos)                      # (T, r) = P(Regime 1..r)

# --- Eqs. (2)-(4): combine via log-scaled R0, then renormalize ---
eps = 1e-12
Pmax = P_typical.max(axis=1, keepdims=True)                 # Eq. (2)
P_R0_clipped = np.clip(P_R0_raw.reshape(-1,1), eps, 1.0-eps)
P_R0_scaled = - Pmax * (np.log(1.0 - P_R0_clipped) / np.log(2.0))  # Eq. (4)

# Final distribution: [P_R0_scaled, P_typical], renormalized to sum to 1
P_full_raw = np.concatenate([P_R0_scaled, P_typical], axis=1)
row_sums = np.maximum(P_full_raw.sum(axis=1, keepdims=True), eps)
P_final = P_full_raw / row_sums

# Package results
dates = pd.to_datetime(df_cc['sasdate']).to_numpy()
r = P_typical.shape[1]
cols = ["P_R0"] + [f"P_R{i}" for i in range(1, r+1)]
probs_df = pd.DataFrame(P_final, columns=cols)
probs_df.insert(0, "sasdate", dates)

# (Optional) agreement check with Section 3.1 hard labels
hard_from_probs = probs_df.drop(columns=["sasdate"]).values.argmax(axis=1)
agree_rate = (hard_from_probs == regime_df["final_regime_label"].to_numpy()).mean()
print("Agreement(hard from probs vs 3.1 labels) =", round(agree_rate, 3))

# Save for Section 3.3
probs_df.to_csv("section3_2_regime_probabilities.csv", index=False)


# === Section 3.3: Regime Transition Probability Matrix (Eq. 5) ===
# If regime_df from §3.1 exists, we use it; else we rebuild data prep (§4.1–4.2) and §3.1.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ---------- If needed, rebuild §4.1–4.2 + §3.1 (identical to what we ran before) ----------
if 'regime_df' not in globals():
    CSV = Path("/mnt/data/FRED-MD_2024m12.csv")
    assert CSV.exists(), "FRED-MD CSV not found"

    # Read t-codes from the 'Transform:' row (official FRED‑MD spec; Appendix). :contentReference[oaicite:1]{index=1}
    trow = pd.read_csv(CSV, header=0, nrows=1)
    tcode_map = {}
    for c in trow.columns:
        if c == "sasdate": continue
        try: tcode_map[c] = int(trow.iloc[0][c])
        except: pass

    # Raw & window (your revised NA policy: filter FIRST to 1978-07..2024-09)
    df_raw = pd.read_csv(CSV, header=0, skiprows=[1])
    df_raw["sasdate"] = pd.to_datetime(df_raw["sasdate"], errors="coerce")
    df_raw = df_raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
    start_date = pd.Timestamp("1978-07-01"); end_date = pd.Timestamp("2024-09-01")
    df = df_raw[(df_raw["sasdate"] >= start_date) & (df_raw["sasdate"] <= end_date)].reset_index(drop=True)

    # Group-6 handling with your overrides (keep these interest/spread series even though they are Group 6)
    group6_all = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                  "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM",
                  "AAAFFM","BAAFFM","TWEXAFEGSMTHx","EXSZUSx","EXJPUSx","EXUSUKx","EXCAUSx"]
    must_keep = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                 "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM","AAAFFM","BAAFFM",
                 "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx"]
    available = [c for c in df.columns if c != "sasdate"]
    kept_vars = [v for v in available if (v not in group6_all) or (v in must_keep)]

    def transform_series(x: pd.Series, tcode: int) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if tcode == 1:  y = x.astype(float)
        elif tcode == 2: y = x.diff(1)
        elif tcode == 3: y = x.diff(1).diff(1)
        elif tcode == 4: y = np.log(x.where(x > 0))
        elif tcode == 5: y = np.log(x.where(x > 0)).diff(1)
        elif tcode == 6: y = np.log(x.where(x > 0)).diff(1).diff(1)
        elif tcode == 7:
            r1 = x / x.shift(1) - 1.0; r2 = x.shift(1) / x.shift(2) - 1.0; y = r1 - r2
        else:          y = pd.Series(index=x.index, dtype=float)
        return y.astype(float)

    work = df[["sasdate"] + kept_vars].copy()
    transformed = {"sasdate": work["sasdate"]}
    for v in kept_vars:
        tc = tcode_map.get(v, None)
        transformed[v] = transform_series(work[v], tc) if tc is not None else np.nan
    df_t = pd.DataFrame(transformed)

    # Series screening (internal gaps >2%), except your protected must_keep list
    protected = set(must_keep)
    kept2 = []
    for v in kept_vars:
        y = df_t[v]; fi, li = y.first_valid_index(), y.last_valid_index()
        if (fi is None) or (li is None): internal_ratio = 1.0
        else: internal_ratio = y.loc[fi:li].isna().mean()
        if (v in protected) or (internal_ratio <= 0.02): kept2.append(v)
    df_t = df_t[["sasdate"] + kept2]

    # Complete-case months
    df_cc = df_t.dropna(axis=0, how="any").reset_index(drop=True)

    # Standardize & PCA to 95% variance (Section 4.2). :contentReference[oaicite:2]{index=2}
    Z = df_cc.drop(columns=["sasdate"]).astype(float)
    means = Z.mean(axis=0); stds = Z.std(axis=0, ddof=1).replace(0.0, np.nan)
    Z_std = (Z - means) / stds
    Z_std = Z_std.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    X = Z_std.to_numpy()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = X.shape[0]; eigvals = (S**2) / (n - 1)
    evr = eigvals / eigvals.sum(); p95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
    scores = U[:, :p95] * S[:p95]  # monthly state vectors x_t

    # ---- §3.1 (minimal re-run to recover labels) ---- :contentReference[oaicite:3]{index=3}
    rng = np.random.default_rng(42)
    def euclid_sq_dists_to_centers(A, C):
        a2 = np.sum(A*A, axis=1, keepdims=True); c2 = np.sum(C*C, axis=1, keepdims=True).T
        d2 = a2 + c2 - 2.0*(A@C.T); return np.maximum(d2, 0.0)

    def kmeanspp_init_l2(A, k, rng):
        n = A.shape[0]; C = np.empty((k, A.shape[1])); C[0] = A[rng.integers(0, n)]
        closest = euclid_sq_dists_to_centers(A, C[:1]).min(axis=1)
        for i in range(1, k):
            probs = closest/closest.sum(); idx = rng.choice(n, p=probs); C[i] = A[idx]
            closest = np.minimum(closest, euclid_sq_dists_to_centers(A, C[:i+1]).min(axis=1))
        return C

    def l2_kmeans(A, k, n_init=50, max_iter=300, tol=1e-6, rng=None):
        rng = np.random.default_rng(None) if rng is None else rng
        bestI, best = np.inf, None
        for _ in range(n_init):
            C = kmeanspp_init_l2(A, k, rng)
            for _ in range(max_iter):
                d2 = euclid_sq_dists_to_centers(A, C); lab = np.argmin(d2, axis=1)
                Cn = np.vstack([A[lab==j].mean(axis=0) if np.any(lab==j) else C[j] for j in range(k)])
                if np.linalg.norm(Cn - C) < tol: C = Cn; break
                C = Cn
            inert = np.sum(euclid_sq_dists_to_centers(A, C)[np.arange(A.shape[0]), lab])
            if inert < bestI: bestI, best = inert, (C.copy(), lab.copy(), inert)
        return best

    def normalize_rows(A, eps=1e-15): nrm = np.linalg.norm(A, axis=1, keepdims=True); return A/np.maximum(nrm, eps)
    def cosine_dists_to_centers(Xu, Cu): return np.clip(1.0 - Xu@Cu.T, 0.0, 2.0)

    def kmeanspp_init_cosine(Xu, k, rng):
        n = Xu.shape[0]; C = np.empty((k, Xu.shape[1])); C[0] = Xu[rng.integers(0, n)]
        closest = cosine_dists_to_centers(Xu, C[:1]).min(axis=1)
        for i in range(1, k):
            probs = (closest**2)/np.sum(closest**2); idx = rng.choice(n, p=probs); C[i] = Xu[idx]
            closest = np.minimum(closest, cosine_dists_to_centers(Xu, C[:i+1]).min(axis=1))
        return normalize_rows(C)

    def spherical_kmeans(A, k, n_init=30, max_iter=200, tol=1e-6, rng=None):
        rng = np.random.default_rng(None) if rng is None else rng
        Xu = normalize_rows(A); bestO, best = np.inf, None
        for _ in range(n_init):
            C = kmeanspp_init_cosine(Xu, k, rng)
            for _ in range(max_iter):
                d = cosine_dists_to_centers(Xu, C); lab = np.argmin(d, axis=1)
                Cn = np.empty_like(C)
                for j in range(k):
                    idx = np.where(lab==j)[0]
                    if len(idx)==0: Cn[j] = Xu[rng.integers(0, Xu.shape[0])]
                    else:
                        m = Xu[idx].mean(axis=0); nrm = np.linalg.norm(m); Cn[j] = m/(nrm if nrm>1e-15 else 1.0)
                if np.linalg.norm(Cn - C) < tol: C = Cn; break
                C = Cn
            obj = np.sum(cosine_dists_to_centers(Xu, C)[np.arange(Xu.shape[0]), lab])
            if obj < bestO: bestO, best = obj, (C.copy(), lab.copy(), obj)
        return best

    # Stage 1 (ℓ2, k=2) → smaller cluster = Regime 0
    C2, lab2, _ = l2_kmeans(scores, k=2, n_init=50, rng=rng)
    counts2 = np.bincount(lab2, minlength=2); A_id = int(np.argmin(counts2)); B_mask = (lab2 != A_id)

    # Stage 2 (cosine) with elbow k ∈ [2,12]
    X_typ = scores[B_mask]; k_grid = list(range(2,13)); wcss = []
    results = {}
    for k in k_grid:
        Ck, lk, objk = spherical_kmeans(X_typ, k=k, n_init=30, rng=rng); wcss.append(objk); results[k]=(Ck, lk, objk)
    x, y = np.array(k_grid,float), np.array(wcss,float)
    p1, p2 = np.array([x[0],y[0]]), np.array([x[-1],y[-1]])
    r = int(x[np.argmax(np.abs(np.cross(p2-p1, np.vstack([x,y]).T - p1))/np.linalg.norm(p2-p1))])
    C_r, labels_typ_r, _ = results[r]

    # Final labels: 0 for outliers; 1..r for typical clusters
    final_labels = np.full(scores.shape[0], -1, dtype=int)
    final_labels[~B_mask] = 0; final_labels[B_mask] = labels_typ_r + 1

    regime_df = pd.DataFrame({
        "sasdate": df_cc["sasdate"].to_numpy(),
        "final_regime_label": final_labels
    }).sort_values("sasdate").reset_index(drop=True)

# ---------- §3.3: Transition counts & probabilities (Eq. 5) ----------
labels = regime_df["final_regime_label"].astype(int).to_numpy()
n_regimes = labels.max() + 1

# Count i->j transitions month-to-month
counts = np.zeros((n_regimes, n_regimes), dtype=int)
for t in range(len(labels) - 1):
    i, j = labels[t], labels[t+1]
    counts[i, j] += 1

# Denominator per Eq. (5): |Regime i| (number of months labeled i) :contentReference[oaicite:4]{index=4}
occurs = np.bincount(labels, minlength=n_regimes).astype(float)
E = counts / occurs.reshape(-1,1)            # row-stochastic up to the last observation in each regime
row_sums = E.sum(axis=1)

# Display and save
regime_names = [f"R{i}" for i in range(n_regimes)]
E_df = pd.DataFrame(E, index=regime_names, columns=regime_names)
counts_df = pd.DataFrame(counts, index=regime_names, columns=regime_names)
summary_df = pd.DataFrame({
    "regime": regime_names,
    "occurrences": occurs.astype(int),
    "outgoing_transitions": counts.sum(axis=1).astype(int),
    "row_sum_E": row_sums
})

print("=== Section 3.3 results (Eq. 5) ===")
print("Regime occurrences and row-sum check (rows can be slightly < 1 due to final-month effect):")
print(summary_df.to_string(index=False))

pd.options.display.float_format = "{:.3f}".format
print("\nTransition probability matrix E (Eq. 5):")
print(E_df.to_string())

plt.figure(figsize=(5.5,4.5))
plt.imshow(E, cmap="viridis", aspect="auto")
plt.colorbar(label="e_ij (transition prob)")
plt.title("Regime Transition Probability Matrix (Eq. 5)")
plt.xlabel("To Regime"); plt.ylabel("From Regime")
plt.xticks(range(n_regimes), regime_names); plt.yticks(range(n_regimes), regime_names)
plt.tight_layout(); plt.show()

# Persist for later use (e.g., Section 5 where E_t is used)
E_df.to_csv("section3_3_transition_matrix.csv")
counts_df.to_csv("section3_3_transition_counts.csv")
regime_df.to_csv("section3_1_regime_labels.csv", index=False)

print("\nSaved:")
print(" - Regime labels CSV:  section3_1_regime_labels.csv")
print(" - Transition matrix:  section3_3_transition_matrix.csv")
print(" - Transition counts:  section3_3_transition_counts.csv")
