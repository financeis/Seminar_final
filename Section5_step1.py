# Step 1: 48-month rolling PCA + Regime model (two-stage k-means)
# Follows the paper’s Sections 3.1–3.3 (k-means ℓ2→cosine, fuzzy probabilities, transition matrix)
# and Section 5.1 (Markov one-step update).  :contentReference[oaicite:8]{index=8}
# FRED-MD transforms (t-codes) per McCracken & Ng appendix.  :contentReference[oaicite:9]{index=9}

import numpy as np, pandas as pd, math
from numpy.linalg import norm
from pathlib import Path

CSV = Path("FRED-MD_2024m12.csv")
assert CSV.exists(), "FRED-MD CSV not found"

# ---------- Read t-codes ----------
trow = pd.read_csv(CSV, header=0, nrows=1)
tcode_map = {c:int(trow.iloc[0][c]) for c in trow.columns if c!="sasdate" and pd.notna(trow.iloc[0][c])}

# ---------- Load data; filter your window FIRST (1978-07 .. 2024-09) ----------
df_raw = pd.read_csv(CSV, header=0, skiprows=[1])
df_raw["sasdate"] = pd.to_datetime(df_raw["sasdate"], errors="coerce")
df_raw = df_raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)

start_date = pd.Timestamp("1978-07-01")
end_date   = pd.Timestamp("2024-09-01")
df = df_raw[(df_raw["sasdate"] >= start_date) & (df_raw["sasdate"] <= end_date)].reset_index(drop=True)

# ---------- Variable set: exclude Group 6 except your must-keep ----------
group6_all = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
              "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM",
              "AAAFFM","BAAFFM","TWEXAFEGSMTHx","EXSZUSx","EXJPUSx","EXUSUKx","EXCAUSx"]
must_keep = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
             "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM","AAAFFM","BAAFFM",
             "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx"]

avail = [c for c in df.columns if c!="sasdate"]
kept_vars = [v for v in avail if (v not in group6_all) or (v in must_keep)]

def transform_series(x, tcode: int):
    x = pd.to_numeric(x, errors="coerce")
    if tcode == 1:  y = x
    elif tcode == 2: y = x.diff(1)
    elif tcode == 3: y = x.diff(1).diff(1)
    elif tcode == 4: y = np.log(x.where(x>0))
    elif tcode == 5: y = np.log(x.where(x>0)).diff(1)
    elif tcode == 6: y = np.log(x.where(x>0)).diff(1).diff(1)
    elif tcode == 7:
        r1 = x/x.shift(1) - 1.0
        r2 = x.shift(1)/x.shift(2) - 1.0
        y = r1 - r2
    else: y = pd.Series(index=x.index, dtype=float)
    return y.astype(float)

# Apply transforms
work = df[["sasdate"]+kept_vars].copy()
df_t = pd.DataFrame({"sasdate": work["sasdate"]})
for v in kept_vars:
    tc = tcode_map.get(v, None)
    df_t[v] = transform_series(work[v], tc) if tc is not None else np.nan

# Drop series with internal gaps >2% (except must-keep), then complete-case months
protected = set(must_keep)
drop_vars = []
for v in kept_vars:
    y = df_t[v]; a = y.first_valid_index(); b = y.last_valid_index()
    gap = 1.0 if (a is None or b is None) else y.loc[a:b].isna().mean()
    if (v not in protected) and (gap > 0.02): drop_vars.append(v)
kept_vars2 = [v for v in kept_vars if v not in drop_vars]
df_t = df_t[["sasdate"]+kept_vars2]
df_cc = df_t.dropna(axis=0, how="any").reset_index(drop=True)

# ---------- Helpers for rolling PCA + k-means ----------
def zscore_window(X: np.ndarray):
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
    keep = sd > 0
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    return Z

def pca_svd_95(Z: np.ndarray, thr=0.95):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n = Z.shape[0]; eig = (S**2)/(n-1); rat = eig/eig.sum()
    p = int(np.searchsorted(np.cumsum(rat), thr) + 1)
    scores = U[:, :p]*S[:p]
    comps  = Vt[:p, :]
    return scores, comps, p

def kmeans_l2(X, k=2, n_init=25, iters=200, tol=1e-6, seed=42):
    rng = np.random.default_rng(seed); n, d = X.shape
    best_inertia, best_lab, best_C = np.inf, None, None
    for _ in range(n_init):
        C = X[rng.choice(n, size=k, replace=False)].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            d2 = np.vstack([np.sum((X-C[i])**2, axis=1) for i in range(k)]).T
            new_lab = np.argmin(d2, axis=1)
            new_C = np.zeros_like(C)
            for i in range(k):
                pts = X[new_lab==i]
                new_C[i] = pts.mean(axis=0) if len(pts) else X[rng.integers(0,n)]
            if norm(new_C-C) < tol: C = new_C; lab = new_lab; break
            C, lab = new_C, new_lab
        inertia = sum(((X[lab==i]-C[i])**2).sum() for i in range(k))
        if inertia < best_inertia: best_inertia, best_lab, best_C = inertia, lab.copy(), C.copy()
    return best_lab, best_C

def spherical_kmeans(X, k=5, n_init=25, iters=200, tol=1e-6, seed=123):
    rng = np.random.default_rng(seed)
    Xn = X/np.maximum(norm(X, axis=1, keepdims=True), 1e-12)
    n, d = Xn.shape
    best_obj, best_lab, best_C = np.inf, None, None
    for _ in range(n_init):
        C = Xn[rng.choice(n, size=k, replace=False)]
        C = C/np.maximum(norm(C, axis=1, keepdims=True), 1e-12)
        for _ in range(iters):
            sims = Xn @ C.T
            lab = np.argmax(sims, axis=1)
            newC = np.zeros_like(C)
            for i in range(k):
                pts = Xn[lab==i]
                if len(pts)==0: newC[i] = Xn[rng.integers(0,n)]
                else:
                    m = pts.mean(axis=0); newC[i] = m/np.maximum(norm(m),1e-12)
            if norm(newC-C) < tol: C = newC; break
            C = newC
        obj = np.sum(1.0 - (Xn @ C.T)[np.arange(n), lab])
        if obj < best_obj: best_obj, best_lab, best_C = obj, lab.copy(), C.copy()
    return best_lab, best_C

def fuzzy_probs_from_dist(d, eps=1e-12):
    d = np.asarray(d, float); s = d.sum()
    if not np.isfinite(s) or s < eps:
        p = np.zeros_like(d); p[np.argmin(d)] = 1.0; return p
    z = 1.0 - d/s; denom = z.sum()
    if not np.isfinite(denom) or abs(denom)<eps:
        p = np.zeros_like(d); p[np.argmin(d)] = 1.0; return p
    p = np.maximum(z/denom, 0.0); s2 = p.sum()
    return (p if s2>eps else np.eye(len(d))[np.argmin(d)])

def cosine_distance_matrix(X, C):
    Xn = X/np.maximum(norm(X, axis=1, keepdims=True), 1e-12)
    Cn = C/np.maximum(norm(C, axis=1, keepdims=True), 1e-12)
    return 1.0 - (Xn @ Cn.T)

# Hungarian matching (cosine)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def match_centroids_cosine(prevC, currC):
    m = min(prevC.shape[1], currC.shape[1])
    P = prevC[:, :m]/np.maximum(norm(prevC[:, :m], axis=1, keepdims=True),1e-12)
    Q = currC[:, :m]/np.maximum(norm(currC[:, :m], axis=1, keepdims=True),1e-12)
    cost = 1.0 - (P @ Q.T)
    r = cost.shape[0]
    if SCIPY_OK:
        row, col = linear_sum_assignment(cost)
        perm = np.empty(r, int)
        for i,j in zip(row,col): perm[i]=j
    else:
        perm, used = -np.ones(r,int), set()
        for i in range(r):
            j = min((j for j in range(r) if j not in used), key=lambda j: cost[i,j])
            perm[i]=j; used.add(j)
    return perm

# ---------- Rolling loop ----------
macro_dates = pd.to_datetime(df_cc["sasdate"].values)
X_full = df_cc.drop(columns=["sasdate"]).to_numpy(float)
T, M = X_full.shape
W, r = 48, 5

records, E_mats = [], {}
prevC2 = None

for end_idx in range(W-1, T-1):
    start_idx = end_idx - (W-1)
    idx = np.arange(start_idx, end_idx+1)
    Xw = X_full[idx, :]
    Zw = zscore_window(Xw)
    scores, comps, p = pca_svd_95(Zw, thr=0.95)  # (W x p)
    # Stage 1: ℓ2 k-means, k=2
    lab2, cen2 = kmeans_l2(scores, k=2)
    size0 = (lab2==0).sum(); size1 = W - size0
    outlier_id = 0 if size0<=size1 else 1
    typical_ids = np.where(lab2 != outlier_id)[0]
    # Stage 2: cosine k-means, k=r on typical months
    Xtyp = scores[typical_ids, :]
    rr = r if Xtyp.shape[0] >= r else max(1, min(r, Xtyp.shape[0]))
    lab_s2, C2 = spherical_kmeans(Xtyp, k=rr)
    # Label matching across windows
    if rr==r and prevC2 is not None:
        perm = match_centroids_cosine(prevC2, C2)
        C2 = C2[perm]
        inv = np.empty_like(perm); inv[perm]=np.arange(len(perm))
        lab_s2 = inv[lab_s2]
    # Hard labels for all months in window
    hard = np.zeros(W, int)  # Regime 0 default
    if rr>0: hard[typical_ids] = lab_s2 + 1
    prevC2 = C2.copy() if rr==r else None

    # Probabilities at t (end of window)
    x_t = scores[-1, :]
    d1 = np.array([norm(x_t-cen2[0]), norm(x_t-cen2[1])], float)
    p_stage1 = fuzzy_probs_from_dist(d1)                   # Eq. (1) stage-1
    P_reg0 = p_stage1[outlier_id]
    d2 = cosine_distance_matrix(x_t.reshape(1,-1), C2).ravel() if rr>0 else np.array([])
    p_stage2 = fuzzy_probs_from_dist(d2) if rr>0 else np.array([])
    if rr<r:  # pad if fewer than r clusters formed in this window
        pad = np.zeros(r); pad[:rr] = p_stage2; p_stage2 = pad
    # Combine with Eq. (4) and renormalize to Eq. (6)
    eps = 1e-12
    Pmax = p_stage2.max() if p_stage2.size>0 else 0.0
    P_reg0_clip = min(max(P_reg0, eps), 1.0-eps)
    P_R0_scaled = - Pmax * math.log2(1.0 - P_reg0_clip) if Pmax>0 else 0.0  # Eq. (4)
    probs = np.concatenate([[P_R0_scaled], p_stage2])
    probs = probs/probs.sum() if probs.sum()>0 else np.eye(r+1)[hard[-1]]
    hard_label_t = int(np.argmax(probs))

    # Transition matrix E_t (Eq. (5)) with self-loop fallback # 수정했음. (대각행렬 정규화)
    K = r+1
    E = np.zeros((K,K), float)
    for a in range(W-1):
        E[hard[a], hard[a+1]] += 1.0
    for i in range(K):
        s = E[i].sum()
        if s == 0:
            E[i,i] = 1.0  # Self-loop if no transitions
        else:
            E[i,:] /= s   # Normalize row to get probabilities

    p_t = probs
    p_tp1 = p_t @ E  # Eq. (7)

    date_t, date_tp1 = macro_dates[end_idx], macro_dates[end_idx+1]
    rec = {"sasdate_t": date_t, "sasdate_tp1": date_tp1, "hard_label_t": hard_label_t,
           "pca_p_components": p, "typical_k_r": rr}
    rec["p_R0_t"] = float(p_t[0]); rec["ptp1_R0"] = float(p_tp1[0])
    for j in range(1, K):
        rec[f"p_R{j}_t"]  = float(p_t[j])
        rec[f"ptp1_R{j}"] = float(p_tp1[j])
    records.append(rec)
    E_mats[str(date_t.date())] = E

df_step1 = pd.DataFrame(records).sort_values("sasdate_t").reset_index(drop=True)
df_step1.to_csv("section5_step1_regime_probs.csv", index=False)
np.savez("section5_step1_E_matrices.npz", **E_mats)

# Simple checks
df_step1["p_sum"]    = df_step1[[f"p_R{k}_t"  for k in range(0, r+1)]].sum(axis=1)
df_step1["ptp1_sum"] = df_step1[[f"ptp1_R{k}" for k in range(0, r+1)]].sum(axis=1)
print("Windows:", df_step1.shape[0], "| first t:", df_step1["sasdate_t"].min().date(),
      "| last t:", df_step1["sasdate_t"].max().date())
print("p_t sums in [", df_step1["p_sum"].min(), ",", df_step1["p_sum"].max(), "]",
      "| p_{t+1} sums in [", df_step1["ptp1_sum"].min(), ",", df_step1["ptp1_sum"].max(), "]")
print("\nHead:\n", df_step1.head(3).to_string(index=False))
print("\nTail:\n", df_step1.tail(3).to_string(index=False))
