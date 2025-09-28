# Step 2: Per-regime linear ridge models and per-regime forecasts (Eq. 12–13)
# - 48-month rolling; PCA(95%); two-stage k-means (ℓ2→cosine, r=5); label matching across windows.
# - For each regime i and ETF j, fit ridge on window months in regime i: (x_tau -> r_{j, tau+1}); predict yhat_{i,j,t+1}.
# - Saves: forecasts, chosen lambdas, and regime sample counts per window.
#
# Precisely follows Sec. 3.1–3.3 (regimes), Sec. 4.2 (PCA 95%), and Sec. 5.2.3 (ridge).  :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

import numpy as np, pandas as pd, math, os
from numpy.linalg import norm
from pathlib import Path

# -------------------- Config + inputs --------------------
FRED_CSV = Path("FRED-MD_2024m12.csv")
ETF_ALIGNED_CSV = Path("etf_bom_returns_aligned_demo.csv")  # replace with your real aligned file if available
assert FRED_CSV.exists(), "FRED-MD CSV not found"

# Load aligned 10-ETF BoM returns (columns: SPY XLB XLE XLF XLI XLK XLP XLU XLV XLY; index=first-of-month)
R = pd.read_csv(ETF_ALIGNED_CSV, index_col=0, parse_dates=True)
TICKERS = list(R.columns)

# -------------------- Macro preparation (same policy as Sec. 3/4.2) --------------------
trow = pd.read_csv(FRED_CSV, header=0, nrows=1)
tcode_map = {c:int(trow.iloc[0][c]) for c in trow.columns if c!="sasdate" and pd.notna(trow.iloc[0][c])}

df_raw = pd.read_csv(FRED_CSV, header=0, skiprows=[1])
df_raw["sasdate"] = pd.to_datetime(df_raw["sasdate"], errors="coerce")
df_raw = df_raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)

start_date = pd.Timestamp("1978-07-01")
end_date   = pd.Timestamp("2024-09-01")
df = df_raw[(df_raw["sasdate"] >= start_date) & (df_raw["sasdate"] <= end_date)].reset_index(drop=True)

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
    elif tcode == 7: y = (x/x.shift(1)-1) - (x.shift(1)/x.shift(2)-1)
    else:            y = pd.Series(index=x.index, dtype=float)
    return y.astype(float)

work = df[["sasdate"]+kept_vars].copy()
df_t = pd.DataFrame({"sasdate": work["sasdate"]})
for v in kept_vars:
    tc = tcode_map.get(v, None)
    df_t[v] = transform_series(work[v], tc) if tc is not None else np.nan

# Drop series with internal gaps >2% (except must_keep) → complete-case months
protected = set(must_keep)
drop_vars = []
for v in kept_vars:
    y = df_t[v]; a = y.first_valid_index(); b = y.last_valid_index()
    gap = 1.0 if (a is None or b is None) else y.loc[a:b].isna().mean()
    if (v not in protected) and (gap > 0.02): drop_vars.append(v)
kept_vars2 = [v for v in kept_vars if v not in drop_vars]
df_t = df_t[["sasdate"]+kept_vars2]
df_cc = df_t.dropna(how="any").reset_index(drop=True)

# Align macro calendar to ETF returns calendar
macro_dates = pd.to_datetime(df_cc["sasdate"])
common_idx = pd.DatetimeIndex(sorted(set(macro_dates) & set(R.index)))
df_cc = df_cc[df_cc["sasdate"].isin(common_idx)].reset_index(drop=True)
macro_dates = pd.to_datetime(df_cc["sasdate"])
R = R.loc[common_idx].sort_index()
X_full = df_cc.drop(columns=["sasdate"]).to_numpy(float)

# -------------------- Helpers: PCA, k-means, matching --------------------
def zscore_window(X):
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
    keep = sd > 0
    return (X[:, keep] - mu[keep]) / sd[keep]

def pca_svd_95(Z, thr=0.95):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n = Z.shape[0]; eig = (S**2)/(n-1); rat = eig/eig.sum()
    p = int(np.searchsorted(np.cumsum(rat), thr) + 1)
    scores = U[:, :p]*S[:p]
    comps  = Vt[:p, :]
    return scores, comps, p

def kmeans_l2(X, k=2, n_init=8, iters=80, tol=1e-6, seed=1):
    rng = np.random.default_rng(seed); n = X.shape[0]
    best_inertia, best_lab, best_C = np.inf, None, None
    for _ in range(n_init):
        C = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(iters):
            d2 = np.vstack([np.sum((X-C[i])**2, axis=1) for i in range(k)]).T
            L = np.argmin(d2, axis=1)
            newC = np.zeros_like(C)
            for i in range(k):
                pts = X[L==i]
                newC[i] = pts.mean(axis=0) if len(pts) else X[rng.integers(0,n)]
            if norm(newC-C) < tol: C = newC; break
            C = newC
        inertia = sum(((X[L==i]-C[i])**2).sum() for i in range(k))
        if inertia < best_inertia: best_inertia, best_lab, best_C = inertia, L.copy(), C.copy()
    return best_lab, best_C

def spherical_kmeans(X, k=5, n_init=8, iters=80, tol=1e-6, seed=2):
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

def cosine_sim(A,B):
    An = A/np.maximum(norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B/np.maximum(norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T

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

# -------------------- Ridge: precompute per λ (fast LOOCV) --------------------
def ridge_precompute_B_Hdiag(X, lambdas):
    n, p = X.shape
    G = X.T @ X
    B_list, hdiag_list = [], []
    for lam in lambdas:
        A = G + lam * np.eye(p)
        try:
            B = np.linalg.solve(A, X.T)  # p x n
        except np.linalg.LinAlgError:
            B = np.linalg.pinv(A) @ X.T
        hdiag = np.einsum('ik,ki->i', X, B)
        B_list.append(B)
        hdiag_list.append(hdiag)
    return B_list, hdiag_list

def ridge_select_lambda_and_predict(X, y, lambdas, B_list, hdiag_list, x_pred=None):
    best_mse, best_idx = np.inf, -1
    for k in range(len(lambdas)):
        B = B_list[k]; hdiag = hdiag_list[k]
        beta = B @ y; yhat = X @ beta
        e = y - yhat
        denom = np.maximum(1.0 - hdiag, 1e-8)
        looe = (e/denom)**2
        mse = looe.mean()
        if mse < best_mse: best_mse, best_idx = mse, k
    lam_best = float(lambdas[best_idx])
    beta_best = B_list[best_idx] @ y
    y_pred_centered = float(x_pred @ beta_best) if x_pred is not None else None
    return lam_best, beta_best, y_pred_centered

# -------------------- Rolling Step 2 --------------------
W, r = 48, 5
dates = list(macro_dates)
T = len(dates)
assert len(R.index) == T, "Macro and ETF calendars must match"

lambdas = np.logspace(-4, 2, 8)  # 1e-4..1e2

fore_rows, lam_rows, cnt_rows = [], [], []
prevC2 = None

start_decision = W-1
end_decision = T-2             # we need tau+1 for targets
LIMIT_WINDOWS = 60             # DEMO: last 60 decision months; set to None for full run
decision_indices = list(range(start_decision, end_decision+1))
if LIMIT_WINDOWS is not None:
    decision_indices = decision_indices[-LIMIT_WINDOWS:]

for t_end in decision_indices:
    t_start = t_end - (W-1)
    idx = np.arange(t_start, t_end+1)

    # PCA on standardized macro panel in the window
    Zw = zscore_window(X_full[idx, :])
    scores, comps, p_dim = pca_svd_95(Zw, thr=0.95)  # W x p

    # Regime clustering (two-stage)
    lab2, cen2 = kmeans_l2(scores, k=2, n_init=8, iters=80, seed=1)
    size = np.bincount(lab2, minlength=2)
    outlier_id = 0 if size[0] <= size[1] else 1
    typical_ids = np.where(lab2 != outlier_id)[0]

    Xtyp = scores[typical_ids, :]
    rr = r if Xtyp.shape[0] >= r else max(1, min(r, Xtyp.shape[0]))
    lab_s2, C2 = spherical_kmeans(Xtyp, k=rr, n_init=8, iters=80, seed=2)
    if rr==r and prevC2 is not None:
        perm = match_centroids_cosine(prevC2, C2)
        C2 = C2[perm]
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        lab_s2 = inv[lab_s2]
    prevC2 = C2.copy() if rr==r else None

    # Hard labels in window (0 = outliers)
    hard = np.zeros(W, dtype=int)
    if rr>0: hard[typical_ids] = lab_s2 + 1

    # Prepare this month's containers
    yhat_row = {"sasdate_t": dates[t_end], "sasdate_tp1": dates[t_end+1]}
    lam_row  = {"sasdate_t": dates[t_end], "sasdate_tp1": dates[t_end+1]}
    cnt_row  = {"sasdate_t": dates[t_end], "sasdate_tp1": dates[t_end+1]}

    x_t = scores[-1, :]  # latest PCA factors at month t

    # Loop regimes 1..r (Eq. 14 later will use only 1..r; 0 reserved for outliers)  :contentReference[oaicite:8]{index=8}
    for i_reg in range(1, r+1):
        tau_idx = np.where((hard[:-1] == i_reg))[0]  # training tau in [0..W-2] with label i
        cnt_row[f"n_R{i_reg}"] = int(len(tau_idx))
        if len(tau_idx) < 6:   # small-sample fallback: use global rows in the window (still produces yhat_Ri_*)
            tau_idx = np.arange(0, W-1)

        Xi = scores[tau_idx, :]
        # center predictors within the regime-set; center target per-asset
        Xi_mu = Xi.mean(axis=0, keepdims=True)
        Xc = Xi - Xi_mu
        x_pred_c = x_t - Xi_mu.ravel()

        # Precompute B, diag(H) once per regime for all λ
        B_list, hdiag_list = ridge_precompute_B_Hdiag(Xc, lambdas)

        # Do each ETF target
        for tkr in TICKERS:
            y_vec = R.iloc[idx[tau_idx] + 1][tkr].to_numpy(float)  # target r_{j, tau+1}
            y_mu = y_vec.mean()
            yc = y_vec - y_mu
            lam_best, beta_best, y_pred_c = ridge_select_lambda_and_predict(
                Xc, yc, lambdas, B_list, hdiag_list, x_pred=x_pred_c
            )
            y_pred = float(y_pred_c + y_mu)
            yhat_row[f"yhat_R{i_reg}_{tkr}"] = y_pred
            lam_row[f"lambda_R{i_reg}_{tkr}"] = float(lam_best)

    fore_rows.append(yhat_row)
    lam_rows.append(lam_row)
    cnt_rows.append(cnt_row)

# Save artifacts
df_fore = pd.DataFrame(fore_rows).sort_values("sasdate_t").reset_index(drop=True)
df_lam  = pd.DataFrame(lam_rows).sort_values("sasdate_t").reset_index(drop=True)
df_cnt  = pd.DataFrame(cnt_rows).sort_values("sasdate_t").reset_index(drop=True)

out_fore = "section5_step2_per_regime_ridge_forecasts_DEMO.csv"
out_lam  = "section5_step2_ridge_lambdas_DEMO.csv"
out_cnt  = "section5_step2_regime_sample_counts_DEMO.csv"
df_fore.to_csv(out_fore, index=False)
df_lam.to_csv(out_lam, index=False)
df_cnt.to_csv(out_cnt, index=False)

print("Saved (DEMO):")
print(" -", out_fore, "| rows:", len(df_fore))
print(" -", out_lam,  "| rows:", len(df_lam))
print(" -", out_cnt,  "| rows:", len(df_cnt))

print("\nForecasts (head):")
print(df_fore.head(3).to_string(index=False))
print("\nForecasts (tail):")
print(df_fore.tail(3).to_string(index=False))

print("\nSample counts per window (head):")
print(df_cnt.head(5).to_string(index=False))

print("\nNaN cells in forecast table:", int(df_fore.isna().sum().sum()))
