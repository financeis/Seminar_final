# Step 3 — Probability-weighted aggregation of per‑regime ridge forecasts (Eq. 14)
# Inputs expected (from your earlier runs):
#   Step 1: /mnt/data/section5_step1_regime_probs.csv
#   Step 2: /mnt/data/section5_step2_per_regime_ridge_forecasts_DEMO.csv
# Outputs:
#   /mnt/data/section5_step3_ridge_aggregated_forecasts.csv
#
# If inputs are missing, the cell auto‑generates a small demo slice (last ~10 t) so aggregation will still run.
# This implements Eq. 14 precisely, using only Regimes 1..r (exclude Regime 0).  :contentReference[oaicite:3]{index=3}

import pandas as pd, numpy as np, math
from pathlib import Path

r = 5
TICKERS = ["SPY","XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
W = 48

STEP1_CANDIDATES = [
    Path("section5_step1_regime_probs.csv"),
    Path("rolling_regime_next_probabilities_step1.csv"),
]
STEP2_CANDIDATES = [
    Path("section5_step2_per_regime_ridge_forecasts_DEMO.csv"),
    Path("section5_step2_per_regime_ridge_forecasts_SKLEARN_MINIDEMO.csv"),
    Path("section5_step2_per_regime_ridge_forecasts_AUTOGEN.csv"),
]

def synthesize_etf_bom_returns(index_dates):
    """Create a realistic-looking 10-ETF monthly return panel for demo purposes."""
    n = len(index_dates)
    rng = np.random.default_rng(123)
    mkt = rng.normal(loc=0.004, scale=0.035, size=n)  # ~0.4% monthly, 3.5% vol
    betas = {"SPY":1.00,"XLB":1.10,"XLE":1.20,"XLF":1.00,"XLI":1.05,"XLK":1.15,"XLP":0.70,"XLU":0.50,"XLV":0.80,"XLY":1.10}
    noise = {"SPY":0.010,"XLB":0.015,"XLE":0.020,"XLF":0.012,"XLI":0.013,"XLK":0.018,"XLP":0.009,"XLU":0.008,"XLV":0.010,"XLY":0.017}
    data = {tkr: betas[tkr]*mkt + rng.normal(0.0, noise[tkr], size=n) for tkr in TICKERS}
    df = pd.DataFrame(data, index=index_dates); df.index.name = "Date"
    return df

def match_centroids_cosine(prevC, currC):
    # robust to different PCA dimensions across windows
    import numpy as np
    m = min(prevC.shape[1], currC.shape[1])
    P = prevC[:, :m] / np.maximum(np.linalg.norm(prevC[:, :m], axis=1, keepdims=True), 1e-12)
    Q = currC[:, :m] / np.maximum(np.linalg.norm(currC[:, :m], axis=1, keepdims=True), 1e-12)
    cost = 1.0 - (P @ Q.T)
    rloc = cost.shape[0]; perm = np.zeros(rloc, int); used=set()
    for i in range(rloc):
        j=min((j for j in range(rloc) if j not in used), key=lambda j: cost[i,j])
        perm[i]=j; used.add(j)
    return perm

# --- Helpers: small auto-generation of Step‑2 and Step‑1 if inputs missing (kept fast & faithful) ---
def load_step2_or_autogen():
    for p in STEP2_CANDIDATES:
        if p.exists():
            return pd.read_csv(p, parse_dates=["sasdate_t","sasdate_tp1"]), p, None
    # Auto‑generate a minimal Step‑2 (last ~10 decision months) with RidgeCV
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    fred = Path("FRED-MD_2024m12.csv")
    if not fred.exists():
        raise FileNotFoundError("Missing FRED‑MD CSV to auto‑generate Step‑2.")
    # 1) Macro prep (t‑codes, 95% PCA)  :contentReference[oaicite:4]{index=4}
    trow = pd.read_csv(fred, header=0, nrows=1)
    tcode_map = {c:int(trow.iloc[0][c]) for c in trow.columns if c!="sasdate" and pd.notna(trow.iloc[0][c])}
    raw = pd.read_csv(fred, header=0, skiprows=[1])
    raw["sasdate"] = pd.to_datetime(raw["sasdate"], errors="coerce")
    raw = raw.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
    start_date = pd.Timestamp("1978-07-01"); end_date = pd.Timestamp("2024-09-01")
    df = raw[(raw["sasdate"]>=start_date)&(raw["sasdate"]<=end_date)].reset_index(drop=True)
    group6_all = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                  "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM",
                  "AAAFFM","BAAFFM","TWEXAFEGSMTHx","EXSZUSx","EXJPUSx","EXUSUKx","EXCAUSx"]
    must_keep = ["FEDFUNDS","CP3Mx","TB3MS","TB6MS","GS1","GS5","GS10","AAA","BAA",
                 "COMPAPFFx","TB3SMFFM","TB6SMFFM","T1YFFM","T5YFFM","T10YFFM","AAAFFM","BAAFFM",
                 "WPSFD49207","WPSFD49502","WPSID61","WPSID62","OILPRICEx"]
    avail = [c for c in df.columns if c!="sasdate"]
    kept = [v for v in avail if (v not in group6_all) or (v in must_keep)]
    def transform_series(x, t):
        x = pd.to_numeric(x, errors="coerce")
        if t==1: y=x
        elif t==2: y=x.diff(1)
        elif t==3: y=x.diff(1).diff(1)
        elif t==4: y=np.log(x.where(x>0))
        elif t==5: y=np.log(x.where(x>0)).diff(1)
        elif t==6: y=np.log(x.where(x>0)).diff(1).diff(1)
        elif t==7: y=(x/x.shift(1)-1) - (x.shift(1)/x.shift(2)-1)
        else: y=pd.Series(index=x.index, dtype=float)
        return y.astype(float)
    work = df[["sasdate"]+kept].copy()
    df_t = pd.DataFrame({"sasdate": work["sasdate"]})
    for v in kept:
        tc = tcode_map.get(v, None)
        df_t[v] = transform_series(work[v], tc) if tc is not None else np.nan
    protected=set(must_keep)
    drop_vars=[]
    for v in kept:
        y=df_t[v]; a=y.first_valid_index(); b=y.last_valid_index()
        gap = 1.0 if (a is None or b is None) else y.loc[a:b].isna().mean()
        if (v not in protected) and (gap>0.02): drop_vars.append(v)
    kept2 = [v for v in kept if v not in drop_vars]
    df_cc = df_t[["sasdate"]+kept2].dropna().reset_index(drop=True)
    macro_dates = pd.to_datetime(df_cc["sasdate"])
    # 2) ETF returns (load if exists; otherwise synthesize macro‑aligned demo)
    etf_path = Path("etf_bom_returns_aligned_demo.csv")
    if etf_path.exists():
        R = pd.read_csv(etf_path, index_col=0, parse_dates=True)
    else:
        R = synthesize_etf_bom_returns(pd.DatetimeIndex(macro_dates))
        R.to_csv(etf_path)
    # Align calendars
    common_idx = pd.DatetimeIndex(sorted(set(macro_dates)&set(R.index)))
    df_cc = df_cc[df_cc["sasdate"].isin(common_idx)].reset_index(drop=True)
    macro_dates = pd.to_datetime(df_cc["sasdate"])
    R = R.loc[common_idx].sort_index()
    X_full = df_cc.drop(columns=["sasdate"]).to_numpy(float)
    # Helpers: z‑score per window, PCA 95%, two‑stage k‑means with label matching  :contentReference[oaicite:5]{index=5}
    def zscore_window(X):
        mu=X.mean(axis=0); sd=X.std(axis=0, ddof=1); keep=sd>0
        return (X[:,keep]-mu[keep])/sd[keep]
    def pca_svd_95(Z, thr=0.95):
        U,S,Vt = np.linalg.svd(Z, full_matrices=False)
        n=Z.shape[0]; ev=(S**2)/(n-1); p=int(np.searchsorted(np.cumsum(ev/ev.sum()),thr)+1)
        return U[:,:p]*S[:p], Vt[:p,:], p
    def kmeans_l2(X, k=2, n_init=8, iters=80, seed=11):
        rng=np.random.default_rng(seed); n=X.shape[0]
        best,bl,bc=np.inf,None,None
        for _ in range(n_init):
            C=X[rng.choice(n,k,replace=False)].copy()
            for _ in range(iters):
                d2=np.vstack([np.sum((X-C[i])**2,axis=1) for i in range(k)]).T
                L=np.argmin(d2,axis=1)
                newC=np.zeros_like(C)
                for i in range(k):
                    pts=X[L==i]
                    newC[i]=pts.mean(axis=0) if len(pts) else X[rng.integers(0,n)]
                if np.linalg.norm(newC-C)<1e-6: C=newC; break
                C=newC
            inertia=sum(((X[L==i]-C[i])**2).sum() for i in range(k))
            if inertia<best: best,bl,bc=inertia,L.copy(),C.copy()
        return bl,bc
    def spherical_kmeans(X, k=5, n_init=8, iters=80, seed=12):
        rng=np.random.default_rng(seed)
        Xn=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
        n=Xn.shape[0]
        best,bl,bc=np.inf,None,None
        for _ in range(n_init):
            C=Xn[rng.choice(n,k,replace=False)]
            C=C/np.maximum(np.linalg.norm(C,axis=1,keepdims=True),1e-12)
            for _ in range(iters):
                sims=Xn@C.T; L=np.argmax(sims,axis=1)
                newC=np.zeros_like(C)
                for i in range(k):
                    pts=Xn[L==i]
                    if len(pts)==0: newC[i]=Xn[rng.integers(0,n)]
                    else:
                        m=pts.mean(axis=0); newC[i]=m/np.maximum(np.linalg.norm(m),1e-12)
                if np.linalg.norm(newC-C)<1e-6: C=newC; break
                C=newC
            obj=np.sum(1.0-(Xn@C.T)[np.arange(n),L])
            if obj<best: best,bl,bc=obj,L.copy(),C.copy()
        return bl,bc

    # Rolling ridge — small slice (last ~10 decision months)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    ALPHAS=np.logspace(-3,1,8); cv=LeaveOneOut()
    T=len(macro_dates); start_decision=W-1; end_decision=T-2
    decision_idx = list(range(start_decision, end_decision+1))[-10:]
    rows=[]; prevC=None
    for t_end in decision_idx:
        t_start=t_end-(W-1); idx=np.arange(t_start,t_end+1)
        Z=zscore_window(X_full[idx,:])
        scores, comps, p_dim = pca_svd_95(Z, thr=0.95)
        lab2, cen2 = kmeans_l2(scores, k=2)
        size=np.bincount(lab2, minlength=2); outlier_id=0 if size[0]<=size[1] else 1
        typical=np.where(lab2!=outlier_id)[0]
        Xtyp=scores[typical,:]
        rloc=r if Xtyp.shape[0]>=r else max(1, min(r, Xtyp.shape[0]))
        lab_s2, C2 = spherical_kmeans(Xtyp, k=rloc)
        if rloc==r and prevC is not None:
            perm = match_centroids_cosine(prevC, C2)
            C2 = C2[perm]
            inv=np.empty_like(perm); inv[perm]=np.arange(len(perm))
            lab_s2=inv[lab_s2]
        prevC=C2.copy() if rloc==r else None
        hard=np.zeros(W,int)
        if rloc>0: hard[typical]=lab_s2+1
        x_t=scores[-1,:].reshape(1,-1)
        row={"sasdate_t": macro_dates[t_end], "sasdate_tp1": macro_dates[t_end+1]}
        for i_reg in range(1, r+1):
            tau = np.where(hard[:-1]==i_reg)[0]
            if len(tau)<6: tau=np.arange(0, W-1)
            Xi=scores[tau,:]; y_idx=idx[tau]+1
            for tkr in TICKERS:
                y_vec = R.iloc[y_idx][tkr].to_numpy(float)
                model = RidgeCV(alphas=ALPHAS, fit_intercept=True, cv=cv, scoring="neg_mean_squared_error")
                model.fit(Xi, y_vec)
                row[f"yhat_R{i_reg}_{tkr}"] = float(model.predict(x_t)[0])
        rows.append(row)
    df2 = pd.DataFrame(rows).sort_values("sasdate_t").reset_index(drop=True)
    out2 = Path("section5_step2_per_regime_ridge_forecasts_AUTOGEN.csv")
    df2.to_csv(out2, index=False)
    macro_ctx = (df_cc.drop(columns=["sasdate"]).to_numpy(float), pd.to_datetime(df_cc["sasdate"]))
    return df2, out2, macro_ctx

def load_step1_or_autogen(macro_ctx, decision_dates):
    for p in STEP1_CANDIDATES:
        if p.exists():
            return pd.read_csv(p, parse_dates=["sasdate_t","sasdate_tp1"]), p
    # Auto‑generate p_{t+1} for those decision months (Eqs. 6–7 + the k‑means probability heuristic, including Eq. 4 scaling)  :contentReference[oaicite:6]{index=6}
    X_full, macro_dates = macro_ctx
    def zscore_window(X):
        mu=X.mean(axis=0); sd=X.std(axis=0, ddof=1); keep=sd>0
        return (X[:,keep]-mu[keep])/sd[keep]
    def pca_svd_95(Z, thr=0.95):
        U,S,Vt=np.linalg.svd(Z, full_matrices=False)
        n=Z.shape[0]; ev=(S**2)/(n-1); p=int(np.searchsorted(np.cumsum(ev/ev.sum()),thr)+1)
        return U[:,:p]*S[:p], Vt[:p,:], p
    def kmeans_l2(X, k=2, n_init=8, iters=80, seed=9):
        rng=np.random.default_rng(seed); n=X.shape[0]
        best,bl,bc=np.inf,None,None
        for _ in range(n_init):
            C=X[rng.choice(n,k,replace=False)].copy()
            for _ in range(iters):
                d2=np.vstack([np.sum((X-C[i])**2,axis=1) for i in range(k)]).T
                L=np.argmin(d2,axis=1)
                newC=np.zeros_like(C)
                for i in range(k):
                    pts=X[L==i]
                    newC[i]=pts.mean(axis=0) if len(pts) else X[rng.integers(0,n)]
                if np.linalg.norm(newC-C)<1e-6: C=newC; break
                C=newC
            inertia=sum(((X[L==i]-C[i])**2).sum() for i in range(k))
            if inertia<best: best,bl,bc=inertia,L.copy(),C.copy()
        return bl,bc
    def spherical_kmeans(X, k=5, n_init=8, iters=80, seed=10):
        rng=np.random.default_rng(seed)
        Xn=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
        n=Xn.shape[0]
        best,bl,bc=np.inf,None,None
        for _ in range(n_init):
            C=Xn[rng.choice(n,k,replace=False)]
            C=C/np.maximum(np.linalg.norm(C,axis=1,keepdims=True),1e-12)
            for _ in range(iters):
                sims=Xn@C.T; L=np.argmax(sims,axis=1)
                newC=np.zeros_like(C)
                for i in range(k):
                    pts=Xn[L==i]
                    if len(pts)==0: newC[i]=Xn[rng.integers(0,n)]
                    else:
                        m=pts.mean(axis=0); newC[i]=m/np.maximum(np.linalg.norm(m),1e-12)
                if np.linalg.norm(newC-C)<1e-6: C=newC; break
                C=newC
            obj=np.sum(1.0-(Xn@C.T)[np.arange(n),L])
            if obj<best: best,bl,bc=obj,L.copy(),C.copy()
        return bl,bc
    def fuzzy_from_dist(d):
        d=np.asarray(d,float); s=d.sum()
        if not np.isfinite(s) or s<=1e-12:
            p=np.zeros_like(d); p[d.argmin()]=1.0; return p
        w=1.0 - d/s; den=w.sum()
        if den<=1e-12:
            inv=1.0/np.maximum(d,1e-12); inv/=inv.sum(); return inv
        p=w/den; p=np.maximum(p,0.0); p/=p.sum(); return p
    # compute p_{t+1}
    records=[]; prevC=None
    date_to_idx = {d:i for i,d in enumerate(macro_dates)}
    decision_idx = [date_to_idx[pd.Timestamp(d)] for d in decision_dates if pd.Timestamp(d) in date_to_idx]
    for t_end in decision_idx:
        t_start=t_end-(W-1)
        if t_start<0 or (t_end+1)>=len(macro_dates): continue
        idx=np.arange(t_start,t_end+1)
        Z=zscore_window(X_full[idx,:])
        scores, comps, p_dim = pca_svd_95(Z, thr=0.95)
        lab2, cen2 = kmeans_l2(scores, k=2)
        size=np.bincount(lab2, minlength=2); outlier_id=0 if size[0]<=size[1] else 1
        typical=np.where(lab2!=outlier_id)[0]
        Xtyp=scores[typical,:]
        rloc=r if Xtyp.shape[0]>=r else max(1, min(r, Xtyp.shape[0]))
        lab_s2, C2 = spherical_kmeans(Xtyp, k=rloc)
        if rloc==r and prevC is not None:
            perm = match_centroids_cosine(prevC, C2)
            C2 = C2[perm]
            inv=np.empty_like(perm); inv[perm]=np.arange(len(perm))
            lab_s2=inv[lab_s2]
        prevC=C2.copy() if rloc==r else None
        # hard labels for transition counts  (Eq. 5)  :contentReference[oaicite:7]{index=7}
        hard=np.zeros(W,int)
        if rloc>0: hard[typical]=lab_s2+1
        # probability at t (k‑means distance heuristic + Eq. 4 scaling for R0)
        x_t=scores[-1,:]
        d_l2=np.array([np.linalg.norm(x_t-cen2[0]), np.linalg.norm(x_t-cen2[1])], float)
        p2=fuzzy_from_dist(d_l2); P0=p2[outlier_id]
        x_n = x_t/ max(np.linalg.norm(x_t), 1e-12)
        Cn  = C2 / np.maximum(np.linalg.norm(C2,axis=1,keepdims=True),1e-12) if rloc>0 else np.zeros((0,len(x_t)))
        d_cos = (1.0 - (Cn @ x_n)) if rloc>0 else np.array([])
        pR = fuzzy_from_dist(d_cos) if rloc>0 else np.array([])
        if rloc<r:
            pad=np.zeros(r); pad[:rloc]=pR; pR=pad
        eps=1e-12; Pmax=pR.max() if pR.size>0 else 0.0
        P0c=min(max(P0,eps),1.0-eps)
        PR0 = - Pmax * math.log2(1.0 - P0c) if Pmax>0 else 0.0   # Eq. 4  :contentReference[oaicite:8]{index=8}
        p_all = np.concatenate([[PR0], pR])
        p_all = p_all/p_all.sum() if p_all.sum()>0 else np.eye(r+1)[hard[-1]]
        # Markov one‑step update to next‑month (Eq. 7):  \tilde p_{t+1} = \tilde p_t^T E_t.  :contentReference[oaicite:9]{index=9}
        K=r+1; E=np.zeros((K,K))
        for a,b in zip(hard[:-1], hard[1:]): E[a,b]+=1.0
        for i in range(K):
            s=E[i].sum()
            if s==0: E[i,i]=1.0
            else: E[i,:]/=s
        p_next = p_all @ E
        rec={"sasdate_t": macro_dates[t_end], "sasdate_tp1": macro_dates[t_end+1]}
        for k in range(0, r+1): rec[f"ptp1_R{k}"]=float(p_next[k])
        records.append(rec)
    df1 = pd.DataFrame(records).sort_values("sasdate_t").reset_index(drop=True)
    out1 = Path("section5_step1_regime_probs_AUTOGEN_for_step3.csv")
    df1.to_csv(out1, index=False)
    return df1, out1

# ---- Load / autogen Step‑2 and Step‑1 ----
step2 = step2_path = macro_ctx = None
step2, step2_path, macro_ctx = load_step2_or_autogen()
decision_dates = list(step2["sasdate_t"])
step1, step1_path = load_step1_or_autogen(macro_ctx, decision_dates)

# ---- Aggregate (Eq. 14; use only Regimes 1..r) ----
prob_cols = [f"ptp1_R{k}" for k in range(0, r+1)]
dfm = pd.merge(step2, step1[["sasdate_t","sasdate_tp1"]+prob_cols], on=["sasdate_t","sasdate_tp1"], how="inner").sort_values("sasdate_t")
rows=[]
for _, row in dfm.iterrows():
    rec={"sasdate_t": row["sasdate_t"], "sasdate_tp1": row["sasdate_tp1"]}
    probs = np.array([row[f"ptp1_R{k}"] for k in range(1, r+1)], dtype=float)
    if not np.isfinite(probs.sum()) or probs.sum()<=0: probs = np.ones(r)/r
    for tkr in TICKERS:
        yis = np.array([row.get(f"yhat_R{i}_{tkr}", np.nan) for i in range(1, r+1)], dtype=float)
        rec[tkr] = np.nan if np.isnan(yis).any() else float((probs * yis).sum())
    rows.append(rec)

df_agg = pd.DataFrame(rows).sort_values("sasdate_t").reset_index(drop=True)
out_path = Path("section5_step3_ridge_aggregated_forecasts.csv")
df_agg.to_csv(out_path, index=False)

print("=== Step 3 Aggregation Complete (Eq. 14) ===")
print("Step-1 probs source:", step1_path)
print("Step-2 forecasts source:", step2_path)
print("Aggregated rows:", len(df_agg), "| range:", df_agg['sasdate_t'].min().date(), "→", df_agg['sasdate_t'].max().date())
print("\nHead (5):")
print(df_agg.head(5).to_string(index=False))
print("\nTail (5):")
print(df_agg.tail(5).to_string(index=False))
print("\nSaved:", out_path)
