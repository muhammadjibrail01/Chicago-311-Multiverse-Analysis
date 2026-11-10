import numpy as np, pandas as pd
from scipy import stats
def _codes(a): return pd.factorize(np.asarray(a))[0]
def demean(M, fes, tol=1e-9, maxit=200, w=None):
    M=M.astype(np.float64).copy()
    if not fes: return M-M.mean(0)
    cs=[_codes(f) for f in fes]; ns=[c.max()+1 for c in cs]
    ww=np.ones(len(M)) if w is None else w
    cnt=[np.bincount(c,weights=ww,minlength=n) for c,n in zip(cs,ns)]
    for j in range(M.shape[1]):
        v=M[:,j]
        for it in range(maxit):
            old=v.copy()
            for c,n,k in zip(cs,ns,cnt):
                v-= (np.bincount(c,weights=v*ww,minlength=n)/np.maximum(k,1e-12))[c]
            if np.max(np.abs(v-old))<tol: break
        M[:,j]=v
    return M
def feols(df, y, X, fes=(), cluster="ca", w=None, drop_na=True):
    cols=[y]+list(X)+list(fes)+[cluster]+([w] if w else [])
    d=df[cols].dropna() if drop_na else df
    W=None if w is None else d[w].to_numpy(np.float64)
    M=demean(d[[y]+list(X)].to_numpy(np.float64),[d[f].to_numpy() for f in fes],w=W)
    yt,Xt=M[:,0],M[:,1:]
    if W is None: W1=np.ones(len(yt))
    else: W1=W
    XtW=Xt*W1[:,None]; A=XtW.T@Xt; b=np.linalg.solve(A,XtW.T@yt); e=yt-Xt@b
    g=_codes(d[cluster].to_numpy()); G=g.max()+1; n,k=Xt.shape
    S=np.column_stack([np.bincount(g,weights=XtW[:,j]*e,minlength=G) for j in range(k)])
    A_inv=np.linalg.inv(A); V=A_inv@(S.T@S)@A_inv*(G/(G-1))*((n-1)/(n-k))
    se=np.sqrt(np.diag(V)); t=b/se; p=2*stats.t.sf(np.abs(t),G-1)
    r2w=1-np.sum(W1*e**2)/np.sum(W1*(yt-np.average(yt,weights=W1))**2)
    return {"coef":pd.Series(b,index=X),"se":pd.Series(se,index=X),"p":pd.Series(p,index=X),"n":n,"G":G,"r2w":r2w,"ymean":float(np.average(d[y],weights=W)) if W is not None else float(d[y].mean()),"resid":e,"xt":Xt,"idx":d.index}
def stars(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
def fmt(res,v,dig=3):
    return f"{res['coef'][v]:.{dig}f}{stars(res['p'][v])}", f"({res['se'][v]:.{dig}f})"
