import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,"code")
from fe import demean, _codes
def savej(name,obj): json.dump(obj,open(f"results/{name}.json","w"),indent=1,default=float)
class Res:
    def __init__(s,b,V,names,n,G): s.b=pd.Series(b,index=names); s.V=pd.DataFrame(V,index=names,columns=names); s.n=n; s.G=G
    def se(s,k): return float(np.sqrt(s.V.loc[k,k]))
    def p(s,k): return float(2*stats.t.sf(abs(s.b[k]/s.se(k)),s.G-1))
    def lin(s,w):
        v=pd.Series(w); est=float((s.b[v.index]*v).sum()); se=float(np.sqrt(v.values@s.V.loc[v.index,v.index].values@v.values)); return est,se,float(2*stats.t.sf(abs(est/se),s.G-1))
    def wald(s,names):
        b=s.b[names].values; Vi=np.linalg.inv(s.V.loc[names,names].values); F=float(b@Vi@b)/len(names); return F,float(stats.f.sf(F,len(names),s.G-1))
def reg(df,y,X,fes,cluster,tol=1e-7,maxit=120):
    d=df[list(dict.fromkeys([y]+X+fes+[cluster]))].dropna()
    M=demean(d[[y]+X].to_numpy(np.float64),[d[f].to_numpy() for f in fes],tol=tol,maxit=maxit); yt,Xt=M[:,0],M[:,1:]
    A=Xt.T@Xt; A_inv=np.linalg.inv(A); b=A_inv@(Xt.T@yt); e=yt-Xt@b; g=_codes(d[cluster].to_numpy()); G=g.max()+1; n,k=Xt.shape
    S=np.column_stack([np.bincount(g,weights=Xt[:,j]*e,minlength=G) for j in range(k)]); V=A_inv@(S.T@S)@A_inv*(G/(G-1))*((n-1)/(n-k)); return Res(b,V,X,n,G)
def _se(Xt,e,A_inv,groups):
    n,k=Xt.shape
    if groups is None: Z=Xt*e[:,None]; V=A_inv@(Z.T@Z)@A_inv*(n/(n-k)); df=n-k
    else:
        G=groups.max()+1; S=np.column_stack([np.bincount(groups,weights=Xt[:,j]*e,minlength=G) for j in range(k)]); V=A_inv@(S.T@S)@A_inv*(G/(G-1))*((n-1)/(n-k)); df=G-1
    return float(np.sqrt(V[0,0])),df
def fit_cell(Y,Mz,lwl,fes,groups,ycols,mcols,tol=1e-7):
    """Demean all outcomes (Y: dict name->array), measures (Mz: dict) and workload once; then run every outcome x measure regression.
    groups: dict name->int codes (None key 'hc1' handled). Returns list of dict records."""
    names=list(Y)+list(Mz)+(["lwl"] if lwl is not None else []); cols=[Y[k] for k in Y]+[Mz[k] for k in Mz]+([lwl] if lwl is not None else [])
    M=np.column_stack(cols).astype(np.float64); M=demean(M,fes,tol=tol,maxit=120); ix={k:i for i,k in enumerate(names)}; out=[]
    for yo in Y:
        yt=M[:,ix[yo]]
        for mk in Mz:
            cols_=[ix[mk]]+([ix["lwl"]] if lwl is not None else []); Xt=M[:,cols_]; A=Xt.T@Xt; A_inv=np.linalg.inv(A); b=A_inv@(Xt.T@yt); e=yt-Xt@b
            rec={"outcome":yo,"measure":mk,"coef":float(b[0]),"n":int(len(yt))}
            for gname in ("hc1","ward","ca"):
                se,df=_se(Xt,e,A_inv,None if gname=="hc1" else groups[gname]); rec["se_"+gname]=se; rec["p_"+gname]=float(2*stats.t.sf(abs(b[0]/se),df))
            out.append(rec)
    return out
