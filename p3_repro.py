# Reproduce the main paper's reference estimates (Table 4 cols 1-4, Table 7, unresolved share) on the rebuilt data
import time; from p3lib import *
t0=time.time(); A=pd.read_parquet("data/p3_all.parquet"); M=pd.read_csv("data/ca_measures.csv"); z=lambda s:(s-s.mean())/s.std()
for nm in ("hardship","low_inc","pct_black","pct_hisp","viol_rate"): A[nm]=A.ca.map(dict(zip(M.ca,z(M[nm])))).astype("float32")
A["ty"]=(A.tcode.astype(np.int32)*10+(A.year-2019)).astype(np.int32)
S=A[(~A["info"])&(A.chan!=3)&A.nd&A.done].copy(); S["lhrs"]=np.log1p(np.minimum(S.hours,S.groupby("tcode").hours.transform(lambda x:x.quantile(0.99)))).astype("float32"); print(f"baseline sample {len(S):,} [{time.time()-t0:.0f}s]",flush=True)
out={}; spec={"Type x year FE only":(["hardship"],["ty"]),"+ month, weekday, hour FE":(["hardship"],["ty","ym","dow","hod"]),"+ channel FE":(["hardship"],["ty","ym","dow","hod","chan"]),"+ workload (baseline)":(["hardship","lwl"],["ty","ym","dow","hod","chan"])}
for k,(X,F) in spec.items():
    r=reg(S,"lhrs",X,F,"ca"); out[k]={"coef":float(r.b["hardship"]),"se":r.se("hardship"),"n":int(r.n)}; print(f"{k:32s} hardship {r.b['hardship']:.3f} ({r.se('hardship'):.3f}) n={r.n:,} [{time.time()-t0:.0f}s]",flush=True)
F=["ty","ym","dow","hod","chan"]
for tag,X in (("hardship + Black + Hispanic",["hardship","pct_black","pct_hisp","lwl"]),("hardship + Black",["hardship","pct_black","lwl"]),("low-income + Black",["low_inc","pct_black","lwl"]),("hardship + violent crime",["hardship","viol_rate","lwl"])):
    r=reg(S,"lhrs",X,F,"ca"); out[tag]={k:[float(r.b[k]),r.se(k),r.p(k)] for k in X if k!="lwl"}; print(tag,{k:(round(float(r.b[k]),3),round(r.se(k),3)) for k in X if k!="lwl"},flush=True)
# unresolved share (quality sample): resident non-duplicate requests, status Open
c=pd.read_parquet("data/core_request_file.parquet",columns=["chan","nd","status","year","ym","dow","hod","ca","type","lwl"]); c=c[(c.chan!="city")&c.nd].copy()
c["hardship"]=c.ca.map(dict(zip(M.ca,z(M.hardship)))); c["y"]=(100*(c.status=="Open")).astype("float32"); c["ty"]=(c.type.astype("category").cat.codes.astype(np.int32)*10+(c.year-2019)).astype(np.int32); c["chc"]=c.chan.astype("category").cat.codes
r=reg(c,"y",["hardship","lwl"],["ty","ym","dow","hod","chc"],"ca"); out["unresolved"]={"coef":float(r.b["hardship"]),"se":r.se("hardship"),"mean":float(c.y.mean()),"n":int(r.n)}; print("unresolved (pp per SD):",round(float(r.b['hardship']),3),round(r.se('hardship'),3),"mean",round(float(c.y.mean()),2),f"n={r.n:,}")
savej("R0_reproduction",out)
