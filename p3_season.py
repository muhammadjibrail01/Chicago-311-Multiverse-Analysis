# Does the hardship gradient widen in winter, and does violent crime account for it? (baseline sample, all years)
import time; from p3lib import *
t0=time.time(); A=pd.read_parquet("data/p3_all.parquet"); n_core=int((~A["info"]).sum())
dept=pd.read_parquet("data/core_request_file.parquet",columns=["dept"]).dept.astype(str).values
A["dept"]=np.concatenate([dept,np.array(["info"]*(len(A)-n_core))]); del dept
M=pd.read_csv("data/ca_measures.csv"); z=lambda s:(s-s.mean())/s.std()
for nm in ("hardship","viol_rate"): A[nm]=A.ca.map(dict(zip(M.ca,z(M[nm])))).astype("float32")
S=A[(~A["info"])&(A.chan!=3)&A.nd&A.done].reset_index(drop=True); del A
S["lhrs"]=np.log1p(np.minimum(S.hours,S.groupby("tcode").hours.transform(lambda x:x.quantile(0.99)))).astype("float32")
S["ty"]=(S.tcode.astype(np.int32)*10+(S.year-2019)).astype(np.int32)
S["season"]=S.month.map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).astype("int8"); names=["winter","spring","summer","fall"]
S["tys"]=(S.ty*4+S.season).astype(np.int32)
for i,s in enumerate(names):
    m=(S.season==i).astype("float32"); S["H_"+s]=S.hardship*m; S["V_"+s]=S.viol_rate*m
S["win"]=(S.season==0).astype("float32"); S["H"]=S.hardship; S["H_win"]=S.hardship*S.win; S["V"]=S.viol_rate; S["V_win"]=S.viol_rate*S.win
for y in range(2020,2025): S[f"H_y{y}"]=S.hardship*(S.year==y).astype("float32")
for m in range(1,13): S[f"H_m{m}"]=S.hardship*(S.month==m).astype("float32")
F=["ty","ym","dow","hod","chan"]; HS=["H_"+s for s in names]; VS=["V_"+s for s in names]; out={}
print(f"sample {len(S):,} [{time.time()-t0:.0f}s]; share of requests created in winter: {S.win.mean():.3f}",flush=True)
def pk(r,keys): return {k:[float(r.b[k]),r.se(k),r.p(k)] for k in keys}
def season_block(tag,s):
    r=reg(s,"lhrs",HS+["lwl"],F,"ca"); o=pk(r,HS); o["n"]=int(r.n)
    o["winter_minus_others"]=list(r.lin({"H_winter":1,"H_spring":-1/3,"H_summer":-1/3,"H_fall":-1/3})); o["winter_minus_summer"]=list(r.lin({"H_winter":1,"H_summer":-1}))
    r2=reg(s,"lhrs",["H","H_win","lwl"],F,"ca"); o["H_win"]=pk(r2,["H","H_win"])
    r3=reg(s,"lhrs",["H","H_win"]+[f"H_y{y}" for y in range(2020,2025)]+["lwl"],F,"ca"); o["H_win_year_adj"]=pk(r3,["H","H_win"]); out[tag]=o
    print(f"{tag:34s} n={r.n:>9,} winter {o['H_winter'][0]:.3f} spring {o['H_spring'][0]:.3f} summer {o['H_summer'][0]:.3f} fall {o['H_fall'][0]:.3f} | winter minus others {o['winter_minus_others'][0]:.3f} (SE {o['winter_minus_others'][1]:.3f}) | year-adjusted H x winter {o['H_win_year_adj']['H_win'][0]:.3f} ({o['H_win_year_adj']['H_win'][1]:.3f}) [{time.time()-t0:.0f}s]",flush=True); savej("S1_season",out)
season_block("All departments",S)
# composition-robust version: type x year x season fixed effects
r=reg(S,"lhrs",HS+["lwl"],["tys","ym","dow","hod","chan"],"ca"); out["All departments, type x year x season FE"]=pk(r,HS); out["All departments, type x year x season FE"]["winter_minus_others"]=list(r.lin({"H_winter":1,"H_spring":-1/3,"H_summer":-1/3,"H_fall":-1/3}))
print("tys FE:",{k:round(float(v[0]),3) for k,v in out["All departments, type x year x season FE"].items() if k.startswith("H_")},"winter minus others",round(out["All departments, type x year x season FE"]["winter_minus_others"][0],3),f"[{time.time()-t0:.0f}s]",flush=True); savej("S1_season",out)
# crime: does violent crime account for a winter-specific gradient? (hardship and crime interacted with season)
r=reg(S,"lhrs",HS+VS+["lwl"],F,"ca"); o=pk(r,HS+VS); o["winter_minus_others_H"]=list(r.lin({"H_winter":1,"H_spring":-1/3,"H_summer":-1/3,"H_fall":-1/3})); o["winter_minus_others_V"]=list(r.lin({"V_winter":1,"V_spring":-1/3,"V_summer":-1/3,"V_fall":-1/3})); o["corr_hardship_crime"]=float(M.hardship.corr(M.viol_rate)); out["Crime x season"]=o
print("crime x season: H winter-minus-others %.3f (%.3f) | V winter-minus-others %.3f (%.3f) | V by season %s"%(o["winter_minus_others_H"][0],o["winter_minus_others_H"][1],o["winter_minus_others_V"][0],o["winter_minus_others_V"][1],[round(o[k][0],3) for k in VS]),f"[{time.time()-t0:.0f}s]",flush=True); savej("S1_season",out)
r=reg(S,"lhrs",["H_m%d"%m for m in range(1,13)]+["lwl"],F,"ca"); out["By month"]=pk(r,["H_m%d"%m for m in range(1,13)]); print("by month:",[round(float(out["By month"]["H_m%d"%m][0]),3) for m in range(1,13)],f"[{time.time()-t0:.0f}s]",flush=True); savej("S1_season",out)
for dname,lab in (("Streets and Sanitation","Streets and Sanitation"),("CDOT - Department of Transportation","Transportation (CDOT)"),("DWM - Department of Water Management","Water Management")):
    season_block(lab,S[S.dept==dname].reset_index(drop=True))
