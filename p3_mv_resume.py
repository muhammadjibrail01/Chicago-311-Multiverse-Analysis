# Resumes the multiverse grid (skips cells already in results/M1_grid.json) with a memory-lean loader.
import time, sys, json, os, gc; from p3lib import *; from p3lib import _codes
t0=time.time(); NEED=["info","chan","nd","done","hours","year","month","ym","dow","hod","tcode","ca","ward","lwl","type"]
A=pd.read_parquet("data/p3_all.parquet",columns=NEED)
A["hours"]=A.hours.astype("float32"); A["ca"]=A.ca.astype("int16"); A["ward"]=A.ward.astype("int16"); A["lwl"]=A.lwl.astype("float32")
M=pd.read_csv("data/ca_measures.csv"); z=lambda s:(s-s.mean())/s.std()
for nm in ("hardship","low_inc","pct_black"): A[nm]=A.ca.map(dict(zip(M.ca,z(M[nm])))).astype("float32")
print(f"frame {len(A):,} rows, {A.memory_usage(deep=True).sum()/1e9:.2f} GB [{time.time()-t0:.0f}s]",flush=True)
RECS={"Resident, non-duplicate (baseline)":lambda a:~a["info"]&(a.chan!=3)&a.nd,"+ duplicates":lambda a:~a["info"]&(a.chan!=3),"+ city-generated":lambda a:~a["info"]&a.nd,
      "+ information-only calls":lambda a:((~a["info"]&(a.chan!=3)&a.nd)|a["info"]&a.nd),"+ all three":lambda a:np.ones(len(a),bool)}
WINS={"2019-2024":lambda a:np.ones(len(a),bool),"Excluding 2020":lambda a:a.year!=2020,"2019 only":lambda a:a.year==2019,"2022-2024":lambda a:a.year>=2022,"Winter (Dec-Feb)":lambda a:a.month.isin([12,1,2]),"Summer (Jun-Aug)":lambda a:a.month.isin([6,7,8])}
MEAS=["hardship","low_inc","pct_black"]
out=json.load(open("results/M1_grid.json")) if os.path.exists("results/M1_grid.json") else []
print("cells already in file:",len({(r["records"],r["window"]) for r in out}),flush=True)
done={(r["records"],r["window"]) for r in out}; print("already done cells:",len(done),flush=True)
CELL=sys.argv[1] if len(sys.argv)>1 else None
for rname,rf in RECS.items():
    Ar=A[rf(A)&A.done].copy()
    for wname,wf in WINS.items():
        if (rname,wname) in done: continue
        if CELL is not None and CELL!=f'{list(RECS).index(rname)},{list(WINS).index(wname)}': continue
        s=Ar[wf(Ar)].reset_index(drop=True); h=s.hours.to_numpy(np.float64); ty=(s.tcode.astype(np.int32)*10+(s.year-2019)).to_numpy()
        cap=s.groupby("tcode").hours.quantile(0.99); hw=np.minimum(h,s.tcode.map(cap).to_numpy(np.float64))
        g=pd.Series(h).groupby(ty); rank=(g.rank(pct=True)*100).to_numpy(); med=g.transform("median").to_numpy()
        Y={"log":np.log1p(hw),"rank":rank,"slower":100.0*(h>med),"days":hw/24.0}; sdy={k:float(np.std(v)) for k,v in Y.items()}
        Mz={m:s[m].to_numpy(np.float64) for m in MEAS}; lwl=s.lwl.to_numpy(np.float64)
        groups={"ca":_codes(s.ca.to_numpy()),"ward":_codes(s.ward.to_numpy())}
        for ctrl in ("none","ty","full"):
            if ctrl=="none": fes=[]; l=None
            elif ctrl=="ty": fes=[ty]; l=None
            else: fes=[ty,s.ym.to_numpy(),s.dow.to_numpy(),s.hod.to_numpy(),s.chan.to_numpy()]; l=lwl
            for rec in fit_cell(Y,Mz,l,fes,groups,list(Y),MEAS):
                rec.update({"records":rname,"window":wname,"controls":ctrl,"sd_y":sdy[rec["outcome"]],"mean_y":float(Y[rec["outcome"]].mean())}); out.append(rec)
        savej("M1_grid",out); print(f"{rname:36s} | {wname:17s} n={len(s):>9,} specs={len(out)} [{time.time()-t0:.0f}s]",flush=True)
        del s,h,hw,ty,g,rank,med,Y,Mz,lwl,groups; gc.collect()
    del Ar; gc.collect()
print("complete:",len(out),"specifications",flush=True)
