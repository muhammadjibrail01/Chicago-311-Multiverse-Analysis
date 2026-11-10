# All-records analysis frame for the multiverse: core requests (incl. duplicates and city-generated) + information-only calls
import pandas as pd, numpy as np, glob, time
t0=time.time()
raw=pd.read_parquet("work/core_raw.parquet",columns=["SR_NUMBER","SR_TYPE","STATUS","ORIGIN","CREATED_DATE","CLOSED_DATE","DUPLICATE","COMMUNITY_AREA","WARD"])
core=pd.read_parquet("data/core_request_file.parquet",columns=["sr","chan","speed_s","hours","nd"])
assert (raw.SR_NUMBER.values==core.sr.values).all()
print("STATUS x DUPLICATE among core rows (counts):"); print(pd.crosstab(raw.STATUS,raw.DUPLICATE.astype(str)).to_string(),flush=True)
tract=np.full(len(raw),-1,dtype="int32")
def frame(df,chan,nd,tract):
    c=df.CREATED_DATE; hrs=(df.CLOSED_DATE-df.CREATED_DATE).dt.total_seconds()/3600
    done=(df.STATUS=="Completed")&df.CLOSED_DATE.notna()&(df.CLOSED_DATE>=c)&c.notna()
    return pd.DataFrame({"type":df.SR_TYPE.astype(str).values,"year":c.dt.year.astype("int16").values,"ym":((c.dt.year-2019)*12+c.dt.month-1).astype("int16").values,"month":c.dt.month.astype("int8").values,
      "dow":c.dt.dayofweek.astype("int8").values,"hod":c.dt.hour.astype("int8").values,"week":((c-pd.Timestamp("2019-01-01")).dt.days//7).astype("int16").values,"chan":chan,"nd":nd,"done":done.values,
      "hours":hrs.where(done).astype("float32").values,"ca":pd.to_numeric(df.COMMUNITY_AREA,errors="coerce").values,"ward":pd.to_numeric(df.WARD,errors="coerce").fillna(0).astype("int16").values,"tract":tract})
A=frame(raw,core.chan.values,(raw.DUPLICATE.astype(str)=="false").values,tract); A["info"]=False
# consistency with the main package: hours of the speed sample
sp=core.speed_s.values; print("hours match the main package on the speed sample:",bool(np.allclose(A.hours.values[sp],core.hours.values[sp],equal_nan=True)),f"[{time.time()-t0:.0f}s]",flush=True)
del raw,core
# information-only calls with a valid community area
cols=["SR_NUMBER","SR_TYPE","STATUS","ORIGIN","CREATED_DATE","CLOSED_DATE","DUPLICATE","COMMUNITY_AREA","WARD"]; parts=[]
city={"Mass Entry","Generated In House","DWM","City Department","WOFromTerraGo","Salesforce Mobile App","SPOTCSR","CSCC","XTERFACE","Chicago Police Department","Chicago Fire Department","IMMTOWQA","RMS","CreateWorkOrder","Radio","HealthProfessionals"}
phone={"Phone Call","Phone","Walk-in","Mail","Fax"}; ald={"Alderman's Office","ALDERMAN","Budget Town Hall Meeting","FY25 Budget Engagement","Mayor's Office"}
for f in sorted(glob.glob("raw_data/raw311/*.csv")):
    for ch in pd.read_csv(f,usecols=cols,chunksize=600_000,dtype=str):
        ch=ch[ch.SR_TYPE=="311 INFORMATION ONLY CALL"]
        ca=pd.to_numeric(ch.COMMUNITY_AREA,errors="coerce"); ch=ch[ca.between(1,77)].copy()
        if len(ch)==0: continue
        for c in ("CREATED_DATE","CLOSED_DATE"): ch[c]=pd.to_datetime(ch[c],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
        o=ch.ORIGIN.astype(str); chn=np.select([o.isin(city),o.isin(phone),o.isin(ald)],[3,0,2],default=1).astype("int8")
        F=frame(ch,chn,(ch.DUPLICATE.astype(str)=="false").values,np.full(len(ch),-1,dtype="int32")); F["info"]=True; parts.append(F)
    print("  info-only rows so far:",sum(len(p) for p in parts),f"[{time.time()-t0:.0f}s]",flush=True)
A["chan"]=A.chan.map({"phone":0,"digital":1,"alderman":2,"city":3}).astype("int8")
I=pd.concat(parts,ignore_index=True); A=pd.concat([A,I],ignore_index=True); del I,parts
A["type"]=A.type.astype("category"); A["tcode"]=A.type.cat.codes.astype("int16"); A["ca"]=A.ca.astype("int16")
# workload: resident non-duplicate requests of the same type in the same community area and week (main paper definition), floor of 1
res_nd=A.nd&(A.chan!=3); g=A[res_nd].groupby(["tcode","ca","week"],observed=True).size().rename("wl"); A=A.join(g,on=["tcode","ca","week"]); A["lwl"]=np.log(A.wl.fillna(1).clip(lower=1)).astype("float32"); del A["wl"]
print("rows:",f"{len(A):,}","| core:",int((~A['info']).sum()),"| info-only:",int(A['info'].sum()),"| completed:",f"{int(A.done.sum()):,}",f"[{time.time()-t0:.0f}s]")
A.to_parquet("p3_all.parquet",index=False)
