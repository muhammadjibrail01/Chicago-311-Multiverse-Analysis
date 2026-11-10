# Extra analyses for the paper: monthly figure, records-by-controls grid, snow-type seasonal composition.
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
S=json.load(open("results/S1_season.json")); bm=S["By month"]
mon=np.array([bm[f"H_m{m}"][0] for m in range(1,13)]); se=np.array([bm[f"H_m{m}"][1] for m in range(1,13)])
fig,ax=plt.subplots(figsize=(6.2,2.8)); x=np.arange(1,13)
ax.axvspan(0.5,2.5,color="#dde8f3",alpha=.8,label="Winter (Dec–Feb)"); ax.axvspan(11.5,12.5,color="#dde8f3",alpha=.8)
ax.errorbar(x,mon,yerr=1.96*se,fmt="o-",color="#c0392b",ms=4,lw=1,capsize=2.5)
ax.set_xticks(x); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]); ax.set_ylabel("Hardship gradient\n(log points per SD)")
ax.axhline(0,color="#999",lw=.7); ax.legend(frameon=False,fontsize=7.5,loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig_monthly.png",dpi=220); plt.close()
R=pd.DataFrame(json.load(open("results/M1_grid.json"))); R["std"]=R.coef/R.sd_y; H=R[(R.measure=="hardship")&(R.window=="2019-2024")&(R.outcome=="log")]
grid=H.pivot_table(index="records",columns="controls",values="std")[["none","ty","full"]]
grid.round(3).to_csv("results/A2_records_by_controls.csv"); print(grid.round(3).to_string())
# snow-related types: share of requests by season, and share in high-hardship communities
A=pd.read_parquet("data/p3_all.parquet",columns=["type","month","ca","info","chan","nd","done"])
A=A[(~A["info"])&(A.chan!=3)&A.nd]
M=pd.read_csv("data/ca_measures.csv"); hi=set(M[M.hardship>=M.hardship.median()].ca)
snow=["Ice and Snow Removal Request","Snow - Object/Dibs Removal Request","Snow - Uncleared Sidewalk Complaint"]
A["season"]=A.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"})
A["snow"]=A.type.astype(str).isin(snow); A["hi"]=A.ca.isin(hi)
comp=A.groupby("season").agg(n=("type","size"),snow_share=("snow","mean"),hi_share=("hi","mean")).reindex(["Winter","Spring","Summer","Fall"])
comp["snow_share"]=comp.snow_share*100; comp["hi_share"]=comp.hi_share*100
print(comp.round(2).to_string()); comp.round(2).to_csv("results/A3_season_composition.csv")
hs=A[A.snow].groupby("season").hi.mean()*100; print("snow requests in high-hardship CAs by season:",hs.round(1).to_dict())
