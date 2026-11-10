# Summarize the multiverse: standardized estimates (coef / SD of outcome), significance, and the contribution of each analytical choice.
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
R=pd.DataFrame(json.load(open("results/M1_grid.json")))
R["std"]=R.coef/R.sd_y; R["sig_ca"]=(R.p_ca<0.05); R["sig_hc1"]=(R.p_hc1<0.05)
R["mod"]=R.measure.map({"hardship":"Hardship Index (2008-12)","low_inc":"Share of households under $25k (ACS)","pct_black":"Share Black residents (ACS)"})
print("specifications:",len(R))
S={"n_specs":int(len(R)),"n_records":int(R.records.nunique()),"n_windows":int(R.window.nunique()),"n_controls":int(R.controls.nunique()),"n_outcomes":int(R.outcome.nunique()),"n_measures":int(R.measure.nunique())}
H=R[R.measure=="hardship"]
S["hardship_specs"]=int(len(H)); S["hardship_std_min"]=float(H["std"].min()); S["hardship_std_max"]=float(H["std"].max()); S["hardship_std_median"]=float(H["std"].median())
S["hardship_share_positive"]=float((H["std"]>0).mean()); S["hardship_share_sig_ca"]=float(H.sig_ca.mean()); S["hardship_share_sig_hc1"]=float(H.sig_hc1.mean())
S["hardship_share_pos_sig"]=float(((H["std"]>0)&H.sig_ca).mean())
S["all_share_positive"]=float((R["std"]>0).mean()); S["all_share_sig_ca"]=float(R.sig_ca.mean())
S["low_inc_share_pos_sig"]=float((R[R.measure=="low_inc"]["std"].gt(0)&R[R.measure=="low_inc"].sig_ca).mean())
S["pct_black_share_pos_sig"]=float((R[R.measure=="pct_black"]["std"].gt(0)&R[R.measure=="pct_black"].sig_ca).mean())
# marginal contribution of each choice: mean standardized estimate by level, hardship measure only, relative to the baseline level
base={"records":"Resident, non-duplicate (baseline)","window":"2019-2024","controls":"none","outcome":"log"}
CH={}
for ch in ["records","window","controls","outcome"]:
    g=H.groupby(ch)["std"].agg(["mean","min","max","size"]); CH[ch]={k:{"mean":float(v["mean"]),"min":float(v["min"]),"max":float(v["max"]),"n":int(v["size"])} for k,v in g.iterrows()}
# regression of the standardized estimate on choice dummies (hardship specs): the spread attributable to each choice
import statsmodels.formula.api as smf
X=H.copy(); X["records"]=X.records.astype("category"); X["window"]=X.window.astype("category"); X["controls"]=X.controls.astype("category"); X["outcome"]=X.outcome.astype("category")
m=smf.ols("std ~ C(records)+C(window)+C(controls)+C(outcome)",data=X).fit(); S["r2_choices_hardship"]=float(m.rsquared)
S["choice_r2_by_dim"]={}
for ch in ["records","window","controls","outcome"]:
    mm=smf.ols(f"std ~ C({ch})",data=X).fit(); S["choice_r2_by_dim"][ch]=float(mm.rsquared)
S["choice_means_hardship"]=CH
# which records/windows drive the sign-reversal or non-significance
S["nonsig_hardship"]=H[~H.sig_ca][["records","window","controls","outcome"]].astype(str).value_counts().head(10).reset_index().values.tolist()
S["hardship_specs_nonpositive"]=int((H["std"]<=0).sum())
json.dump(S,open("results/S2_multiverse_summary.json","w"),indent=1,default=float)
print(json.dumps({k:v for k,v in S.items() if k not in("choice_means_hardship","nonsig_hardship")},indent=1,default=float)[:2500])
for ch,d in CH.items(): print(ch,{k:round(v["mean"],3) for k,v in d.items()})
# specification curve, hardship measure
Hs=H.sort_values("std").reset_index(drop=True); fig,(a1,a2)=plt.subplots(2,1,figsize=(6.5,5.2),sharex=True,gridspec_kw={"height_ratios":[2.2,1]})
col=np.where(Hs.sig_ca,"#c0392b","#999999"); a1.scatter(range(len(Hs)),Hs["std"],s=6,c=col); a1.axhline(0,color="#444",lw=.7); a1.set_ylabel("Standardized hardship gradient\n(coef / SD of outcome)")
a1.set_title(f"Hardship Index specifications (n={len(Hs)}); red = significant at 5%",fontsize=8.5)
rows={"records":sorted(H.records.unique()),"window":sorted(H.window.unique()),"controls":["none","ty","full"],"outcome":["log","rank","slower","days"]}
ys=[]; labs=[]
for ch,levels in rows.items():
    for lv in levels: ys.append(((Hs[ch]==lv).astype(float)).values); labs.append(f"{ch}: {lv}")
M=np.vstack(ys); a2.imshow(M,aspect="auto",cmap="Greys",interpolation="nearest"); a2.set_yticks(range(len(labs))); a2.set_yticklabels(labs,fontsize=5.5); a2.set_xlabel("Specifications, sorted by estimate")
fig.tight_layout(); fig.savefig("figs/fig_specurve.png",dpi=220); plt.close()
# choice contributions
fig,ax=plt.subplots(figsize=(6.2,3.2)); yl=[]; i=0
for ch in ["records","window","controls","outcome"]:
    for lv,v in CH[ch].items():
        ax.plot([v["min"],v["max"]],[i,i],color="#bbbbbb",lw=3); ax.plot(v["mean"],i,"o",color="#c0392b",ms=4); yl.append(f"{ch}: {lv}"); i+=1
ax.axvline(0,color="#444",lw=.7); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl,fontsize=6.5); ax.invert_yaxis(); ax.set_xlabel("Standardized hardship gradient: mean (dot) and range (bar) across other choices")
fig.tight_layout(); fig.savefig("figs/fig_choices.png",dpi=220); plt.close(); print("figures written")
