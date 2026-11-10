# Community-area measures rebuilt from the raw public files; verified against the main paper's Table 2 and Table 7 correlations.
import pandas as pd, numpy as np, geopandas as gpd, glob, json
from shapely import wkt
U="raw_data/"
cen=pd.read_csv(U+"Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012_20260923.csv")
cen.columns=[c.strip() for c in cen.columns]
cen=cen.rename(columns={"Community Area Number":"ca","HARDSHIP INDEX":"hardship"})[["ca","hardship"]]
cen["ca"]=pd.to_numeric(cen.ca,errors="coerce"); cen=cen.dropna(subset=["ca"]); cen["ca"]=cen.ca.astype(int)
acs=pd.read_csv(glob.glob(U+"ACS_5_Year_Data_by_Community_Area_*.csv")[0]); print("ACS years:",sorted(acs["ACS Year"].unique()))
acs=acs[acs["ACS Year"]==acs["ACS Year"].max()].copy(); _bd=pd.read_csv(U+"Boundaries_-_Community_Areas_20260923.csv"); _nm=dict(zip(_bd.COMMUNITY.str.strip().str.upper(),_bd.AREA_NUMBE.astype(int)))
acs["ca"]=acs["Community Area"].astype(str).str.strip().str.upper().map(_nm); print("ACS names unmatched:",int(acs.ca.isna().sum())); acs=acs.dropna(subset=["ca"]); acs["ca"]=acs.ca.astype(int)
for c in acs.columns:
    if c not in ("ACS Year","Record ID"): acs[c]=pd.to_numeric(acs[c].astype(str).str.replace(",","").str.strip(),errors="coerce")
brk=["Under $25,000","$25,000 to $49,999","$50,000 to $74,999","$75,000 to $125,000","$125,000 +"]
acs["low_inc"]=100*acs["Under $25,000"]/acs[brk].sum(axis=1)
acs["pct_black"]=100*acs["Black or African American"]/acs["Total Population"]; acs["pct_hisp"]=100*acs["Hispanic or Latino"]/acs["Total Population"]; print("ACS CA rows:",len(acs),"| NaN:",int(acs[["low_inc","pct_black","Total Population"]].isna().sum().sum()))
acs["pop"]=acs["Total Population"]; acs=acs[["ca","pop","low_inc","pct_black","pct_hisp"]]
# violent Index crimes, union of the three uploaded extracts (deduplicated on ID), 2019-2024
cr=[]
for f in sorted(glob.glob(U+"Crimes_-_2001_to_Present_*.csv")):
    d=pd.read_csv(f,usecols=["ID","Date","FBI Code","Community Area"],dtype={"FBI Code":str},low_memory=False); cr.append(d)
cr=pd.concat(cr).drop_duplicates("ID"); cr["dt"]=pd.to_datetime(cr.Date,format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
vio=cr[cr["FBI Code"].isin(["01A","02","03","04A","04B"])&cr.dt.dt.year.between(2019,2024)&cr["Community Area"].between(1,77)]
vio=vio.assign(ca=vio["Community Area"].astype(int)); vc=vio.groupby("ca").size().rename("viol_n"); print("unique crime IDs:",len(cr),"| violent in 2019-2024 within CAs:",len(vio))
# building violations geocoded to community areas (point-in-polygon), 2019-2024
bd=[]
for f in sorted(glob.glob(U+"Building_Violations_*.csv")):
    d=pd.read_csv(f,usecols=["ID","VIOLATION DATE","LATITUDE","LONGITUDE"],low_memory=False); bd.append(d)
bd=pd.concat(bd).drop_duplicates("ID"); bd["dt"]=pd.to_datetime(bd["VIOLATION DATE"],errors="coerce")
bd=bd[bd.dt.dt.year.between(2019,2024)&bd.LATITUDE.notna()]
ca=pd.read_csv(U+"Boundaries_-_Community_Areas_20260923.csv"); cap=gpd.GeoDataFrame({"ca":ca.AREA_NUMBE.astype(int)},geometry=ca.the_geom.apply(wkt.loads),crs="EPSG:4326")
pts=gpd.GeoDataFrame(bd[["ID"]],geometry=gpd.points_from_xy(bd.LONGITUDE,bd.LATITUDE),crs="EPSG:4326")
j=gpd.sjoin(pts,cap,how="inner",predicate="within"); bc=j.groupby("ca").size().rename("bv_n"); print("building violations 2019-2024 geocoded to CAs:",len(j),"of",len(bd))
print("keys: cen",cen.ca.dtype,len(cen),"acs",acs.ca.dtype,len(acs),"| viol keys",len(vc),"| bv keys",len(bc))
M=cen.merge(acs,on="ca").merge(vc,left_on="ca",right_index=True,how="left").merge(bc,left_on="ca",right_index=True,how="left")
M["viol_rate"]=1000*M.viol_n/(6*M["pop"]); M["bv_rate"]=1000*M.bv_n/(6*M["pop"])
M=M.sort_values("ca").reset_index(drop=True); M.to_csv("work/ca_measures_v2.csv",index=False)
print("CAs:",len(M)); print(M[["hardship","low_inc","pct_black","viol_rate","bv_rate"]].describe().round(2).to_string())
q=pd.qcut(M.hardship,4,labels=[1,2,3,4])
print("Table 2 check (paper: viol Q1 4.9, Q4 22.1, all 11.7; bv Q1 12.5, Q4 37.4, all 24.2; hardship mean 49.5 sd 28.7):")
print(" viol by quartile:",M.groupby(q,observed=True).viol_rate.mean().round(1).tolist(),"all",round(M.viol_rate.mean(),1))
print(" bv by quartile:",M.groupby(q,observed=True).bv_rate.mean().round(1).tolist(),"all",round(M.bv_rate.mean(),1))
print(" hardship mean/sd:",round(M.hardship.mean(),1),round(M.hardship.std(),1))
print(" corr hardship-pctBlack %.2f (paper 0.47) | lowinc-pctBlack %.2f (paper 0.74) | hardship-lowinc %.2f (paper 0.74) | hardship-violrate %.2f"%(M.hardship.corr(M.pct_black),M.low_inc.corr(M.pct_black),M.hardship.corr(M.low_inc),M.hardship.corr(M.viol_rate)))
M.to_csv("data/ca_measures.csv",index=False); print("saved ca_measures.csv (replaces the earlier file)")
