#!/usr/bin/env python3

import os, warnings, numpy as np, pandas as pd
MAT_PATH, OUT, FS = "Propeller_fault_data.mat", "stage1b_out", 312497.5
NEW = {"CV":{0,10,11}, "CCV":{0,6,7}}
PP = {("CV",0):1.0,("CV",10):1.0,("CV",11):1.0,("CV",1):.90,("CV",2):.83,("CV",3):.81,
 ("CV",4):.85,("CV",5):.90,("CV",6):.86,("CV",7):.90,("CV",8):.86,("CV",9):.83,
 ("CCV",0):1.0,("CCV",6):1.0,("CCV",7):1.0,("CCV",1):.86,("CCV",2):.82,("CCV",3):.80,
 ("CCV",4):.91,("CCV",5):.86}
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

def load(p):
    from pymatreader import read_mat
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); d=read_mat(p)
    return {k:v for k,v in d.items() if not k.startswith("__")}
def idx(s): return int("".join(c for c in s if c.isdigit()))

def scan_force(node, depth=0, hits=None):
    """recursively report any force-like leaf + its path (diagnostic)."""
    if hits is None: hits=[]
    if isinstance(node, dict):
        for k,v in node.items():
            if "force" in k.lower() and not isinstance(v,dict):
                a=np.squeeze(np.asarray(v,float)); hits.append((k, getattr(a,'shape',None)))
            elif isinstance(v,dict) and depth<4: scan_force(v,depth+1,hits)
    return hits

def f0_of(x, fs):
    x=x-x.mean(); X=np.abs(np.fft.rfft(x*np.hanning(x.size))); fr=np.fft.rfftfreq(x.size,1/fs)
    X[fr<=5]=0; return fr[np.argmax(X)]

def speed_feats(x, fs, nwin=8):
    """fundamental freq in sliding windows -> mean + variance (speed cue)."""
    x=np.asarray(x,float); x=x[np.isfinite(x)]; x=x[len(x)//8:]  # drop startup
    w=x.size//nwin; f0s=[f0_of(x[i*w:(i+1)*w],fs) for i in range(nwin) if (i+1)*w<=x.size]
    f0s=np.array(f0s); return f0s.mean(), f0s.std(), f0s.std()/(f0s.mean()+1e-9)

def cfeat(ipt):
    f={}; rmss=[]
    for p in ("MeasCurrA","MeasCurrB","MeasCurrC"):
        if p not in ipt: continue
        x=np.squeeze(np.asarray(ipt[p],float)); x=x[np.isfinite(x)]
        rms=np.sqrt(np.mean(x**2)); rmss.append(rms); f[f"{p}_rms"]=rms
    x=np.squeeze(np.asarray(ipt["MeasCurrA"],float))
    m,s,cv=speed_feats(x,FS); f["spd_mean"]=m; f["spd_std"]=s; f["spd_cv"]=cv
    if len(rmss)==3: f["imbalance"]=np.std(rmss)/(np.mean(rmss)+1e-9)
    pw=0.0
    for c,v in (("MeasCurrA","MeasVoltA"),("MeasCurrB","MeasVoltB"),("MeasCurrC","MeasVoltC")):
        if c in ipt and v in ipt:
            I=np.squeeze(np.asarray(ipt[c],float)); V=np.squeeze(np.asarray(ipt[v],float))
            n=min(I.size,V.size); pw+=np.mean(I[:n]*V[:n])
    f["elec_power"]=pw
    return f

def main():
    os.makedirs(OUT,exist_ok=True); data=load(MAT_PATH); rows=[]; force_paths=set()
    for g in ("CV","CCV"):
        if g not in data: continue
        # diagnostic: where is force?
        for k in scan_force(data[g]): force_paths.add(k[0])
        for rn,run in data[g].items():
            if rn.lower()=="omega" or not isinstance(run,dict): continue
            pid=idx(rn)
            for iname,ipt in run.items():
                if not isinstance(ipt,dict) or "MeasCurrA" not in ipt: continue
                r={"dir":g,"prop":pid,"grp":f"{g}{pid}","input":iname,"pwm":idx(iname)*10,
                   "is_new":int(pid in NEW[g]),"PP":PP.get((g,pid),np.nan)}
                r.update(cfeat(ipt)); rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(f"{OUT}/dataset.csv",index=False)
    log=[f"rows={len(df)}  force-like keys found in tree: {sorted(force_paths) or 'NONE'}"]
    feats=[c for c in df.columns if c not in ("dir","prop","grp","input","pwm","is_new","PP")]
    feats=[c for c in feats if df[c].notna().all()]
    logo=LeaveOneGroupOut()

    # overall A/B with new speed features
    clf=RandomForestClassifier(300,random_state=0); reg=RandomForestRegressor(300,random_state=0)
    pc=cross_val_predict(clf,df[feats],df["is_new"],cv=logo,groups=df["grp"],method="predict_proba")[:,1]
    log.append(f"[A] new-vs-worn AUC={roc_auc_score(df['is_new'],pc):.3f}")
    m=df["PP"].notna()
    pr=cross_val_predict(reg,df.loc[m,feats],df.loc[m,"PP"],cv=logo,groups=df.loc[m,"grp"])
    log.append(f"[B] PP R2={r2_score(df.loc[m,'PP'],pr):.3f} MAE={mean_absolute_error(df.loc[m,'PP'],pr):.3f} rho={spearmanr(df.loc[m,'PP'],pr).statistic:+.3f}")

    # single-feature check: does speed variance alone track PP?
    for fcol in ("spd_std","spd_cv","spd_mean"):
        s=df.loc[m,[fcol,"PP"]].dropna()
        log.append(f"    {fcol:9s} vs PP rho={spearmanr(s[fcol],s['PP']).statistic:+.3f}")

    # per-input AUC (effect is input-specific)
    log.append("[C] per-input new-vs-worn AUC:")
    for inp,sub in df.groupby("input"):
        if sub["is_new"].nunique()<2: continue
        try:
            p=cross_val_predict(clf,sub[feats],sub["is_new"],
                                cv=logo,groups=sub["grp"],method="predict_proba")[:,1]
            log.append(f"    {inp} (pwm {sub['pwm'].iloc[0]}): AUC={roc_auc_score(sub['is_new'],p):.3f}")
        except Exception as e: log.append(f"    {inp}: {e}")

    open(f"{OUT}/results.txt","w").write("\n".join(log)); print("\n".join(log)); print(f"-> {OUT}/")

if __name__=="__main__": main()