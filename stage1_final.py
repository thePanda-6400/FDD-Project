
import os, warnings, numpy as np, pandas as pd
MAT_PATH, OUT, FS = "Propeller_fault_data.mat", "stage1_final", 312497.5
NEW={"CV":{0,10,11},"CCV":{0,6,7}}
PP={("CV",0):1.0,("CV",10):1.0,("CV",11):1.0,("CV",1):.90,("CV",2):.83,("CV",3):.81,
 ("CV",4):.85,("CV",5):.90,("CV",6):.86,("CV",7):.90,("CV",8):.86,("CV",9):.83,
 ("CCV",0):1.0,("CCV",6):1.0,("CCV",7):1.0,("CCV",1):.86,("CCV",2):.82,("CCV",3):.80,
 ("CCV",4):.91,("CCV",5):.86}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import kurtosis, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, roc_curve

def load(p):
    from pymatreader import read_mat
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); d=read_mat(p)
    return {k:v for k,v in d.items() if not k.startswith("__")}
def ix(s): return int("".join(c for c in s if c.isdigit()))
def f0(x):
    x=np.squeeze(np.asarray(x,float)); x=x[np.isfinite(x)]-np.mean(x)
    X=np.abs(np.fft.rfft(x*np.hanning(x.size))); fr=np.fft.rfftfreq(x.size,1/FS)
    X[fr<=5]=0; return fr[np.argmax(X)]

def feats(ipt):
    f={}; rms=[]
    for p in ("MeasCurrA","MeasCurrB","MeasCurrC"):
        if p not in ipt: continue
        x=np.squeeze(np.asarray(ipt[p],float)); x=x[np.isfinite(x)]
        r=np.sqrt(np.mean(x**2)); rms.append(r)
        f[f"{p}_rms"]=r; f[f"{p}_kurt"]=float(kurtosis(x))
        # 5th-harmonic (normalised) for completeness
        xm=x-x.mean(); X=np.abs(np.fft.rfft(xm*np.hanning(xm.size))); fr=np.fft.rfftfreq(xm.size,1/FS)
        X[fr<=5]=0; j=np.argmax(X); d=fr[1]-fr[0]; fund=X[max(j-2,0):j+3].max()+1e-9
        k5=int(round(5*fr[j]/d)); f[f"{p}_h5"]=(X[max(k5-2,0):k5+3].max() if k5<len(X) else 0)/fund
    if len(rms)==3: f["imbalance"]=np.std(rms)/(np.mean(rms)+1e-9)
    f["spd"]=f0(ipt["MeasCurrA"])
    pw=0.0
    for c,v in (("MeasCurrA","MeasVoltA"),("MeasCurrB","MeasVoltB"),("MeasCurrC","MeasVoltC")):
        if c in ipt and v in ipt:
            I=np.squeeze(np.asarray(ipt[c],float)); V=np.squeeze(np.asarray(ipt[v],float))
            n=min(I.size,V.size); pw+=np.mean(I[:n]*V[:n])
    f["power"]=pw
    return f

def main():
    os.makedirs(OUT,exist_ok=True); data=load(MAT_PATH); rows=[]
    for g in ("CV","CCV"):
        if g not in data: continue
        for rn,run in data[g].items():
            if rn.lower()=="omega" or not isinstance(run,dict): continue
            pid=ix(rn)
            for ik,iv in run.items():
                if not isinstance(iv,dict) or "MeasCurrA" not in iv: continue
                r={"dir":g,"prop":pid,"grp":f"{g}{pid}","input":ik,"pwm":ix(ik)*10,
                   "is_new":int(pid in NEW[g]),"PP":PP.get((g,pid),np.nan)}
                r.update(feats(iv)); rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(f"{OUT}/dataset.csv",index=False)
    F=[c for c in df.columns if c not in ("dir","prop","grp","input","pwm","is_new","PP")]
    F=[c for c in F if df[c].notna().all()]
    logo=LeaveOneGroupOut(); grp=df["grp"]; log=[f"rows={len(df)} feats={len(F)}"]

    # 1) model comparison (LOPO) : RF vs logistic
    rf=RandomForestClassifier(400,random_state=0)
    lr=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    for name,m in (("RF",rf),("LogReg",lr)):
        p=cross_val_predict(m,df[F],df["is_new"],cv=logo,groups=grp,method="predict_proba")[:,1]
        log.append(f"[{name}] new-vs-worn AUC={roc_auc_score(df['is_new'],p):.3f}")
        if name=="RF": rf_p=p

    # 2) feature importance (fit on all; indicative)
    rf.fit(df[F],df["is_new"])
    imp=pd.Series(rf.feature_importances_,index=F).sort_values(ascending=False)
    log.append("top features: "+", ".join(f"{k}={v:.2f}" for k,v in imp.head(5).items()))

    # 3) per-input AUC
    perinp={}
    for inp,sub in df.groupby("input"):
        if sub["is_new"].nunique()<2: continue
        p=cross_val_predict(rf,sub[F],sub["is_new"],cv=logo,groups=sub["grp"],method="predict_proba")[:,1]
        perinp[sub["pwm"].iloc[0]]=roc_auc_score(sub["is_new"],p)

    # 4) severity regression (best effort)
    m=df["PP"].notna()
    pr=cross_val_predict(RandomForestRegressor(400,random_state=0),
                         df.loc[m,F],df.loc[m,"PP"],cv=logo,groups=df.loc[m,"grp"])
    log.append(f"[severity] PP R2={r2_score(df.loc[m,'PP'],pr):.3f} "
               f"MAE={mean_absolute_error(df.loc[m,'PP'],pr):.3f} "
               f"rho={spearmanr(df.loc[m,'PP'],pr).statistic:+.3f}")

    # plots
    fpr,tpr,_=roc_curve(df["is_new"],rf_p)
    plt.figure(figsize=(4.5,4.5)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"k--",lw=.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC new-vs-worn (LOPO)")
    plt.tight_layout(); plt.savefig(f"{OUT}/roc.png",dpi=120); plt.close()
    imp.head(10)[::-1].plot.barh(figsize=(6,4)); plt.title("Feature importance")
    plt.tight_layout(); plt.savefig(f"{OUT}/importance.png",dpi=120); plt.close()
    pd.Series(perinp).sort_index().plot.bar(figsize=(6,3.5)); plt.ylabel("AUC"); plt.xlabel("PWM")
    plt.title("Per-input detection AUC"); plt.tight_layout(); plt.savefig(f"{OUT}/per_input_auc.png",dpi=120)

    open(f"{OUT}/results.txt","w").write("\n".join(log)); print("\n".join(log)); print(f"-> {OUT}/")

if __name__=="__main__": main()