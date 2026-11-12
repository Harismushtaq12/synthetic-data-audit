"""R8: dataset as the inferential unit. Per dataset: one-sided t-test over seeds of the strongest-attack AUC against the dataset's own
permutation-null mean for the same statistic (max of three attacks), then Fisher's method across the four datasets (Phase I and held-out separately), Holm over mechanisms."""
import numpy as np, pandas as pd, json
from scipy import stats
N = pd.read_csv("outputs/null_controls_all.csv").groupby("dataset").null_auc_mean_max.mean().to_dict()
SYN = ["Independent","GaussianCopula","CART","CTGAN","TVAE","DDPM","MST (ε=1)"]
A = {"Adult":[(.506,.007),(.506,.013),(.592,.006),(.510,.003),(.515,.011),(.511,.017),(.506,.008)],
     "German":[(.534,.008),(.539,.009),(.621,.021),(.526,.015),(.517,.021),(.548,.016),(.523,.020)],
     "Fraud":[(.513,.020),(.509,.004),(.559,.013),(.509,.009),(.515,.005),(.535,.019),(.505,.003)],
     "Pima":[(.553,.022),(.520,.024),(.629,.027),(.512,.028),(.541,.021),(.712,.034),(.524,.024)]}
NS = {"Adult":3,"German":5,"Fraud":3,"Pima":5}
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]); adj[j]=min(1,run)
    return adj
rows=[]
for i,g in enumerate(SYN):
    ps=[]; k=0
    for d in A:
        mean,ci=A[d][i]; n=NS[d]; se=ci/stats.t.ppf(.975,n-1); t=(mean-N[d])/se; p=stats.t.sf(t,n-1); ps.append(p); k+=(p<0.05)
    chi=-2*np.sum(np.log(np.maximum(ps,1e-300))); pf=stats.chi2.sf(chi,2*len(ps))
    rows.append(dict(phase="I",mechanism=g,datasets_sig_05=k,fisher_p=pf))
P1=pd.DataFrame(rows); P1["holm_p"]=holm(P1.fisher_p.values)
H=pd.read_csv("../notebooks/phase2/outputs2/tables/X5_results.csv"); H=H[H.dataset.isin(["COMPAS","Credit","Bank","Diabetes130"])]
rows=[]
for g in ["Independent","GaussianCopula","CART","CTGAN","TVAE","DDPM","MST (ε=1)","AR-Transformer"]:
    ps=[];k=0
    for d,s in H[H.generator==g].groupby("dataset"):
        x=s.mia_auc_max.values; t=(x.mean()-N[d])/(x.std(ddof=1)/np.sqrt(len(x))); p=stats.t.sf(t,len(x)-1); ps.append(p); k+=(p<0.05)
    chi=-2*np.sum(np.log(np.maximum(ps,1e-300))); rows.append(dict(phase="II",mechanism=g,datasets_sig_05=k,fisher_p=stats.chi2.sf(chi,2*len(ps))))
P2=pd.DataFrame(rows); P2["holm_p"]=holm(P2.fisher_p.values)
R=pd.concat([P1,P2]); R.to_csv("outputs/R8_dataset_level_calibrated.csv",index=False); print(R.round(5).to_string()); print(N)
