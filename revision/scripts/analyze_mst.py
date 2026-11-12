import pandas as pd, numpy as np, glob
from scipy import stats
fs=[f for f in ["outputs/mst_main_a_gp.csv","outputs/mst_main_a3_compas.csv","outputs/mst_main_a3.csv","outputs/mst_main_a4.csv","outputs/mst_main_b.csv"] if glob.glob(f)]
d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).drop_duplicates(["dataset","seed","variant"],keep="first")
# keep Anonymeter-bearing compas seed0 from a2 only if a3 lacks it
d["anon_any"]=(d[["anon_singling_lo","anon_link_lo","anon_infer_lo"]].fillna(0)>0).any(axis=1) if "anon_singling_lo" in d else np.nan
if "NOANON" and d[["anon_singling_lo"]].isna().all(axis=1).any(): d.loc[d.anon_singling_lo.isna(),"anon_any"]=np.nan
g=d.groupby(["dataset","variant"]).agg(n=("seed","count"),util=("util_ratio","mean"),auc=("mia_auc_max","mean"),auc_sd=("mia_auc_max","std"),tpr=("mia_tpr1_rand_max","mean"),
   null95=("mia_tpr1_rand_max_null_p95","mean"),anon=("anon_any","mean")).round(3)
print(g.to_string())
# C6: protected-group loss minus reference-group loss
rows=[]
for _,r in d.iterrows():
    for a in ["sex","race","age<=25","age<30","age>=60"]:
        if f"{a}|auc_unpriv" in r and pd.notna(r.get(f"{a}|auc_unpriv")):
            rows.append(dict(dataset=r.dataset,attr=a,seed=r.seed,variant=r.variant,
              drop_prot=r[f"{a}|auc_unpriv_real"]-r[f"{a}|auc_unpriv"], drop_ref=r[f"{a}|auc_priv_real"]-r[f"{a}|auc_priv"]))
C=pd.DataFrame(rows); C["diff"]=C.drop_prot-C.drop_ref
P1=C[C.dataset.isin(["Adult","German","Pima"])]
print("\nC6 on Phase I datasets with protected groups (protected loss minus reference loss; >0 supports C6)")
for v,s in P1.groupby("variant"):
    w=stats.wilcoxon(s["diff"],alternative="greater") if len(s)>5 else None
    print(v,"n=",len(s),"mean diff=%.3f"%s["diff"].mean(),"p(one-sided)=%.3f"%(w.pvalue if w else np.nan),"prot loss=%.3f ref loss=%.3f"%(s.drop_prot.mean(),s.drop_ref.mean()))
print(P1.groupby(["dataset","attr","variant"])[["drop_prot","drop_ref","diff"]].mean().round(3).to_string())
g.to_csv("outputs/mst_main_summary.csv"); C.to_csv("outputs/mst_c6_rows.csv",index=False)
