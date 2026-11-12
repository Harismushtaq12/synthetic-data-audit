import pandas as pd, numpy as np, glob, os
from scipy import stats
OUT="tables_tex/"; os.makedirs(OUT,exist_ok=True)
def L(x): return x.replace("%","\\%").replace("_","\\_")
# ---------- MST main
fs=[f for f in ["outputs/mst_main_a_gp.csv","outputs/mst_main_a3_compas.csv","outputs/mst_main_a3.csv","outputs/mst_main_a4.csv","outputs/mst_main_b.csv"] if glob.glob(f)]
d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).drop_duplicates(["dataset","seed","variant"],keep="first")
d["anon_flag"]=np.where(d.anon_singling_lo.isna(),np.nan,(d[["anon_singling_lo","anon_link_lo","anon_infer_lo"]].fillna(0)>0).any(axis=1).astype(float))
order=["Adult","German","Fraud","Pima","COMPAS","Credit","Bank","Diabetes130"]
t="\\begin{table}[t]\\centering\\footnotesize\n\\caption{MST at $\\varepsilon=1$ with category lists read from the training data (D) and from a public schema (P); means over seeds (exploratory re-analysis). AUC: strongest of three attacks. Over null: runs above the dataset's null 95th percentile of the AUC. Anonymeter: runs with a risk above zero among runs where it was computed.}\\label{tab:mstmain}\n\\begin{tabular}{lcrrrrrr}\\toprule\nDataset & List & Seeds & $U$ & AUC & TPR@1\\% & Over null & Anonymeter\\\\\\midrule\n"
for ds in order:
    for v,l in (("data_derived","D"),("public_schema","P")):
        s=d[(d.dataset==ds)&(d.variant==v)]
        if len(s)==0: continue
        over=int((s.mia_auc_max>s.mia_auc_max_null_p95).sum())
        an=s.anon_flag.dropna(); ans=f"{int(an.sum())}/{len(an)}" if len(an) else "--"
        t+=f"{ds if v=='data_derived' else ''} & {l} & {len(s)} & {s.util_ratio.mean():.2f} & {s.mia_auc_max.mean():.3f} & {s.mia_tpr1_rand_max.mean():.3f} & {over}/{len(s)} & {ans}\\\\\n"
t+="\\bottomrule\\end{tabular}\\end{table}\n"
open(OUT+"tab_mst_main.tex","w").write(t)
# ---------- C6 rows
rows=[]
for _,r in d.iterrows():
    for a in ["sex","race","age<=25","age<30"]:
        k=f"{a}|auc_unpriv"
        if k in r and pd.notna(r[k]):
            rows.append(dict(dataset=r.dataset,attr=a,seed=r.seed,variant=r.variant,drop_prot=r[k+"_real"]-r[k],drop_ref=r[f"{a}|auc_priv_real"]-r[f"{a}|auc_priv"]))
C=pd.DataFrame(rows); C=C[C.dataset.isin(["Adult","German","Pima"])]; C["diff"]=C.drop_prot-C.drop_ref
c6={}
for v,s in C.groupby("variant"):
    w=stats.wilcoxon(s["diff"],alternative="greater"); c6[v]=dict(n=len(s),prot=s.drop_prot.mean(),ref=s.drop_ref.mean(),diff=s["diff"].mean(),p=w.pvalue)
print(c6)
pd.DataFrame(c6).to_csv("outputs/c6_summary.csv")
print(d.groupby("variant").agg(n=("seed","count"),auc=("mia_auc_max","mean"),aucmax=("mia_auc_max","max"),over=("mia_auc_max",lambda x:0)).round(3))
ph1=d[d.dataset.isin(["Adult","German","Fraud","Pima"])]
print("anon flagged phase I:",ph1.groupby("variant").anon_flag.agg(["sum","count"]))
d["over"]=d.mia_auc_max>d.mia_auc_max_null_p95; print(d.groupby("variant").over.agg(["sum","count"]))
p=d.pivot_table(index=["dataset","seed"],columns="variant",values="util_ratio").dropna(); print("util diff",(p.public_schema-p.data_derived).mean().round(3),(p.public_schema-p.data_derived).min().round(3),(p.public_schema-p.data_derived).max().round(3),len(p))
