import pandas as pd, numpy as np, glob, os
OUT="tables_tex/"; os.makedirs(OUT,exist_ok=True)
def foot(): return "\\bottomrule\\end{tabular}\\end{table}\n"
# ---------- pooled S1
if os.path.exists("outputs/pooled_s1.csv"):
    p=pd.read_csv("outputs/pooled_s1.csv")
    t="\\begin{table}[t]\\centering\\footnotesize\n\\caption{Pooled attacks at the S1 setting (Adult, 500 training rows, 500 holdout rows, 5{,}000 reference rows); means over seeds (exploratory re-analysis). A1, A2, A3: attack AUC. Over null: seeds whose strongest AUC exceeded the 95th percentile of the label-permutation null. TPR: tie-aware TPR@1\\% of the strongest attack.}\\label{tab:pooled}\n\\begin{tabular}{lrrrrrrr}\\toprule\nGenerator & Seeds & A1 & A2 & A3 & Strongest & Over null & TPR\\\\\\midrule\n"
    for g,s in p.groupby("generator",sort=False):
        over=int((s.mia_auc_max>s.mia_auc_max_null_p95).sum())
        t+=f"{g} & {len(s)} & {s.mia_auc_A1_distance.mean():.3f} & {s.mia_auc_A2_density_ratio.mean():.3f} & {s.mia_auc_A3_classifier_ratio.mean():.3f} & {s.mia_auc_max.mean():.3f} & {over}/{len(s)} & {s.mia_tpr1_rand_max.mean():.3f}\\\\\n"
    open(OUT+"tab_pooled.tex","w").write(t+foot())
    print(p.groupby("generator",sort=False).agg(n=("seed","count"),auc=("mia_auc_max","mean"),p95=("mia_auc_max_null_p95","mean")).round(3))
# ---------- sweep
fs=[f for f in glob.glob("outputs/mst_sweep*.csv")]
if fs:
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).drop_duplicates(["dataset","seed","eps","variant"])
    d["gl"]=d[["sex|drop_unpriv","sex|drop_priv"]].apply(lambda r:r.iloc[0]-r.iloc[1],axis=1) if "sex|drop_unpriv" in d else np.nan
    t="\\begin{table}[t]\\centering\\footnotesize\n\\caption{MST with a public schema across privacy budgets (exploratory re-analysis); means over seeds. AUC: strongest of three attacks. Over null: runs above the dataset's null 95th percentile.}\\label{tab:mstsweep}\n\\begin{tabular}{lrrrrrr}\\toprule\nDataset & $\\varepsilon$ & Seeds & $U$ & AUC & TPR@1\\% & Over null\\\\\\midrule\n"
    for ds,g in d.groupby("dataset",sort=False):
        for e,s in g.groupby("eps"):
            over=int((s.mia_auc_max>s.mia_auc_max_null_p95).sum())
            t+=f"{ds if e==g.eps.min() else ''} & {e:g} & {len(s)} & {s.util_ratio.mean():.2f} & {s.mia_auc_max.mean():.3f} & {s.mia_tpr1_rand_max.mean():.3f} & {over}/{len(s)}\\\\\n"
    open(OUT+"tab_mst_sweep.tex","w").write(t+foot())
    print(d.groupby(["dataset","eps"]).agg(n=("seed","count"),u=("util_ratio","mean"),auc=("mia_auc_max","mean"),aucmax=("mia_auc_max","max")).round(3))
# ---------- planted
fs=glob.glob("outputs/planted_*.csv")
if fs:
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).drop_duplicates(["dataset","seed","alpha"])
    d["anon"]=(d[["anon_singling_lo","anon_link_lo","anon_infer_lo"]].fillna(0)>0).any(axis=1)
    t="\\begin{table}[t]\\centering\\footnotesize\n\\caption{Planted leakage: a share $\\alpha$ of the training rows is copied into the release (exploratory). Means over seeds. TPR: tie-aware TPR@1\\% of the strongest attack. Pass shares give the fraction of runs that each gate approves. Orig.: original privacy gate (DCR); Attack: evidence-based attack gate; Tie: its tie-aware version; Anon.: Anonymeter gate.}\\label{tab:planted}\n\\begin{tabular}{lrrrrrrrrr}\\toprule\nData & $\\alpha$ & Runs & DCR share & AUC & TPR & Orig. & Attack & Tie & Anon.\\\\\\midrule\n"
    for ds,g in d.groupby("dataset",sort=False):
        for a,s in g.groupby("alpha"):
            f=lambda c:f"{s[c].astype(float).mean():.2f}"
            t+=f"{ds if a==g.alpha.min() else ''} & {a:g} & {len(s)} & {s.dcr_share.mean():.3f} & {s.mia_auc_max.mean():.3f} & {s.mia_tpr1_rand_max.mean():.3f} & {f('gate_orig_privacy')} & {f('gate_new_attack')} & {f('gate_new_attack_tieaware')} & {f('gate_anonymeter')}\\\\\n"
    open(OUT+"tab_planted.tex","w").write(t+foot())
    print(d.groupby(["dataset","alpha"]).agg(n=("seed","count"),auc=("mia_auc_max","mean"),tpr=("mia_tpr1_rand_max","mean"),o=("gate_orig_privacy","mean"),a=("gate_new_attack","mean"),ta=("gate_new_attack_tieaware","mean"),an=("gate_anonymeter","mean")).round(3))
