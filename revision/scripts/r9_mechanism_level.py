"""R9/R8: mechanism-level re-analysis of the utility-leakage link.
Phase I: cell means (dataset x mechanism) printed in the executed Part I notebook (T3 utility, strongest-attack AUC).
Held-out: run-level rows in X5_results.csv (Phase II)."""
import numpy as np, pandas as pd, itertools, json
from scipy import stats
rs = np.random.RandomState(0)
SYN = ["Independent","GaussianCopula","CART","CTGAN","TVAE","DDPM","MST (ε=1)"]
U = {"Adult":[.484,.824,.988,.936,.960,.933,.771],"German":[.666,.795,.936,.678,.775,.894,.694],
     "Fraud":[.706,.916,.984,.985,.986,.989,.677],"Pima":[.644,.879,.976,.656,.964,.981,.822]}
A = {"Adult":[.506,.506,.592,.510,.515,.511,.506],"German":[.534,.539,.621,.526,.517,.548,.523],
     "Fraud":[.513,.509,.559,.509,.515,.535,.505],"Pima":[.553,.520,.629,.512,.541,.712,.524]}
out = {}
def perm_p(x, y):
    """exact two-sided permutation p-value of Spearman rho over all n! pairings (n<=8)"""
    r0 = stats.spearmanr(x, y)[0]; n = len(x); cnt = tot = 0
    for p in itertools.permutations(range(n)):
        r = stats.spearmanr(x, np.asarray(y)[list(p)])[0]; tot += 1; cnt += abs(r) >= abs(r0) - 1e-12
    return r0, cnt / tot
# --- Phase I, mechanism level (n=7), pooled means over the four datasets
Um = np.mean([U[d] for d in U], 0); Am = np.mean([A[d] for d in A], 0)
r0, p = perm_p(Um, Am); out["phase1_mech_rho"] = r0; out["phase1_mech_perm_p"] = p
loo = [stats.spearmanr(np.delete(Um, i), np.delete(Am, i))[0] for i in range(7)]
out["phase1_loo_min"], out["phase1_loo_max"] = min(loo), max(loo)
out["phase1_loo_names"] = {SYN[i]: round(l, 2) for i, l in enumerate(loo)}
# per dataset, mechanism level
out["phase1_per_dataset_rho"] = {d: round(stats.spearmanr(U[d], A[d])[0], 2) for d in U}
# cell level (28 points) with mechanism-cluster bootstrap and dataset-rank pooling like the paper
rows = [(d, SYN[i], U[d][i], A[d][i]) for d in U for i in range(7)]
C = pd.DataFrame(rows, columns=["dataset","mech","U","A"])
C["rU"] = C.groupby("dataset").U.rank(pct=True); C["rA"] = C.groupby("dataset").A.rank(pct=True)
out["phase1_cell_rho"] = stats.spearmanr(C.rU, C.rA)[0]
def cluster_boot(df, xcol, ycol, B=5000):
    mechs = df.mech.unique(); b = []
    for _ in range(B):
        pick = rs.choice(mechs, len(mechs), replace=True)
        s = pd.concat([df[df.mech == m] for m in pick])
        if s[xcol].nunique() > 1 and s[ycol].nunique() > 1: b.append(stats.spearmanr(s[xcol], s[ycol])[0])
    return np.percentile(b, [2.5, 97.5])
out["phase1_cell_rho_cluster_ci"] = list(cluster_boot(C, "rU", "rA"))
# --- held-out, run level
H = pd.read_csv("../notebooks/phase2/outputs2/tables/X5_results.csv")
H = H[H.dataset.isin(["COMPAS","Credit","Bank","Diabetes130"]) & ~H.generator.isin(["Identity","k-anon (k=10)"])].copy()
H["mech"] = H.generator
H["rU"] = H.groupby("dataset").util_ratio.rank(pct=True); H["rA"] = H.groupby("dataset").mia_auc_max.rank(pct=True)
out["heldout_runs"] = len(H); out["heldout_mechs"] = sorted(H.mech.unique())
out["heldout_run_rho"] = stats.spearmanr(H.rU, H.rA)[0]
rr = [stats.spearmanr(*(H[["rU","rA"]].values[rs.randint(0,len(H),len(H))].T))[0] for _ in range(5000)]
out["heldout_run_boot_ci"] = list(np.nanpercentile(rr, [2.5, 97.5]))
out["heldout_cluster_ci"] = list(cluster_boot(H, "rU", "rA"))
M = H.groupby("mech")[["util_ratio","mia_auc_max"]].mean()
r0, p = perm_p(M.util_ratio.values, M.mia_auc_max.values); out["heldout_mech_rho"] = r0; out["heldout_mech_perm_p"] = p; out["heldout_mech_n"] = len(M)
out["heldout_mech_means"] = M.round(3).to_dict()
loo = {m: round(stats.spearmanr(M.drop(m).util_ratio, M.drop(m).mia_auc_max)[0], 2) for m in M.index}; out["heldout_loo"] = loo
# within-mechanism slope: demeaned within (mechanism) ranks
H["rU_w"] = H.rU - H.groupby("mech").rU.transform("mean"); H["rA_w"] = H.rA - H.groupby("mech").rA.transform("mean")
out["heldout_within_mech_rho"] = stats.spearmanr(H.rU_w, H.rA_w)[0]
# --- R8: dataset as the unit. Held-out, per mechanism: per-dataset one-sided Wilcoxon on strongest-attack AUC>0.5 (3 seeds each, so use t-free sign statistic)
Hh = pd.read_csv("../notebooks/phase2/outputs2/tables/X5_results.csv")
Hh = Hh[Hh.dataset.isin(["COMPAS","Credit","Bank","Diabetes130"])]
res = []
for g in sorted(Hh.generator.unique()):
    s = Hh[Hh.generator == g]
    dm = s.groupby("dataset").mia_auc_max.mean()                     # dataset means (n=4)
    # dataset-level exact sign-flip test on (mean AUC - 0.5): H0 symmetric about 0
    d = dm.values - 0.5; n = len(d); obs = d.mean()
    flips = np.array([np.mean(np.array(sg) * np.abs(d)) for sg in itertools.product([-1, 1], repeat=n)])
    p = np.mean(flips >= obs - 1e-12)
    res.append(dict(mechanism=g, datasets_above_half=int((d > 0).sum()), mean_auc=round(dm.mean(), 3), dataset_level_p=round(p, 4)))
R8 = pd.DataFrame(res); R8["p_holm"] = np.minimum(1, R8.dataset_level_p.rank(method="first").pipe(lambda r: 0) + 0)  # placeholder, fixed below
o = np.argsort(R8.dataset_level_p.values); m = len(R8); adj = np.empty(m); run = 0
for i, j in enumerate(o): run = max(run, (m - i) * R8.dataset_level_p.values[j]); adj[j] = min(1, run)
R8["p_holm"] = adj.round(4)
R8.to_csv("outputs/R8_dataset_level.csv", index=False)
out["R8"] = R8.to_dict("records")
C.to_csv("outputs/R9_phase1_cells.csv", index=False)
json.dump(out, open("outputs/R9.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
