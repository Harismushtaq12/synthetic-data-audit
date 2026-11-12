"""R7/R8: tie-aware TPR@1%, duplicate ceiling for the positive control, and a permutation null per dataset and run.
For each Phase I / Phase II dataset-seed we score members vs non-members against (a) Identity (positive control) and (b) an Independent
release. Attack scores do not depend on the membership labels, so the null is obtained by permuting labels 500 times."""
import sys, numpy as np, pandas as pd, time
import sdaudit as S
from sklearn.metrics import roc_auc_score
PH1 = {"adult": range(3), "german": range(5), "fraud": range(3), "pima": range(5)}
PH2 = {"compas": range(3), "credit": range(3), "bank": range(3), "diabetes": range(3)}
keys = sys.argv[1].split(",")
rows = []
for key in keys:
    df, meta = S.load(key); seeds = {**PH1, **PH2}[key]
    for seed in seeds:
        t0 = time.time(); tr, ho, te = S.make_split(df, meta, seed); enc = S.Encoder(tr, meta)
        n = min(len(ho), len(tr)); rs = np.random.RandomState(seed); mi = rs.choice(len(tr), n, replace=False)
        mem = tr.iloc[mi].reset_index(drop=True); non = ho.iloc[:n].reset_index(drop=True)
        E_mem, E_non, E_ref = enc(mem), enc(non), enc(te); T = np.vstack([E_mem, E_non]); y = np.r_[np.ones(n), np.zeros(n)]
        q = S.duplicate_share(tr, non)
        # positive control
        sc_pos = S.mia_scores(T, enc(tr), E_ref, np.random.RandomState(seed))
        # independent release
        ind = S.postprocess(S.gen_independent(tr, meta, seed), tr, meta)
        sc_ind = S.mia_scores(T, enc(ind), E_ref, np.random.RandomState(seed))
        p = np.random.RandomState(1000 + seed); R = 500
        nulls = {a: dict(auc=[], tc=[], tr=[]) for a in sc_ind}; nulls["max"] = dict(auc=[], tc=[], tr=[])
        for _ in range(R):
            yp = y[p.permutation(2 * n)]; mx = dict(auc=0, tc=0, tr=0)
            for a, s in sc_ind.items():
                auc = roc_auc_score(yp, s); tc = S._tpr_at(yp, s, 0.01); trr = S.tpr_at_randomized(yp, s, 0.01)[0]
                nulls[a]["auc"].append(auc); nulls[a]["tc"].append(tc); nulls[a]["tr"].append(trr)
                mx = dict(auc=max(mx["auc"], auc), tc=max(mx["tc"], tc), tr=max(mx["tr"], trr))
            for k_, v in mx.items(): nulls["max"][k_].append(v)
        r = dict(dataset=meta["name"], seed=seed, n=n, n_neg_in_1pct=0.01 * n, dup_share_holdout=q, ceiling_tpr1=min(1.0, 0.01 / q) if q > 0 else 1.0)
        for a, s in sc_pos.items():
            r[f"pos_auc_{a}"] = roc_auc_score(y, s); r[f"pos_tpr1_conv_{a}"] = S._tpr_at(y, s, 0.01); r[f"pos_tpr1_rand_{a}"] = S.tpr_at_randomized(y, s, 0.01)[0]
        r["pos_tpr1_rand_max"] = max(r[f"pos_tpr1_rand_{a}"] for a in sc_pos)
        for a in nulls:
            r[f"null_auc_mean_{a}"] = float(np.mean(nulls[a]["auc"])); r[f"null_auc_p95_{a}"] = float(np.percentile(nulls[a]["auc"], 95))
            r[f"null_tpr1_conv_mean_{a}"] = float(np.mean(nulls[a]["tc"])); r[f"null_tpr1_conv_p95_{a}"] = float(np.percentile(nulls[a]["tc"], 95))
            r[f"null_tpr1_rand_mean_{a}"] = float(np.mean(nulls[a]["tr"])); r[f"null_tpr1_rand_p95_{a}"] = float(np.percentile(nulls[a]["tr"], 95))
            r[f"null_share_tpr1_ge_0.02_{a}"] = float(np.mean(np.array(nulls[a]["tr"]) >= 0.02))
            r[f"null_share_tpr1_conv_ge_0.02_{a}"] = float(np.mean(np.array(nulls[a]["tc"]) >= 0.02))
        for a, s in sc_ind.items():
            r[f"ind_auc_{a}"] = roc_auc_score(y, s); r[f"ind_tpr1_rand_{a}"] = S.tpr_at_randomized(y, s, 0.01)[0]
        rows.append(r); print(key, seed, f"n={n} dup={q:.3f} pos_conv={r['pos_tpr1_conv_A1_distance']:.3f} pos_rand={r['pos_tpr1_rand_A1_distance']:.3f} ceil={r['ceiling_tpr1']:.3f} "
              f"null_max_p95_rand={r['null_tpr1_rand_p95_max']:.3f} share>=.02={r['null_share_tpr1_ge_0.02_max']:.2f}  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"outputs/null_controls_{sys.argv[2]}.csv", index=False)
