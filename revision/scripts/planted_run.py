"""S1/R10: graded planted-leakage controls. Release = alpha*n exact copies of training rows + (1-alpha)*n fresh real rows never seen in training.
Known leakage = alpha. Report the dose-response of attack AUC, TPR@1% (conventional and tie-aware), DCR, Anonymeter, and the gate outputs."""
import sys, numpy as np, pandas as pd, time
import sdaudit as S
ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
rows = []
for key in sys.argv[1].split(","):
    df, meta = S.load(key)
    for seed in range(int(sys.argv[3])):
        tr, ho, te = S.make_split(df, meta, seed); enc = S.Encoder(tr, meta); n = len(tr)
        used = pd.concat([tr, ho, te]).astype(str).agg("|".join, axis=1)
        keyrow = df.astype(str).agg("|".join, axis=1); pool = df[~keyrow.isin(set(used))].reset_index(drop=True)
        if len(pool) < n: print("pool too small", key, len(pool)); continue
        pool = pool.sample(n, random_state=seed).reset_index(drop=True)
        for al in ALPHAS:
            t0 = time.time(); m = int(round(al * n)); rs = np.random.RandomState(seed)
            syn = pd.concat([tr.iloc[rs.choice(n, m, replace=False)], pool.iloc[: n - m]]).sample(frac=1, random_state=seed).reset_index(drop=True)
            pv = S.privacy(tr, ho, te, syn, meta, enc, seed, anonymeter_on=True); pv.pop("_scores")
            r = dict(dataset=meta["name"], seed=seed, alpha=al, **{k: v for k, v in pv.items()})
            r["gate_orig_privacy"] = (r["exact_match_syn"] < 0.001) and (r["dcr_share"] <= 0.55)
            r["gate_new_attack"] = (r["mia_tpr1_max"] <= 0.02) and (r.get("mia_tpr10_outlier", 0.0) <= 0.20)
            r["gate_new_attack_tieaware"] = (r["mia_tpr1_rand_max"] <= 0.02) and (r.get("mia_tpr10_outlier", 0.0) <= 0.20)
            r["gate_anonymeter"] = all((r.get(f"anon_{k}_lo", 0) or 0) <= 0 for k in ("singling", "link", "infer"))
            rows.append(r); print(key, seed, al, f"auc={r['mia_auc_max']:.3f} tpr={r['mia_tpr1_max']:.3f}/{r['mia_tpr1_rand_max']:.3f} dcr={r['dcr_share']:.3f} gates(o,n,a)={r['gate_orig_privacy']},{r['gate_new_attack']},{r['gate_anonymeter']} {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f"outputs/planted_{sys.argv[2]}.csv", index=False)
