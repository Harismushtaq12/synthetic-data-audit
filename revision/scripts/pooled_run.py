"""R1/EIC Q3: like-for-like pooled attack test at the S1 setting (Adult, 500 training rows, 500 holdout rows, reference sample 5,000), 10 seeds per generator."""
import numpy as np, pandas as pd, time
from scipy import stats
import sdaudit as S
df, meta = S.load("adult"); rows = []
GEN = {"Independent": S.gen_independent, "GaussianCopula": S.gen_gc, "CART": S.gen_cart, "CTGAN": S.gen_ctgan}
for seed in range(10):
    idx = np.random.RandomState(100 + seed).permutation(len(df))
    tr, ho, te = (df.iloc[idx[:500]].reset_index(drop=True), df.iloc[idx[500:1000]].reset_index(drop=True), df.iloc[idx[1000:6000]].reset_index(drop=True))
    enc = S.Encoder(tr, meta)
    for g, fn in GEN.items():
        t0 = time.time(); syn = S.postprocess(fn(tr, meta, seed), tr, meta)
        pv = S.privacy(tr, ho, te, syn, meta, enc, seed, anonymeter_on=False); pv.pop("_scores")
        r = dict(seed=seed, generator=g, **{k: v for k, v in pv.items() if k.startswith(("mia_auc_A", "mia_tpr1_rand", "mia_auc_max", "dcr_share", "exact"))})
        rows.append(r); print(seed, g, f"A1={r['mia_auc_A1_distance']:.3f} A2={r['mia_auc_A2_density_ratio']:.3f} A3={r['mia_auc_A3_classifier_ratio']:.3f} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("outputs/pooled_s1.csv", index=False)
