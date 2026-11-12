"""R12: rerun the MST arm with category lists from a PUBLIC schema, next to the original data-derived-list arm.
Same splits, seeds, evaluation code as Phase I / Phase II. eps = 1 on all 8 datasets; eps sweep on Adult and German."""
import sys, os, numpy as np, pandas as pd, time
import sdaudit as S
PH1 = {"adult": range(3), "german": range(5), "fraud": range(3), "pima": range(5)}
PH2 = {"compas": range(3), "credit": range(3), "bank": range(3), "diabetes": range(3)}
mode, keys = sys.argv[1], sys.argv[2].split(",")
rows = []
for key in keys:
    df, meta = S.load(key); seeds = {**PH1, **PH2}[key]
    if os.environ.get('NSEED'): seeds = range(int(os.environ['NSEED']))
    for seed in seeds:
        tr, ho, te = S.make_split(df, meta, seed); enc = S.Encoder(tr, meta)
        u_r, p_r = S.utility(tr, te, meta, seed); f_r = S.fairness(tr, te, p_r["HGB"], meta)
        real = (u_r, f_r)
        eps_list = [1.0] if mode == "main" else [0.1, 1.0, 10.0, 100.0]
        for eps in eps_list:
            for variant in (("data_derived", "public_schema") if mode == "main" else ("public_schema",)):
                t0 = time.time()
                syn = S.postprocess(S.gen_mst(tr, meta, seed, eps=eps, public_domain=(variant == "public_schema")), tr, meta)
                if mode == "main":
                    r, sc = S.evaluate_release(tr, ho, te, syn, meta, seed, real=real, anonymeter_on=(os.environ.get('NOANON') != '1'))
                else:
                    u, p = S.utility(syn, te, meta, seed); f = S.fairness(syn, te, p["HGB"], meta)
                    pv = S.privacy(tr, ho, te, syn, meta, enc, seed, anonymeter_on=False); pv.pop("_scores")
                    r = dict(util_ratio=u["auc_mean"] / u_r["auc_mean"], **pv)
                    for a in meta["sensitive"]:
                        r[f"{a}|drop_unpriv"] = f_r[f"{a}|auc_unpriv"] - f[f"{a}|auc_unpriv"]; r[f"{a}|drop_priv"] = f_r[f"{a}|auc_priv"] - f[f"{a}|auc_priv"]
                r.update(dataset=meta["name"], seed=seed, eps=eps, variant=variant, secs=time.time() - t0)
                rows.append(r)
                print(key, seed, eps, variant, f"util={r['util_ratio']:.3f} auc={r['mia_auc_max']:.3f} tpr={r['mia_tpr1_max']:.3f} rand={r['mia_tpr1_rand_max']:.3f} {r['secs']:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(f"outputs/mst_{mode}_{sys.argv[3]}.csv", index=False)
