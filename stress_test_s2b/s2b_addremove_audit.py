"""Stress test S2b: add/remove canary audit of SmartNoise MST with a fixed dataset.

D  = 499 Adult records (7-column schema) that contain NONE of the canary's rare categories.
D' = D plus the canary.  This is the add/remove neighbouring relation used by MST's guarantee.
Test statistic (no training needed): does the release contain marital_status == "Married-AF-spouse"
or education == "Preschool"?  FPR is measured on releases of D, TPR on releases of D'.

Variants
  default : no transformer fitted on a public schema, so SmartNoise fits it on the training data and reads the
            category lists from there (the same configuration as our main MST runs).
  public  : category encoders fitted on the public schema (all categories of the full Adult file).
"""
import os, sys, json, time, random, warnings, urllib.request, multiprocessing as mp
import concurrent.futures as cf
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
COLS7 = ["age", "education", "marital_status", "race", "sex", "hours_per_week", "income"]
NUM = ["age", "hours_per_week"]; CAT = [c for c in COLS7 if c not in NUM]
CANARY = dict(age=90.0, education="Preschool", marital_status="Married-AF-spouse", race="Other", sex="Female", hours_per_week=99.0, income=">50K")
RARE = {"marital_status": "Married-AF-spouse", "education": "Preschool"}
DELTA = 1e-5

def load_adult():
    p = os.path.join(OUT, "adult-all.csv")
    if not os.path.exists(p): urllib.request.urlretrieve(URL, p)
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship",
            "race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
    df = pd.read_csv(p, header=None, names=cols, na_values="?", skipinitialspace=True).dropna()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df = df[COLS7].reset_index(drop=True)
    for c in CAT: df[c] = df[c].astype(str)
    for c in NUM: df[c] = df[c].astype(float)
    return df

def fit_release(args):
    train, bounds, domains, seed, eps, public, n = args
    from snsynth import Synthesizer
    from snsynth.transform import TableTransformer, BinTransformer, LabelTransformer
    import snsynth.mst.mst as _m
    _md = _m.estimation.mirror_descent
    def _capped(*a, **k):  # same iteration cap as the main canary audit (post-processing only)
        k["iters"] = min(k.get("iters", 1000), 1000); return _md(*a, **k)
    _m.estimation.mirror_descent = _capped
    try:
        import jax
        jax.config.update("jax_compilation_cache_dir", os.path.join(OUT, "jaxcache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except Exception:
        pass
    np.random.seed(seed); random.seed(seed)
    tt = TableTransformer([BinTransformer(bins=20, lower=bounds[c][0], upper=bounds[c][1]) if c in NUM else LabelTransformer(nullable=False)
                           for c in train.columns])
    if public:
        m = max(len(domains[c]) for c in CAT)
        dom = pd.DataFrame({c: ([domains[c][i % len(domains[c])] for i in range(m)] if c in CAT else [bounds[c][0]] * m) for c in train.columns})
        tt.fit(dom)
    s = Synthesizer.create("mst", epsilon=float(eps), verbose=False)
    s.fit(train, transformer=tt, preprocessor_eps=0.0, categorical_columns=CAT, continuous_columns=NUM)
    out = s.sample(n)
    return int(any((out[c].astype(str) == v).any() for c, v in RARE.items()))

def cp_upper(k, m, a=0.025):
    return 1.0 if k == m else float(stats.beta.ppf(1 - a, k + 1, m - k))

def eps_lb(tp, m1, fp, m0):
    fnr_u, fpr_u = cp_upper(m1 - tp, m1), cp_upper(fp, m0)
    vals = []
    for num, den in [(1 - DELTA - fpr_u, fnr_u), (1 - DELTA - fnr_u, fpr_u)]:
        if num > 0 and den > 0: vals.append(np.log(num / den))
    return max([0.0] + vals), fpr_u, fnr_u

if __name__ == "__main__":
    plan = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [["default", 1.0, 100], ["default", 0.1, 60], ["public", 1.0, 50]]
    df = load_adult()
    bounds = {c: (float(df[c].min()), float(df[c].max())) for c in NUM}
    domains = {c: sorted(df[c].unique()) for c in CAT}
    pool = df[~(df["marital_status"].eq(RARE["marital_status"]) | df["education"].eq(RARE["education"]))].reset_index(drop=True)
    D = pool.sample(499, random_state=7).reset_index(drop=True)
    Dp = pd.concat([D, pd.DataFrame([CANARY])], ignore_index=True)
    for c in NUM: Dp[c] = Dp[c].astype(float)
    print(f"Adult rows {len(df)}, pool without rare categories {len(pool)}, |D|={len(D)}, |D'|={len(Dp)}", flush=True)
    rows = []
    for variant, eps, m in plan:
        t0 = time.time()
        jobs = [(Dp if w else D, bounds, domains, 1000 * w + i, eps, variant == "public", 500) for w in (0, 1) for i in range(m)]
        with cf.ProcessPoolExecutor(2, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
            res = list(ex.map(fit_release, jobs))
        fp, tp = sum(res[:m]), sum(res[m:])
        lb, fpr_u, fnr_u = eps_lb(tp, m, fp, m)
        ceil = eps_lb(m, m, 0, m)[0]
        r = dict(variant=variant, eps_nominal=eps, releases_per_world=m, rare_in_D=fp, rare_in_Dprime=tp,
                 FPR=fp / m, TPR=tp / m, FPR_upper=fpr_u, FNR_upper=fnr_u, eps_lb=lb, eps_lb_ceiling=ceil, minutes=(time.time() - t0) / 60)
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "S2b_addremove_audit.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string())
