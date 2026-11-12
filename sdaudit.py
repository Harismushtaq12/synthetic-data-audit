"""
sdaudit — an audit toolkit for synthetic tabular data releases.

Companion code for "Is synthetic data private by default? An empirical audit of twelve claims about synthetic tabular data"
(Mushtaq & Yousuf, NUTECH). The audit tests claims from an earlier, unpublished draft on synthetic data for privacy-compliant AI.
The module bundles:
  * loaders for eight public datasets (download + SHA-256),
  * eleven release mechanisms (references, statistical, tree, deep, autoregressive transformer, DP, k-anonymity)
    plus representation-preserving wrappers,
  * the evaluation suite (fidelity, diversity, utility, privacy, fairness, cost),
  * stronger attacks (record-targeted shadow-model MIA, vulnerable-record scoring, linear reconstruction,
    empirical DP auditing),
  * the original SDSAF gates and the proposed evidence-based gates, via `audit()`.
"""
from __future__ import annotations
import os, io, json, math, time, zipfile, hashlib, logging, warnings, itertools, urllib.request
import concurrent.futures, multiprocessing
warnings.filterwarnings("ignore")
for _n in ["codecarbon", "sdv", "rdt", "SingleTableSynthesizer", "copulas", "snsynth", "mbi"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from scipy import stats
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve

torch.set_num_threads(2)
__version__ = "0.3.0"  # 0.3.0: tie-aware TPR@1% (randomised threshold), duplicate ceiling, screening wording

CFG = dict(ctgan_epochs=300, tvae_epochs=300, ddpm_steps=5000, ar_steps=1500, mst_eps=1.0, kanon_k=10,
           anon_attacks=500, data_dir="data_raw", jax_cache=os.path.abspath("jaxcache"))

def set_seed(s):
    import random
    np.random.seed(s); torch.manual_seed(s); random.seed(s)

# =====================================================================================================
# 1. DATA
# =====================================================================================================
_JB = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
_FD = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/"
URLS = {"adult": _JB + "adult-all.csv", "german": _JB + "german.csv", "pima": _JB + "pima-indians-diabetes.csv",
        "fraud": _JB + "creditcard.csv.zip", "compas": _FD + "compas-scores-two-years_clean.csv",
        "credit": _FD + "credit-card-clients.csv", "bank": _FD + "bank-full.csv", "diabetes": _FD + "diabetes-clean.csv"}

def fetch(key):
    os.makedirs(CFG["data_dir"], exist_ok=True)
    p = os.path.join(CFG["data_dir"], URLS[key].split("/")[-1])
    if not os.path.exists(p): urllib.request.urlretrieve(URLS[key], p)
    return p, hashlib.sha256(open(p, "rb").read()).hexdigest()

def _adult(p):
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship",
            "race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
    df = pd.read_csv(p, header=None, names=cols, na_values="?", skipinitialspace=True).dropna()
    df = df.drop(columns=["fnlwgt", "education_num"]); df["income"] = df["income"].str.replace(".", "", regex=False)
    return df.reset_index(drop=True), dict(name="Adult", sector="HR / income", target="income", pos=">50K",
        num=["age","capital_gain","capital_loss","hours_per_week"],
        sensitive={"sex": ("sex", "==", "Female"), "race": ("race", "!=", "White")},
        qi=["age","sex","race","marital_status","education","native_country"])

def _german(p):
    cols = ["checking","duration","credit_history","purpose","amount","savings","employment","installment_rate",
            "personal_status","other_debtors","residence_since","property","age","other_plans","housing",
            "n_credits","job","n_liable","telephone","foreign_worker","risk"]
    df = pd.read_csv(p, header=None, names=cols)
    df["sex"] = np.where(df["personal_status"].isin(["A92", "A95"]), "female", "male")
    df = df.drop(columns=["personal_status"]); df["risk"] = df["risk"].map({1: "good", 2: "bad"})
    return df, dict(name="German", sector="Credit scoring", target="risk", pos="bad",
        num=["duration","amount","installment_rate","residence_since","age","n_credits","n_liable"],
        sensitive={"sex": ("sex", "==", "female"), "age<=25": ("age", "<=", 25)},
        qi=["age","sex","job","housing","foreign_worker","employment"])

def _fraud(p, seed=2024):
    with zipfile.ZipFile(p) as z:
        df = pd.read_csv(z.open("creditcard.csv"), header=None, names=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])
    df = df.drop(columns=["Time"]); fraud = df[df.Class == 1]; normal = df[df.Class == 0].sample(10000 - len(fraud), random_state=seed)
    df = pd.concat([fraud, normal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["Class"] = df["Class"].map({0: "normal", 1: "fraud"})
    return df, dict(name="Fraud", sector="Fraud detection", target="Class", pos="fraud", num=[c for c in df.columns if c != "Class"],
        sensitive={}, qi=["Amount","V1","V2","V3","V4","V5"])

def _pima(p):
    cols = ["pregnancies","glucose","blood_pressure","skin_thickness","insulin","bmi","dpf","age","outcome"]
    df = pd.read_csv(p, header=None, names=cols); df["outcome"] = df["outcome"].map({0: "neg", 1: "pos"})
    return df, dict(name="Pima", sector="Healthcare", target="outcome", pos="pos", num=cols[:-1],
        sensitive={"age<30": ("age", "<", 30)}, qi=["age","pregnancies","bmi","blood_pressure"])

def _compas(p):
    d = pd.read_csv(p)
    df = d[["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree","two_year_recid"]].copy()
    df["two_year_recid"] = df["two_year_recid"].map({0: "no", 1: "yes"})
    return df.dropna().reset_index(drop=True), dict(name="COMPAS", sector="Criminal justice", target="two_year_recid", pos="yes",
        num=["age","juv_fel_count","juv_misd_count","juv_other_count","priors_count"],
        sensitive={"race": ("race", "==", "African-American"), "sex": ("sex", "==", "Female")},
        qi=["age","sex","race","c_charge_degree"])

def _credit(p):
    df = pd.read_csv(p).rename(columns={"default payment": "default"})
    df["SEX"] = df["SEX"].map({1: "male", 2: "female"}); df["EDUCATION"] = df["EDUCATION"].astype(str); df["MARRIAGE"] = df["MARRIAGE"].astype(str)
    df["default"] = df["default"].map({0: "no", 1: "yes"})
    return df, dict(name="Credit", sector="Consumer credit", target="default", pos="yes",
        num=[c for c in df.columns if c not in ("SEX","EDUCATION","MARRIAGE","default")],
        sensitive={"sex": ("SEX", "==", "female"), "age<=25": ("AGE", "<=", 25)}, qi=["AGE","SEX","EDUCATION","MARRIAGE"])

def _bank(p):
    df = pd.read_csv(p)
    return df, dict(name="Bank", sector="Marketing", target="y", pos="yes",
        num=["age","balance","day","duration","campaign","pdays","previous"],
        sensitive={"age>=60": ("age", ">=", 60)}, qi=["age","job","marital","education"])

def _diabetes(p):
    d = pd.read_csv(p)
    keep = ["race","gender","age","admission_type_id","discharge_disposition_id","admission_source_id","time_in_hospital",
            "num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient",
            "number_diagnoses","max_glu_serum","A1Cresult","insulin","change","diabetesMed","readmitted"]
    df = d[keep].copy(); df[["max_glu_serum", "A1Cresult"]] = df[["max_glu_serum", "A1Cresult"]].fillna("None")  # "None" = not measured
    df = df[(df.race != "?") & (df.gender.isin(["Female", "Male"]))]
    for c in ["admission_type_id","discharge_disposition_id","admission_source_id"]: df[c] = df[c].astype(str)
    df["readmitted"] = df["readmitted"].map({"<30": "yes", ">30": "no"})
    return df.dropna().reset_index(drop=True), dict(name="Diabetes130", sector="Hospital readmission", target="readmitted", pos="yes",
        num=["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency",
             "number_inpatient","number_diagnoses"],
        sensitive={"race": ("race", "!=", "Caucasian"), "sex": ("gender", "==", "Female")}, qi=["race","gender","age","admission_source_id"])

LOADERS = {"adult": _adult, "german": _german, "fraud": _fraud, "pima": _pima,
           "compas": _compas, "credit": _credit, "bank": _bank, "diabetes": _diabetes}

def group_mask(df, spec):
    col, op, v = spec
    x = df[col]
    return {"==": x == v, "!=": x != v, "<=": x <= v, "<": x < v, ">=": x >= v}[op].values

def load(key):
    p, h = fetch(key); df, meta = LOADERS[key](p)
    meta["cat"] = [c for c in df.columns if c not in meta["num"]]
    for c in meta["cat"]: df[c] = df[c].astype(str)
    for c in meta["num"]: df[c] = df[c].astype(float)
    meta["int_cols"] = [c for c in meta["num"] if np.allclose(df[c], df[c].round())]
    meta["bounds"] = {c: (float(df[c].min()), float(df[c].max())) for c in meta["num"]}
    meta["domains"] = {c: sorted(df[c].unique()) for c in meta["cat"]}   # public category lists (schema)
    meta["key"], meta["sha256"] = key, h
    return df, meta

def make_split(df, meta, seed, n_large=None):
    """Equal-size train/holdout (unbiased DCR and MIA) and a test set.
    Adult: 5,000/5,000/5,000. Other datasets with >= 7,500 rows: 3,000/3,000/1,500. Smaller: 40/40/20 stratified."""
    if meta["key"] == "adult" or len(df) >= 7500 and meta["key"] != "fraud":
        m = 5000 if meta["key"] == "adult" else (n_large or 3000); t = 5000 if meta["key"] == "adult" else m // 2
        idx = np.random.RandomState(seed).permutation(len(df))
        tr, ho, te = df.iloc[idx[:m]], df.iloc[idx[m:2 * m]], df.iloc[idx[2 * m:2 * m + t]]
    else:
        tr, rest = train_test_split(df, train_size=0.4, stratify=df[meta["target"]], random_state=seed)
        ho, te = train_test_split(rest, train_size=2 / 3, stratify=rest[meta["target"]], random_state=seed)
    return tr.reset_index(drop=True), ho.reset_index(drop=True), te.reset_index(drop=True)

# =====================================================================================================
# 2. RELEASE MECHANISMS
# =====================================================================================================
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer

def postprocess(syn, train, meta):
    syn = syn[train.columns].copy()
    for c in meta["num"]:
        syn[c] = pd.to_numeric(syn[c], errors="coerce").astype(float)
        syn[c] = syn[c].fillna(train[c].median()).clip(train[c].min(), train[c].max())
        if c in meta["int_cols"]: syn[c] = syn[c].round()
    for c in meta["cat"]: syn[c] = syn[c].astype(str)
    return syn.reset_index(drop=True)

def _sdv_meta(train, meta):
    md = Metadata.detect_from_dataframe(train) if hasattr(Metadata, 'detect_from_dataframe') else Metadata.detect_from_dataframes({'data': train})  # SDV 1.x / 2.x
    try: md.remove_primary_key()   # small tables can make a unique numeric column look like a key
    except Exception:
        try: md.remove_primary_key('data')
        except Exception: pass
    for c in train.columns:
        try: md.update_column(c, sdtype="numerical" if c in meta["num"] else "categorical")
        except Exception: md.update_column(column_name=c, sdtype="numerical" if c in meta["num"] else "categorical", table_name='data')
    return md

def gen_identity(train, meta, seed, n=None): return train.copy()

def gen_independent(train, meta, seed, n=None):
    rs = np.random.RandomState(seed); n = n or len(train)
    return pd.DataFrame({c: train[c].values[rs.randint(0, len(train), n)] for c in train.columns})

def gen_gc(train, meta, seed, n=None, distribution="beta"):
    set_seed(seed); s = GaussianCopulaSynthesizer(_sdv_meta(train, meta), default_distribution=distribution)
    return _fit_sample(s, train, n or len(train))

def _fit_sample(s, train, n):
    """SDV 1.x takes and returns a DataFrame; SDV 2.x with Metadata takes and returns {table: DataFrame}."""
    try: s.fit(train); out = s.sample(n)
    except Exception: s.fit({"data": train}); out = s.sample(n)
    return out["data"] if isinstance(out, dict) else out

def _sdv_deep(cls, train, meta, seed, n, epochs, **kw):
    set_seed(seed); md = _sdv_meta(train, meta)
    try: s = cls(md, epochs=epochs, cuda=False, **kw)
    except TypeError: s = cls(md, epochs=epochs, enable_gpu=False, **kw)   # SDV 2.x
    return s, _fit_sample(s, train, n or len(train))

def gen_ctgan(train, meta, seed, n=None, epochs=None): return _sdv_deep(CTGANSynthesizer, train, meta, seed, n, epochs or CFG["ctgan_epochs"])[1]
def gen_tvae(train, meta, seed, n=None, epochs=None): return _sdv_deep(TVAESynthesizer, train, meta, seed, n, epochs or CFG["tvae_epochs"])[1]

def gen_cart(train, meta, seed, n=None, min_leaf=5):
    """synthpop-style sequential CART: x1 ~ bootstrap; x_j | x_<j ~ donor values from the matching CART leaf."""
    rs = np.random.RandomState(seed); n = n or len(train); cols = list(train.columns)
    enc = {c: {v: i for i, v in enumerate(sorted(train[c].unique()))} for c in meta["cat"]}
    X = pd.DataFrame({c: (train[c].map(enc[c]) if c in enc else train[c]).astype(float) for c in cols})
    S = pd.DataFrame(index=range(n), columns=cols, dtype=float)
    S[cols[0]] = X[cols[0]].values[rs.randint(0, len(X), n)]
    for j in range(1, len(cols)):
        c, prev = cols[j], cols[:j]
        T = (DecisionTreeClassifier if c in enc else DecisionTreeRegressor)(min_samples_leaf=min_leaf, random_state=seed)
        T.fit(X[prev].values, X[c].values)
        leaf_tr, leaf_sy = T.apply(X[prev].values), T.apply(S[prev].values.astype(float))
        donors = pd.Series(X[c].values).groupby(leaf_tr).apply(np.array).to_dict()
        S[c] = [donors[l][rs.randint(len(donors[l]))] for l in leaf_sy]
    inv = {c: {i: v for v, i in m.items()} for c, m in enc.items()}
    for c in enc: S[c] = S[c].astype(int).map(inv[c])
    return S

def run_isolated(fn, *args):
    """Run fn in a forked child: private-pgm (MST) keeps 1.5-2 GB per fit that is never returned to the OS."""
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("fork")) as ex:
        return ex.submit(fn, *args).result()

def _mst_core(train, cat, num, bounds, seed, eps, n, iters_cap=None, domains=None):
    from snsynth import Synthesizer as SNSynth
    from snsynth.transform import TableTransformer, BinTransformer, LabelTransformer
    if iters_cap:  # cap private-pgm mirror-descent iterations (post-processing only: the DP guarantee is unchanged)
        import snsynth.mst.mst as _m
        _md = _m.estimation.mirror_descent
        def _capped(*a, **k):
            k["iters"] = min(k.get("iters", iters_cap), iters_cap); return _md(*a, **k)
        _m.estimation.mirror_descent = _capped
    if CFG.get("jax_cache"):
        import jax
        jax.config.update("jax_compilation_cache_dir", CFG["jax_cache"]); jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    set_seed(seed)
    tt = TableTransformer([BinTransformer(bins=20, lower=bounds[c][0], upper=bounds[c][1]) if c in num else LabelTransformer(nullable=False)
                           for c in train.columns])
    if domains is not None:
        # fit the category encoders on the PUBLIC schema, not on the training data. SmartNoise otherwise reads the category list
        # from the private data, so a rare category that appears only when a given person is present changes the output domain.
        m = max(len(domains[c]) for c in cat); dom = pd.DataFrame({c: ([domains[c][i % len(domains[c])] for i in range(m)] if c in cat else [bounds[c][0]] * m)
                                                                 for c in train.columns})
        tt.fit(dom)
    s = SNSynth.create("mst", epsilon=float(eps), verbose=False)
    s.fit(train, transformer=tt, preprocessor_eps=0.0, categorical_columns=cat, continuous_columns=num)
    out = s.sample(n); rs = np.random.RandomState(seed)
    for c in num:
        w = (bounds[c][1] - bounds[c][0]) / 20
        out[c] = np.clip(out[c].astype(float) + rs.uniform(-w / 2, w / 2, len(out)), *bounds[c])
    return out

def gen_mst(train, meta, seed, n=None, eps=None, iters_cap=None, public_domain=False):
    """MST (McKenna et al.); full budget on measurements. Numeric bounds are public. Category lists are read from the
    training data (SmartNoise's default) unless public_domain=True, which fits them on the public schema instead."""
    return run_isolated(_mst_core, train, meta["cat"], meta["num"], meta["bounds"], seed, float(eps or CFG["mst_eps"]), n or len(train),
                        iters_cap, meta.get("domains") if public_domain else None)

def gen_kanon(train, meta, seed, n=None, k=None):
    """Mondrian k-anonymity with local recoding of quasi-identifiers (non-QI attributes released unchanged)."""
    k = k or CFG["kanon_k"]; qi = meta["qi"]; df = train.copy(); Z = pd.DataFrame(index=df.index)
    for c in qi:
        if c in meta["cat"]:
            order = df[c].value_counts().index; Z[c] = df[c].map({v: i for i, v in enumerate(order)}).astype(float)
        else: Z[c] = df[c].astype(float)
    span = (Z.max() - Z.min()).replace(0, 1); parts, stack = [], [df.index.values]
    while stack:
        idx = stack.pop(); sub = Z.loc[idx]; split = False
        for c in ((sub.max() - sub.min()) / span).sort_values(ascending=False).index:
            med = sub[c].median(); L, R = idx[sub[c].values <= med], idx[sub[c].values > med]
            if len(L) < k or len(R) < k: L, R = idx[sub[c].values < med], idx[sub[c].values >= med]
            if len(L) >= k and len(R) >= k: stack += [L, R]; split = True; break
        if not split: parts.append(idx)
    out = df.copy()
    for idx in parts:
        for c in qi: out.loc[idx, c] = df.loc[idx, c].mode().iloc[0] if c in meta["cat"] else df.loc[idx, c].median()
    return out

class _DDPMNet(torch.nn.Module):
    def __init__(self, d, h=512, T=1000):
        super().__init__(); self.T = T
        self.temb = torch.nn.Sequential(torch.nn.Linear(128, h), torch.nn.SiLU(), torch.nn.Linear(h, h)); self.inp = torch.nn.Linear(d, h)
        self.net = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, h),
                                       torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, d))
        b = torch.linspace(1e-4, 0.02, T); a = 1 - b
        self.register_buffer("beta", b); self.register_buffer("abar", torch.cumprod(a, 0)); self.register_buffer("alpha", a)
    def emb(self, t):
        f = torch.exp(-math.log(10000) * torch.arange(64) / 64); x = t[:, None].float() * f[None]; return torch.cat([torch.sin(x), torch.cos(x)], 1)
    def forward(self, x, t): return self.net(self.inp(x) + self.temb(self.emb(t)))

def gen_ddpm(train, meta, seed, n=None, steps=None):
    """Gaussian DDPM on quantile-normalised numerics + one-hot categoricals (simplified TabDDPM), DDIM sampling (100 steps)."""
    set_seed(seed); n = n or len(train); steps = steps or CFG["ddpm_steps"]; num, cat = meta["num"], meta["cat"]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(train)), random_state=seed)
    Xn = qt.fit_transform(train[num].values) if num else np.zeros((len(train), 0))
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore"); Xc = oh.fit_transform(train[cat]) * 2 - 1
    X = torch.tensor(np.hstack([Xn, Xc]), dtype=torch.float32); d = X.shape[1]
    m = _DDPMNet(d); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); bs = min(512, len(X))
    for _ in range(steps):
        xb = X[torch.randint(0, len(X), (bs,))]; t = torch.randint(0, m.T, (bs,)); e = torch.randn_like(xb)
        ab = m.abar[t][:, None]; loss = ((m(ab.sqrt() * xb + (1 - ab).sqrt() * e, t) - e) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        x = torch.randn(n, d); ts = list(range(m.T - 1, -1, -10))
        for i, t in enumerate(ts):
            ab = m.abar[t]; ab_prev = m.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            eps = m(x, torch.full((n,), t)); x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-6, 6)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    x = x.numpy(); out = pd.DataFrame()
    if num: out[num] = qt.inverse_transform(np.clip(x[:, :len(num)], -5.2, 5.2))
    pos = len(num)
    for c, cats in zip(cat, oh.categories_): out[c] = np.array(cats)[x[:, pos:pos + len(cats)].argmax(1)]; pos += len(cats)
    return out

class _ARNet(torch.nn.Module):
    """Causal transformer over one token per column (GReaT-style autoregressive tabular model, trained from scratch)."""
    def __init__(self, vocab, p, d=64, L=2, H=4):
        super().__init__(); self.tok = torch.nn.Embedding(vocab + 1, d); self.pos = torch.nn.Embedding(p + 1, d)
        layer = torch.nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, L); self.out = torch.nn.Linear(d, vocab); self.p = p
        self.register_buffer("mask", torch.triu(torch.full((p + 1, p + 1), float("-inf")), 1))
    def forward(self, x):  # x: (B, p+1) tokens with BOS at position 0
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.out(self.enc(h, mask=self.mask[: x.shape[1], : x.shape[1]]))

def gen_ar(train, meta, seed, n=None, steps=None, bins=32):
    """Autoregressive transformer: numerics -> 32 quantile bins (decoded by drawing a training value from the bin),
    categoricals -> tokens; p(x) = prod_j p(x_j | x_<j) modelled by a 3-layer causal transformer."""
    set_seed(seed); n = n or len(train); steps = steps or CFG["ar_steps"]; cols = list(train.columns); p = len(cols)
    vocab_of, offs, toks, donors, off = {}, [], [], {}, 0
    for c in cols:
        if c in meta["num"]:
            edges = np.unique(np.quantile(train[c], np.linspace(0, 1, bins + 1)))
            b = np.clip(np.searchsorted(edges[1:-1], train[c].values, side="right"), 0, max(len(edges) - 2, 0))
            vals = np.arange(max(len(edges) - 1, 1)); donors[c] = {k: train[c].values[b == k] for k in vals if (b == k).any()}
            toks.append(b); vocab_of[c] = vals
        else:
            cats = np.array(sorted(train[c].unique())); mp = {v: i for i, v in enumerate(cats)}
            toks.append(train[c].map(mp).values); vocab_of[c] = cats
        offs.append(off); off += len(vocab_of[c])
    V = off; X = np.stack([t + o for t, o in zip(toks, offs)], 1)
    X = torch.tensor(np.hstack([np.full((len(X), 1), V), X]), dtype=torch.long)
    allowed = torch.full((p, V), float("-inf"))
    for j, c in enumerate(cols): allowed[j, offs[j]: offs[j] + len(vocab_of[c])] = 0
    m = _ARNet(V, p); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps); bs = min(256, len(X))
    for _ in range(steps):
        xb = X[torch.randint(0, len(X), (bs,))]; logits = m(xb[:, :-1]) + allowed[None]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval(); rs = np.random.RandomState(seed)
    with torch.no_grad():
        S = torch.full((n, 1), V, dtype=torch.long)
        for j in range(p):
            logits = m(S)[:, -1] + allowed[j]; nxt = torch.multinomial(torch.softmax(logits, -1), 1); S = torch.cat([S, nxt], 1)
    S = S[:, 1:].numpy(); out = pd.DataFrame()
    for j, c in enumerate(cols):
        t = S[:, j] - offs[j]
        if c in meta["num"]:
            out[c] = [donors[c][k][rs.randint(len(donors[c][k]))] if k in donors[c] else np.nan for k in t]
        else: out[c] = vocab_of[c][t]
    return out

GENERATORS = {"Identity": gen_identity, "Independent": gen_independent, "GaussianCopula": gen_gc, "CART": gen_cart,
              "CTGAN": gen_ctgan, "TVAE": gen_tvae, "DDPM": gen_ddpm, "AR-Transformer": gen_ar,
              "MST (ε=1)": gen_mst, "k-anon (k=10)": gen_kanon}
FAMILY = {"Identity": "reference", "Independent": "reference", "GaussianCopula": "statistical", "CART": "trees", "CTGAN": "deep",
          "TVAE": "deep", "DDPM": "deep", "AR-Transformer": "deep", "MST (ε=1)": "DP", "k-anon (k=10)": "anonymisation"}
MAINSTREAM = ["GaussianCopula", "CART", "CTGAN", "TVAE", "DDPM", "AR-Transformer"]

# ---- representation-preserving wrappers ---------------------------------------------------------
def stratify_pool(pool, train, meta, seed, attr):
    """Subsample a large synthetic pool so the protected group's share equals its training share (rejection sampling)."""
    n = len(train); g_real = group_mask(train, meta["sensitive"][attr]).mean()
    g = group_mask(pool, meta["sensitive"][attr]); rs = np.random.RandomState(seed)
    k1 = int(round(n * g_real)); k0 = n - k1
    i1 = np.where(g)[0]; i0 = np.where(~g)[0]
    take1 = rs.choice(i1, min(k1, len(i1)), replace=False) if len(i1) else np.array([], int)
    take0 = rs.choice(i0, n - len(take1), replace=False)
    out = pool.iloc[np.r_[take1, take0]].sample(frac=1, random_state=seed).reset_index(drop=True)
    out.attrs["shortfall"] = k1 - len(take1); return out

def stratified_release(gen, train, meta, seed, attr, pool_mult=20, **kw):
    """Generate a pool pool_mult x the target size, then stratify it (see stratify_pool)."""
    pool = postprocess(gen(train, meta, seed, n=len(train) * pool_mult, **kw), train, meta)
    return stratify_pool(pool, train, meta, seed, attr)

def per_group_release(gen, train, meta, seed, attr, **kw):
    """Fit one generator per protected group and sample each group in proportion to its training share."""
    g = group_mask(train, meta["sensitive"][attr]); parts = []
    for flag in (True, False):
        sub = train[g == flag].reset_index(drop=True)
        parts.append(postprocess(gen(sub, meta, seed, n=len(sub), **kw), sub, meta))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

# =====================================================================================================
# 3. EVALUATION
# =====================================================================================================
class Encoder:
    """Numerics min-max scaled on train; categoricals one-hot / sqrt(2) (a mismatch costs distance 1)."""
    def __init__(self, train, meta):
        self.num, self.cat = meta["num"], meta["cat"]
        self.lo = train[self.num].min(); self.rng = (train[self.num].max() - self.lo).replace(0, 1)
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[self.cat])
    def __call__(self, df):
        return np.hstack([((df[self.num] - self.lo) / self.rng).values, self.oh.transform(df[self.cat]) / np.sqrt(2)]).astype(np.float32)

def _bins(col, k=10):
    e = np.unique(np.quantile(col, np.linspace(0, 1, k + 1))); e[0], e[-1] = -np.inf, np.inf; return e

def _discretize(real, syn, meta):
    R, S = pd.DataFrame(index=real.index), pd.DataFrame(index=syn.index)
    for c in real.columns:
        if c in meta["num"]: e = _bins(real[c]); R[c] = np.digitize(real[c], e[1:-1]); S[c] = np.digitize(syn[c], e[1:-1])
        else: R[c], S[c] = real[c].astype(str), syn[c].astype(str)
    return R, S

def _cramers_v(a, b):
    ct = pd.crosstab(a, b).values
    if min(ct.shape) < 2: return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]; return float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1))))

def fidelity(real, syn, meta, rs=0, holdout=None):
    shape = [1 - stats.ks_2samp(real[c], syn[c]).statistic if c in meta["num"] else
             1 - 0.5 * real[c].value_counts(normalize=True).subtract(syn[c].value_counts(normalize=True), fill_value=0).abs().sum()
             for c in real.columns]
    R, S = _discretize(real, syn, meta); pairs = list(itertools.combinations(real.columns, 2))
    if len(pairs) > 200: pairs = [pairs[i] for i in np.random.RandomState(rs).choice(len(pairs), 200, replace=False)]
    tvd, vr, vs = [], [], []
    for a, b in pairs:
        pr = R.groupby([a, b]).size() / len(R); ps = S.groupby([a, b]).size() / len(S)
        tvd.append(0.5 * pr.subtract(ps, fill_value=0).abs().sum()); vr.append(_cramers_v(R[a], R[b])); vs.append(_cramers_v(S[a], S[b]))
    assoc = float(np.corrcoef(vr, vs)[0, 1]) if np.std(vs) > 0 and np.std(vr) > 0 else 0.0
    ref = holdout if holdout is not None else real
    Z = pd.concat([ref.assign(_y=0), syn.assign(_y=1)], ignore_index=True)
    pre = ColumnTransformer([("n", "passthrough", meta["num"]), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), meta["cat"])])
    clf = make_pipeline(pre, HistGradientBoostingClassifier(max_iter=150, random_state=rs))
    p = cross_val_predict(clf, Z.drop(columns="_y"), Z["_y"], cv=StratifiedKFold(3, shuffle=True, random_state=rs), method="predict_proba")[:, 1]
    return dict(shape=float(np.mean(shape)), pair=float(1 - np.mean(tvd)), assoc_sim=assoc, detect_auc=float(roc_auc_score(Z["_y"], p)))

def diversity(Er, Es, rs=0, k=5, m=3000):
    r = np.random.RandomState(rs)
    Er = Er[r.choice(len(Er), min(m, len(Er)), replace=False)]; Es = Es[r.choice(len(Es), min(m, len(Es)), replace=False)]
    rad_r = NearestNeighbors(n_neighbors=k + 1).fit(Er).kneighbors(Er)[0][:, -1]; rad_s = NearestNeighbors(n_neighbors=k + 1).fit(Es).kneighbors(Es)[0][:, -1]
    D = NearestNeighbors(n_neighbors=min(50, len(Er))).fit(Er).kneighbors(Es); prec = float(np.mean((D[0] <= rad_r[D[1]]).any(1)))
    D2 = NearestNeighbors(n_neighbors=min(50, len(Es))).fit(Es).kneighbors(Er); rec = float(np.mean((D2[0] <= rad_s[D2[1]]).any(1)))
    km = KMeans(20, n_init=3, random_state=rs).fit(Er)
    ss = np.bincount(km.predict(Es), minlength=20) / len(Es); sr = np.bincount(km.labels_, minlength=20) / len(Er)
    return dict(precision=prec, recall=rec, max_mode_syn=float(ss.max()), max_mode_real=float(sr.max()), mode_tvd=float(0.5 * np.abs(ss - sr).sum()))

MODELS = {"LR": lambda s: LogisticRegression(max_iter=2000, C=1.0),
          "RF": lambda s: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=s),
          "HGB": lambda s: HistGradientBoostingClassifier(random_state=s)}

def fit_predict(train_df, test_df, meta, model="HGB", seed=0):
    y = (train_df[meta["target"]] == meta["pos"]).astype(int).values
    X, Xt = train_df.drop(columns=meta["target"]), test_df.drop(columns=meta["target"])
    cat = [c for c in meta["cat"] if c != meta["target"]]
    if len(np.unique(y)) < 2: return np.full(len(Xt), float(y.mean()))
    pre = ColumnTransformer([("n", StandardScaler(), meta["num"]), ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return make_pipeline(pre, MODELS[model](seed)).fit(X, y).predict_proba(Xt)[:, 1]

def utility(train_df, test_df, meta, seed=0, models=("LR", "RF", "HGB")):
    yt = (test_df[meta["target"]] == meta["pos"]).astype(int).values; out, preds = {}, {}
    for m in models:
        p = fit_predict(train_df, test_df, meta, m, seed); preds[m] = p
        out[f"auc_{m}"] = roc_auc_score(yt, p); out[f"ap_{m}"] = average_precision_score(yt, p)
    out["auc_mean"] = float(np.mean([out[f"auc_{m}"] for m in models])); return out, preds

def group_metrics(y, p, g, thr=0.5):
    yh = (p >= thr).astype(int)
    def rates(mask):
        yy, hh = y[mask], yh[mask]
        tpr = hh[yy == 1].mean() if (yy == 1).any() else np.nan; fpr = hh[yy == 0].mean() if (yy == 0).any() else np.nan
        auc = roc_auc_score(yy, p[mask]) if len(np.unique(yy)) == 2 else np.nan
        return hh.mean(), tpr, fpr, auc
    su, tu, fu, au = rates(g); sp, tp, fp, ap = rates(~g)
    return dict(dp_diff=abs(su - sp), eo_diff=np.nanmax([abs(tu - tp), abs(fu - fp)]), auc_unpriv=au, auc_priv=ap)

def fairness(train_like, test_df, pred, meta):
    out = {}; yt = (test_df[meta["target"]] == meta["pos"]).astype(int).values
    for a, spec in meta["sensitive"].items():
        g = group_mask(train_like, spec); y = (train_like[meta["target"]] == meta["pos"]).values
        out[f"{a}|share"] = g.mean()
        out[f"{a}|base_gap"] = (y[~g].mean() if (~g).any() else np.nan) - (y[g].mean() if g.any() else np.nan)
        for k, v in group_metrics(yt, pred, group_mask(test_df, spec)).items(): out[f"{a}|{k}"] = v
    return out

def _tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(f, fpr, tpr))

def tpr_at_randomized(y, s, f=0.01):
    """TPR at FPR = f with a randomised threshold (Neyman-Pearson), so ties cannot lower the value silently.
    Returns (tpr, achieved_fpr_without_randomisation). Members: y == 1."""
    y = np.asarray(y); s = np.asarray(s, float); neg = s[y == 0]; pos = s[y == 1]; n = len(neg); k = f * n
    nd = np.sort(neg)[::-1]; tau = nd[min(int(np.floor(k)), n - 1)]
    a = float((neg > tau).sum()); b = float((neg == tau).sum()); g = 0.0 if b == 0 else min(1.0, max(0.0, (k - a) / b))
    return float((pos > tau).mean() + g * (pos == tau).mean()), a / n

def duplicate_share(train, other):
    """Share of rows in `other` that are exact copies of some row in `train` (such rows cannot be told apart from members)."""
    key = lambda d: d.astype(str).agg("|".join, axis=1)
    return float(key(other).isin(set(key(train))).mean())

def mia_scores(E_T, E_syn, E_ref, rs, k=5):
    """Black-box attacks A1 (distance), A2 (k-NN density ratio, DOMIAS-style), A3 (classifier density ratio)."""
    sc = {"A1_distance": -NearestNeighbors(n_neighbors=1).fit(E_syn).kneighbors(E_T)[0][:, 0]}
    m = min(len(E_syn), len(E_ref)); s2 = np.zeros(len(E_T))
    for _ in range(5):
        Si = E_syn[rs.choice(len(E_syn), m, replace=False)]; Ri = E_ref[rs.choice(len(E_ref), m, replace=False)]
        s2 += np.log(NearestNeighbors(n_neighbors=k).fit(Ri).kneighbors(E_T)[0][:, -1] + 1e-6) - np.log(NearestNeighbors(n_neighbors=k).fit(Si).kneighbors(E_T)[0][:, -1] + 1e-6)
    sc["A2_density_ratio"] = s2 / 5
    Z = np.vstack([E_syn[rs.choice(len(E_syn), m, replace=False)], E_ref[rs.choice(len(E_ref), m, replace=False)]])
    sc["A3_classifier_ratio"] = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Z, np.r_[np.ones(m), np.zeros(m)]).predict_proba(E_T)[:, 1]
    return sc

def privacy(train, hold, ref, syn, meta, enc, seed=0, anonymeter_on=True, group_attr=None):
    rs = np.random.RandomState(seed); n = min(len(hold), len(train))
    mi = rs.choice(len(train), n, replace=False); mem = train.iloc[mi].reset_index(drop=True); non = hold.iloc[:n].reset_index(drop=True)
    E_mem, E_non, E_syn, E_ref, E_tr = enc(mem), enc(non), enc(syn), enc(ref), enc(train)
    d_st = NearestNeighbors(n_neighbors=1).fit(E_tr).kneighbors(E_syn)[0][:, 0]
    d_sh = NearestNeighbors(n_neighbors=1).fit(E_non).kneighbors(E_syn)[0][:, 0]
    d_sm = NearestNeighbors(n_neighbors=1).fit(E_mem).kneighbors(E_syn)[0][:, 0]
    out = dict(exact_match_syn=float(np.mean(d_st < 1e-6)), dcr_share=float(np.mean(d_sm < d_sh) + 0.5 * np.mean(d_sm == d_sh)))
    T = np.vstack([E_mem, E_non]); yT = np.r_[np.ones(n), np.zeros(n)]; sc = mia_scores(T, E_syn, E_ref, rs)
    for a, s in sc.items():
        out[f"mia_auc_{a}"] = roc_auc_score(yT, s); out[f"mia_tpr1_{a}"] = _tpr_at(yT, s, 0.01)
        out[f"mia_p_{a}"] = stats.mannwhitneyu(s[yT == 1], s[yT == 0], alternative="greater").pvalue
        out[f"mia_tpr1_rand_{a}"] = tpr_at_randomized(yT, s, 0.01)[0]
    out["mia_auc_max"] = max(out[f"mia_auc_{a}"] for a in sc); out["mia_tpr1_max"] = max(out[f"mia_tpr1_{a}"] for a in sc)
    out["mia_tpr1_rand_max"] = max(out[f"mia_tpr1_rand_{a}"] for a in sc)
    # permutation null for the strongest-attack statistics on THIS release: shuffle membership labels, repeat (scores do not depend on labels)
    pr = np.random.RandomState(seed + 7); nt, na = [], []
    for _ in range(200):
        yp = yT[pr.permutation(len(yT))]
        nt.append(max(tpr_at_randomized(yp, s, 0.01)[0] for s in sc.values())); na.append(max(roc_auc_score(yp, s) for s in sc.values()))
    out["mia_tpr1_rand_max_null_p95"] = float(np.percentile(nt, 95)); out["mia_auc_max_null_p95"] = float(np.percentile(na, 95))
    o = NearestNeighbors(n_neighbors=5).fit(E_ref).kneighbors(T)[0][:, -1]; top = o >= np.quantile(o, 0.9)
    for tag, mask in [("outlier", top), ("inlier", ~top)]:
        if len(np.unique(yT[mask])) == 2: out[f"mia_auc_{tag}"] = max(roc_auc_score(yT[mask], sc[a][mask]) for a in sc)
        if tag == "outlier" and len(np.unique(yT[mask])) == 2: out["mia_tpr10_outlier"] = max(_tpr_at(yT[mask], sc[a][mask], 0.10) for a in sc)
    if group_attr:  # attack success restricted to protected-group targets
        gT = np.r_[group_mask(mem, meta["sensitive"][group_attr]), group_mask(non, meta["sensitive"][group_attr])]
        for tag, mask in [("protected", gT), ("reference", ~gT)]:
            if len(np.unique(yT[mask])) == 2: out[f"mia_auc_{tag}"] = max(roc_auc_score(yT[mask], sc[a][mask]) for a in sc)
    out["_scores"] = (yT, sc)
    if anonymeter_on: out.update(anonymeter_eval(train, syn, hold, meta, seed))
    return out

def anonymeter_eval(train, syn, hold, meta, seed=0):
    from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
    n_att = int(min(CFG["anon_attacks"], len(hold))); out = {}; np.random.seed(seed)
    qi = meta["qi"]; rest = [c for c in train.columns if c not in qi]
    specs = [("singling", lambda: SinglingOutEvaluator(ori=train, syn=syn, control=hold, n_attacks=n_att, n_cols=3, max_attempts=2_000_000), dict(mode="multivariate")),
             ("link", lambda: LinkabilityEvaluator(ori=train, syn=syn, control=hold, n_attacks=n_att, aux_cols=(qi, rest), n_neighbors=10), dict(n_jobs=2)),
             ("infer", lambda: InferenceEvaluator(ori=train, syn=syn, control=hold, aux_cols=[c for c in train.columns if c != meta["target"]],
                                                   secret=meta["target"], n_attacks=n_att), dict(n_jobs=2))]
    for name, mk, kw in specs:
        try:
            e = mk(); e.evaluate(**kw); r = e.risk(); out[f"anon_{name}"] = float(r.value); out[f"anon_{name}_lo"] = float(r.ci[0])
        except Exception:
            out[f"anon_{name}"] = np.nan; out[f"anon_{name}_lo"] = np.nan
    return out

def measured(fn, *a, **kw):
    """Wall time and CodeCarbon energy (machine mode, Pakistan grid factor)."""
    from codecarbon import OfflineEmissionsTracker
    logging.getLogger("codecarbon").disabled = True
    tr = OfflineEmissionsTracker(country_iso_code="PAK", measure_power_secs=1, log_level="error", save_to_file=False, tracking_mode="machine")
    tr.start(); t0 = time.time()
    try: res = fn(*a, **kw)
    finally: em = tr.stop(); dur = time.time() - t0
    return res, dict(seconds=dur, kwh=float(tr.final_emissions_data.energy_consumed), co2_kg=float(em or 0.0))

# =====================================================================================================
# 4. STRONGER ATTACKS
# =====================================================================================================
def vulnerability(df, meta, k=5, sample=None, seed=0):
    """Meeus et al. (2023)-style record vulnerability: mean distance to the k nearest other records."""
    enc = Encoder(df, meta); E = enc(df)
    d = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E)[0][:, 1:].mean(1)
    return d

def target_features(syn, target, meta, enc, subsets, radii=(0.05, 0.1, 0.25, 0.5)):
    """Features of a synthetic table relative to one target record (TAPAS-style counting queries + distances)."""
    E = enc(syn); et = enc(target.to_frame().T)[0]; d = np.linalg.norm(E - et, axis=1)
    f = [np.mean(d <= r) for r in radii] + [d.min(), np.sort(d)[:5].mean()]
    for sub in subsets:
        mask = np.ones(len(syn), bool)
        for c in sub:
            if c in meta["num"]:
                e = meta["_qbins"][c]; mask &= (np.digitize(syn[c].values, e) == np.digitize(target[c], e))
            else: mask &= (syn[c].values == target[c])
        f.append(mask.mean())
    return np.array(f, float)

def shadow_attack(gen, pool, target, meta, n, n_train_sets, n_eval_sets, seed, subsets, **kw):
    """Record-targeted membership inference with shadow modelling (Stadler et al. 2022; Houssiau et al. 2022):
    fit the generator on datasets that do / do not contain the target, learn a classifier on synthetic-data features,
    and evaluate it on fresh, independently generated releases."""
    rs = np.random.RandomState(seed); enc = Encoder(pool, meta); rows = []
    for i in range(n_train_sets + n_eval_sets):
        base = pool.iloc[rs.choice(len(pool), n - 1, replace=False)]
        member = i % 2 == 0
        extra = target.to_frame().T if member else pool.iloc[[rs.randint(len(pool))]]
        D = pd.concat([base, extra]).reset_index(drop=True)
        for c in meta["num"]: D[c] = D[c].astype(float)
        syn = postprocess(gen(D, meta, seed * 1000 + i, n=n, **kw), D, meta)
        rows.append((target_features(syn, target, meta, enc, subsets), int(member), i < n_train_sets))
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); tr = np.array([r[2] for r in rows])
    clf = RandomForestClassifier(300, min_samples_leaf=2, random_state=seed)
    oof = cross_val_predict(clf, X[tr], y[tr], cv=StratifiedKFold(max(2, min(5, int(np.bincount(y[tr]).min()))), shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    s = clf.fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    bb = -X[~tr, 4]  # black-box baseline on the same releases: negative distance from the target to its nearest synthetic record
    return dict(auc=roc_auc_score(y[~tr], s), acc=float(np.mean((s > 0.5) == y[~tr])), bb_auc=roc_auc_score(y[~tr], bb),
                scores=s, labels=y[~tr], train_scores=oof, train_labels=y[tr], bb_scores=bb)

def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0; hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def eps_lb_at(scores, labels, thr, delta=1e-5, alpha=0.05):
    """Empirical epsilon lower bound at a threshold fixed in advance (Clopper-Pearson, both directions)."""
    P, N = int(labels.sum()), int((1 - labels).sum()); pred = scores >= thr
    fpr_hi = clopper_pearson(int((pred & (labels == 0)).sum()), N, alpha)[1]; fnr_hi = clopper_pearson(int((~pred & (labels == 1)).sum()), P, alpha)[1]
    vals = [math.log(a / b) for a, b in [(1 - delta - fpr_hi, fnr_hi), (1 - delta - fnr_hi, fpr_hi)] if a > 0 and b > 0]
    return max(vals + [0.0])

def eps_lower_bound(scores, labels, delta=1e-5, alpha=0.05):
    """Empirical DP lower bound (Jagielski et al. 2020; Nasr et al. 2021): max over thresholds of
    log((1 - delta - FPR_hi) / FNR_hi) and log((1 - delta - FNR_hi) / FPR_hi) with Clopper-Pearson bounds.
    NB: choosing the threshold on the same data makes this optimistic for the attacker; reported as indicative."""
    best = 0.0; P, N = int(labels.sum()), int((1 - labels).sum())
    for t in np.unique(scores):
        pred = scores >= t; fp = int((pred & (labels == 0)).sum()); fn = int((~pred & (labels == 1)).sum())
        fpr_hi = clopper_pearson(fp, N, alpha)[1]; fnr_hi = clopper_pearson(fn, P, alpha)[1]
        for num, den in [(1 - delta - fpr_hi, fnr_hi), (1 - delta - fnr_hi, fpr_hi)]:
            if num > 0 and den > 0: best = max(best, math.log(num / den))
    return best

def reconstruction_attack(syn, users, meta, secret, qcols, n_queries=3000, seed=0, qbins=None):
    """Linear reconstruction of a binary secret for a known set of users (Annamalai et al. 2024):
    answer random 3-way conjunction queries with the synthetic data, solve A s ~ b with 0 <= s <= 1, threshold at 0.5."""
    from scipy.optimize import lsq_linear
    rs = np.random.RandomState(seed); n = len(users)
    def disc(df):
        out = {}
        for c in qcols:
            out[c] = np.digitize(df[c].values, qbins[c]) if c in meta["num"] else df[c].values.astype(str)
        return out
    U, S = disc(users), disc(syn); sv = (syn[secret[0]].values == secret[1]).astype(float)
    A, b = [], []
    for _ in range(n_queries):
        cols = rs.choice(qcols, 3, replace=False); j = rs.randint(n)
        mu = np.ones(n, bool); ms = np.ones(len(syn), bool)
        for c in cols: v = U[c][j]; mu &= U[c] == v; ms &= S[c] == v
        if mu.sum() < 2: continue
        A.append(mu.astype(float)); b.append(sv[ms].mean() * mu.sum() if ms.any() else 0.5 * mu.sum())
    A = np.array(A); b = np.array(b)
    sol = lsq_linear(A, b, bounds=(0, 1), lsmr_tol="auto", max_iter=200).x
    return (sol > 0.5).astype(int), len(b)

# =====================================================================================================
# 5. GATES AND AUDIT
# =====================================================================================================
ORIGINAL_GATES = {"fidelity": "assoc_sim > 0.90", "privacy": "exact copies < 0.1% and DCR share <= 0.55",
                  "utility": "TSTR/TRTR >= 0.90", "diversity": "largest mode <= 30%", "fairness": "EO difference <= 0.05"}
PROPOSED_GATES = {"attack": "strongest MIA TPR@1%FPR <= 2x chance overall and TPR@10%FPR <= 2x chance on top-10% outliers",
                  "anonymeter": "all three Anonymeter 95% CI lower bounds <= 0",
                  "representation": "protected-group share ratio in [0.8, 1.25]",
                  "outcome_gap": "base-rate gap within +/-0.05 of the real gap",
                  "utility": "TSTR/TRTR >= task requirement (default 0.90) across 3 model families"}

def original_gates(r, sensitive):
    eo = [r.get(f"{a}|eo_diff", np.nan) for a in sensitive]
    return dict(g_fidelity=r["assoc_sim"] > 0.90, g_privacy=(r["exact_match_syn"] < 0.001) and (r["dcr_share"] <= 0.55),
                g_utility=r["util_ratio"] >= 0.90, g_diversity=r["max_mode_syn"] <= 0.30,
                g_fairness=all((e <= 0.05) for e in eo if not np.isnan(e)))

def proposed_gates(r, sensitive, util_req=0.90):
    rep = [r.get(f"{a}|share", np.nan) / r.get(f"{a}|share_real", np.nan) for a in sensitive]
    gap = [abs(r.get(f"{a}|base_gap", np.nan) - r.get(f"{a}|base_gap_real", np.nan)) for a in sensitive]
    return dict(p_attack=(r["mia_tpr1_max"] <= 0.02) and (r.get("mia_tpr10_outlier", 0.0) <= 0.20),
                p_anonymeter=all((r.get(f"anon_{k}_lo", 0) or 0) <= 0 for k in ("singling", "link", "infer")),
                p_representation=all(0.8 <= x <= 1.25 for x in rep if not np.isnan(x)),
                p_outcome_gap=all(x <= 0.05 for x in gap if not np.isnan(x)),
                p_utility=r["util_ratio"] >= util_req)

def evaluate_release(train, hold, test, syn, meta, seed=0, real=None, anonymeter_on=True, group_attr=None):
    """Full evaluation of one release. `real` = (utility dict, fairness dict) for the real-data baseline."""
    enc = Encoder(train, meta)
    if real is None:
        u_r, p_r = utility(train, test, meta, seed); real = (u_r, fairness(train, test, p_r["HGB"], meta))
    u, preds = utility(syn, test, meta, seed); r = {**fidelity(train, syn, meta, seed, holdout=hold), **diversity(enc(train), enc(syn), seed), **u}
    r.update(fairness(syn, test, preds["HGB"], meta)); pv = privacy(train, hold, test, syn, meta, enc, seed, anonymeter_on, group_attr)
    scores = pv.pop("_scores"); r.update(pv)
    r["util_ratio"] = u["auc_mean"] / real[0]["auc_mean"]
    for a in meta["sensitive"]:
        for k in ("share", "base_gap", "eo_diff", "auc_unpriv", "auc_priv"): r[f"{a}|{k}_real"] = real[1][f"{a}|{k}"]
    return r, scores

def audit(train, hold, test, syn, meta, seed=0, util_req=0.90, include_outcome_gap=False):
    """Basic screening audit: all metrics plus a screening flag (pass/fail) on the original and the proposed gates.
    A pass means that none of these attacks found leakage; it does not show that no leakage exists. The targeted (shadow-model) and
    canary attacks are NOT part of this call. The outcome-gap gate failed on new data (H9) and is excluded from release_ok_proposed
    unless include_outcome_gap=True. `tpr_gate_resolvable` is False when 1% of the holdout is fewer than 20 records or when
    duplicates cap TPR@1% below 0.05, in which case the TPR@1% gate cannot be trusted."""
    r, _ = evaluate_release(train, hold, test, syn, meta, seed)
    r.update(original_gates(r, meta["sensitive"])); r.update(proposed_gates(r, meta["sensitive"], util_req))
    dup = duplicate_share(train, hold); ceil = min(1.0, 0.01 / dup) if dup > 0 else 1.0
    r["holdout_duplicate_share"] = dup; r["tpr1_ceiling"] = ceil
    r["tpr_gate_resolvable"] = bool(0.01 * len(hold) >= 20 and ceil >= 0.05)
    r["release_ok_original"] = all(r[k] for k in ("g_fidelity", "g_privacy", "g_utility", "g_diversity", "g_fairness"))
    keys = ["p_attack", "p_anonymeter", "p_representation", "p_utility"] + (["p_outcome_gap"] if include_outcome_gap else [])
    r["release_ok_proposed"] = all(r[k] for k in keys)
    return r

# =====================================================================================================
# 6. STATISTICS
# =====================================================================================================
def ci_t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

def boot_ci(x, B=10000, seed=0, groups=None):
    rs = np.random.RandomState(seed); x = np.asarray(x, float)
    if groups is None: groups = np.zeros(len(x))
    parts = [x[groups == g] for g in np.unique(groups)]
    means = [np.mean(np.concatenate([p[rs.randint(0, len(p), len(p))] for p in parts])) for _ in range(B)]
    return float(np.nanmean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj

def spearman_boot(x, y, B=5000, seed=0):
    rs = np.random.RandomState(seed); x, y = np.asarray(x), np.asarray(y); r0 = stats.spearmanr(x, y)[0]
    bs = [stats.spearmanr(x[i], y[i])[0] for i in (rs.randint(0, len(x), len(x)) for _ in range(B))]
    return float(r0), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

# =====================================================================================================
# 7. SELF-TEST
# =====================================================================================================
def self_test(df, meta, seed=0):
    """Positive control (release the training data) and negative control (fresh records) must behave as expected."""
    tr, ho, te = make_split(df, meta, seed)
    used = pd.concat([tr, ho, te]); fresh = df.merge(used.drop_duplicates(), how="left", indicator=True)
    fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
    if len(fresh) >= len(tr): fresh = fresh.sample(len(tr), random_state=seed).reset_index(drop=True)
    else:  # small datasets use every record: split the training part in two, one half as members, the other as the fresh sample
        h = len(tr) // 2; fresh = tr.iloc[h:2 * h].reset_index(drop=True); tr = tr.iloc[:h].reset_index(drop=True); ho = ho.iloc[:h].reset_index(drop=True)
    enc = Encoder(tr, meta)
    pos = privacy(tr, ho, te, tr, meta, enc, seed, anonymeter_on=False); neg = privacy(tr, ho, te, fresh, meta, enc, seed, anonymeter_on=False)
    n1 = n2 = min(len(tr), len(ho)); sa_ = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2)); sd = 0.5 / math.sqrt(len(tr))
    # pass criteria scaled to sample size: positive control > 5 null SDs above chance, negative control within 3 null SDs
    ok = (pos["mia_auc_A1_distance"] > 0.5 + 5 * sa_ and pos["dcr_share"] > 0.5 + 5 * sd
          and abs(neg["mia_auc_A1_distance"] - 0.5) < 3 * sa_ and abs(neg["dcr_share"] - 0.5) < 3 * sd)
    pos["null_sd_auc"] = sa_; neg["null_sd_auc"] = sa_
    return ok, {k: v for k, v in pos.items() if not k.startswith("_") and "_p_" not in k}, {k: v for k, v in neg.items() if not k.startswith("_") and "_p_" not in k}
