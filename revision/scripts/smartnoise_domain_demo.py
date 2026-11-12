"""Minimal demonstration (SmartNoise-synth 1.0.8, MST) that category lists are read from the PRIVATE data unless a transformer
is fitted on a public schema. D has no record with the rare value 'rare'; D' = D + one record that has it.
Counts how often 'rare' appears in releases of D and of D'. With data-derived lists (default) it appears only for D'
(an (eps,delta)-DP mechanism cannot do this once the probability exceeds delta). With a public schema it can appear for both.
Usage: python smartnoise_domain_demo.py [n_releases]   (needs smartnoise-synth==1.0.8; ~1 s per fit)."""
import sys, numpy as np, pandas as pd
from snsynth import Synthesizer
from snsynth.transform import TableTransformer, LabelTransformer

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EPS = 1.0
rs = np.random.RandomState(0)
n = 499
common = {"a": ["x", "y", "z"], "b": ["u", "v"], "c": ["p", "q", "r"]}
D = pd.DataFrame({k: rs.choice(v, n) for k, v in common.items()})
D["d"] = rs.choice(["c1", "c2", "c3", "c4"], n)                      # column that will carry the rare value
Dp = pd.concat([D, pd.DataFrame([{"a": "x", "b": "u", "c": "p", "d": "rare"}])], ignore_index=True)
cols = list(D.columns)
public = pd.DataFrame({"a": common["a"] + ["x"] * 3, "b": common["b"] + ["u"] * 4, "c": common["c"] + ["p"] * 3,
                       "d": ["c1", "c2", "c3", "c4", "rare", "c1"]})   # public schema: the full list of allowed values

def release(data, seed, use_public):
    np.random.seed(seed)
    tt = TableTransformer([LabelTransformer(nullable=False) for _ in cols])
    if use_public: tt.fit(public)
    s = Synthesizer.create("mst", epsilon=EPS, verbose=False)
    s.fit(data, transformer=tt, preprocessor_eps=0.0, categorical_columns=cols)
    return s.sample(len(data))

for use_public in (False, True):
    hit = {"D": 0, "Dprime": 0}
    for r in range(R):
        hit["D"] += int((release(D, r, use_public)["d"] == "rare").any())
        hit["Dprime"] += int((release(Dp, r, use_public)["d"] == "rare").any())
    print(f"{'public schema' if use_public else 'data-derived lists (default)':32s} 'rare' appears in {hit['D']}/{R} releases of D and {hit['Dprime']}/{R} releases of D'")
