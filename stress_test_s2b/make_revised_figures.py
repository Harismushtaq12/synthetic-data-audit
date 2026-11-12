"""Regenerate the Phase II / stress-test figures with clearer labels (same data as before)."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "..", "PartII_notebooks", "outputs2", "tables") + "/"
OUT = os.path.join(HERE, "figs") + "/"; os.makedirs(OUT, exist_ok=True)
S2B = os.path.join(HERE, "S2b_addremove_audit.csv")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "axes.titlelocation": "left", "legend.frameon": False})
INK, INK2, MUTED, RED, BLUE, GREEN = "#0b0b0b", "#52514e", "#8a8984", "#b8342f", "#2a78d6", "#1baf7a"
COL = {"CART": "#2a78d6", "DDPM": "#eb6834", "TVAE": "#1baf7a", "CTGAN": "#eda100", "GaussianCopula": "#e87ba4",
       "MST (ε=1)": "#008300", "k-anon (k=10)": "#4a3aa7", "Independent": "#e34948", "AR-Transformer": "#0b0b0b", "Identity": MUTED}
MK = {"CART": "o", "DDPM": "s", "TVAE": "^", "CTGAN": "D", "GaussianCopula": "v", "MST (ε=1)": "P", "k-anon (k=10)": "X", "Independent": "h", "AR-Transformer": "*", "Identity": "."}
LAB = {"GaussianCopula": "Gaussian copula", "MST (ε=1)": "MST (ε = 1)", "k-anon (k=10)": "k-anon. (k = 10)"}
lab = lambda g: LAB.get(g, g)
def save(fig, name): fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=220); plt.close(fig)

# ---- validation: positive control vs the three attacks on the negative control ----
V = pd.read_csv(T + "X0_selftests.csv")
fig, ax = plt.subplots(figsize=(6.6, 2.5)); x = np.arange(len(V)); w = 0.2
ax.bar(x - 1.5 * w, V.pos_A1_auc, w, color=RED, label="released training data (should be caught)")
for k, (c, hatch) in enumerate([("neg_A1_auc", ""), ("neg_A2_auc", "////"), ("neg_A3_auc", "....")]):
    ax.bar(x + (k - 0.5) * w, V[c], w, color="#c9c8c4", edgecolor=INK2, linewidth=0.4, hatch=hatch,
           label=["unseen real data: distance attack", "unseen real data: density ratio", "unseen real data: classifier"][k])
ax.errorbar(x + 0.5 * w, np.full(len(V), 0.5), yerr=3 * V.null_sd_auc, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
ax.axhline(0.5, color="k", lw=0.8); ax.set_ylim(0.4, 1.02); ax.set_xticks(x); ax.set_xticklabels(V.dataset.str.replace("Diabetes130", "Diabetes"))
ax.set_ylabel("attack AUC"); ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=7)
save(fig, "fig_validation")

# ---- S1: 100 targets ----
R1 = pd.read_csv(T + "R1_targets100.csv")
order = ["Independent", "GaussianCopula", "CART", "TVAE", "CTGAN"]
short = {"Independent": "Indep.", "GaussianCopula": "Gauss.\ncopula", "CART": "CART", "TVAE": "TVAE", "CTGAN": "CTGAN"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]; rs = np.random.RandomState(0)
for i, g in enumerate(order):
    for kind, c, dx in [("random", MUTED, -0.18), ("vulnerable", RED, 0.18)]:
        s = R1[(R1.generator == g) & (R1.kind == kind)]
        ax.scatter(i + dx + rs.uniform(-0.08, 0.08, len(s)), s.auc, s=7, color=c, alpha=0.75, lw=0)
        ax.plot([i + dx - 0.12, i + dx + 0.12], [s.auc.mean()] * 2, color=INK, lw=2)
ax.axhline(0.5, color=INK2, lw=0.7); ax.set_xticks(range(len(order))); ax.set_xticklabels([short[g] for g in order], fontsize=7)
ax.set_ylabel("attack AUC per person"); ax.set_title("(a) random (grey) and vulnerable (red)")
ax = axes[1]
for g in order:
    s = R1[R1.generator == g]; ax.scatter(s.bb_auc, s.auc, s=9, color=COL[g], marker=MK[g], alpha=0.7, lw=0, label=lab(g))
ax.plot([0.2, 1], [0.2, 1], color=INK2, lw=0.7); ax.set_xlim(0.25, 1.0); ax.set_ylim(0.25, 1.0)
ax.set_xlabel("distance attack AUC"); ax.set_ylabel("shadow-model attack AUC"); ax.set_title("(b) same people, same releases")
fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=7, handletextpad=0.1, markerscale=1.5)
fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1)); save(fig, "fig_r1_targets")

# ---- S2: canary audit (a) replace-one audit with a trained attack, (b) add/remove audit with a fixed dataset ----
R2 = pd.read_csv(T + "R2_canary_audit.csv")
labs = ["MST ε=1\npublic", "MST ε=10\npublic", "MST ε=100\npublic", "MST ε=1\ndata lists", "CART", "Indep."]
cols = [GREEN, GREEN, GREEN, RED, MUTED, MUTED]
has_b = os.path.exists(S2B)
fig, axes = plt.subplots(1, 2 if has_b else 1, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.6, 1]} if has_b else None)
ax = axes[0] if has_b else axes
x = np.arange(len(R2)); ax.bar(x, R2.eps_lb, 0.55, color=cols)
for i, r in R2.iterrows(): ax.text(i, r.eps_lb + 0.08, f"{r.eps_lb:.2f}", ha="center", fontsize=7)
ax.plot([2.67, 3.33], [1, 1], color="k", lw=1.2, ls="--")
ax.axhline(R2.eps_lb_ceiling.iloc[0], color=INK2, lw=0.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel("certified ε lower bound"); ax.set_ylim(0, 4.6)
ax.set_title("(a) trained attack, 80 + 80 releases")
ax.plot([], [], color="k", ls="--", label="promised ε = 1"); ax.plot([], [], color=INK2, ls=":", label="highest value this audit can certify")
ax.legend(loc="upper left", fontsize=6.5)
if has_b:
    B = pd.read_csv(S2B); B = B.sort_values(["variant", "eps_nominal"], ascending=[True, False])
    ax = axes[1]; xb = np.arange(len(B))
    ax.bar(xb, B.eps_lb, 0.55, color=[RED if v == "default" else GREEN for v in B.variant])
    ax.scatter(xb, B.eps_lb_ceiling, marker="_", s=200, color=INK2, zorder=3)
    for i, (_, r) in enumerate(B.iterrows()):
        if r.eps_lb > 0.6: ax.text(i, r.eps_lb / 2, f"{r.eps_lb:.2f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        else: ax.text(i, r.eps_lb + 0.08, f"{r.eps_lb:.2f}", ha="center", fontsize=7)
        ax.plot([i - 0.33, i + 0.33], [r.eps_nominal] * 2, color="k", lw=1.2, ls="--")
    ax.set_xticks(xb); ax.set_xticklabels([f"ε={r.eps_nominal:g}\n{'data lists' if r.variant == 'default' else 'public'}" for _, r in B.iterrows()], fontsize=6.5)
    ax.set_ylim(0, 4.6); ax.set_title("(b) add/remove, category test")
fig.tight_layout(w_pad=1.2); save(fig, "fig_canary")

# ---- S3: transformer training length (two stacked panels per dataset, log x) ----
R3 = pd.read_csv(T + "R3_ar_small.csv"); g = R3.groupby(["dataset", "steps"]).mean(numeric_only=True).reset_index()
fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
for j, ds in enumerate(["German", "Pima"]):
    s = g[g.dataset == ds]
    a = axes[0, j]; a.plot(s.steps, s.tpr1 / 0.01, "o-", color=RED); a.set_ylim(0, 65); a.set_title(ds)
    a.set_ylabel("attack TPR@1%\n(× chance)" if j == 0 else "")
    b = axes[1, j]; b.plot(s.steps, s.util_ratio, "s-", color=BLUE); b.set_ylim(0.9, 1.0)
    b.set_ylabel("utility ratio U" if j == 0 else ""); b.set_xscale("log"); b.set_xticks([150, 300, 750, 1500]); b.set_xticklabels(["150", "300", "750", "1500"])
    b.minorticks_off(); b.set_xlabel("training steps (log scale)")
fig.tight_layout(h_pad=0.6, w_pad=1.5); save(fig, "fig_ar_steps")

# ---- tuning ----
TU = pd.read_csv(T + "X3_tuning.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, col, yl in [(axes[0], "mia_auc_max", "attack AUC (all people)"), (axes[1], "mia_auc_outlier", "attack AUC (outliers)")]:
    for gname, s in TU.groupby("generator", sort=False):
        a = s.groupby("value", sort=False)[["pair", col]].mean()
        ax.plot(a.pair, a[col], "-", color=COL.get(gname, INK), lw=1, alpha=0.8)
        ax.scatter(a.pair, a[col], marker=MK.get(gname, "o"), color=COL.get(gname, INK), s=18, edgecolor="white", linewidth=0.4, zorder=3)
        d = s[s.is_default].groupby("value")[["pair", col]].mean()
        ax.scatter(d.pair, d[col], marker="o", facecolor="none", edgecolor=INK, s=60, lw=0.8, zorder=4)
    ax.set_xlabel("pairwise fidelity"); ax.set_ylabel(yl); ax.axhline(0.5, color=INK2, lw=0.7)
axes[0].set_title("(a) all people"); axes[1].set_title("(b) outliers")
gs = list(TU.generator.unique())
h = [Line2D([], [], marker=MK.get(x, "o"), color=COL.get(x, INK), ls="-", ms=5, label=lab(x)) for x in gs] + [Line2D([], [], ls="", marker="o", mfc="none", mec=INK, ms=7, label="default setting")]
fig.legend(handles=h, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
fig.tight_layout(w_pad=2, rect=(0, 0.08, 1, 1)); save(fig, "fig_x3_tuning")

# ---- repair ----
X4 = pd.read_csv(T + "X4_representation.csv"); VAR = ["plain", "stratified", "per-group"]; VC = {"plain": MUTED, "stratified": BLUE, "per-group": GREEN}
gens4 = ["TVAE", "CTGAN", "DDPM", "CART"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
ax = axes[0]
for i, gname in enumerate(gens4):
    for j, v in enumerate(VAR):
        s = X4[(X4.generator == gname) & (X4.variant == v)]
        ax.scatter(np.full(len(s), i + (j - 1) * 0.22), s.repr_ratio, color=VC[v], s=14, zorder=3, marker=["o", "s", "^"][j])
ax.axhspan(0.8, 1.25, color="#eef4fc", zorder=0); ax.axhline(1, color=INK2, lw=0.7); ax.set_yscale("log")
ax.set_xticks(range(len(gens4))); ax.set_xticklabels(gens4); ax.set_ylabel("group share ratio"); ax.set_title("(a) group share")
ax = axes[1]
pl = X4[X4.variant == "plain"][["dataset", "seed", "generator", "mia_auc_protected"]]
for j, v in enumerate(VAR[1:]):
    s = X4[X4.variant == v].merge(pl, on=["dataset", "seed", "generator"], suffixes=("", "_pl"))
    for i, gname in enumerate(gens4):
        ss = s[s.generator == gname]; d = ss.mia_auc_protected - ss.mia_auc_protected_pl
        ax.scatter(np.full(len(d), i + (j - 0.5) * 0.3), d, color=VC[v], s=14, marker=["s", "^"][j], zorder=3)
        ax.plot([i + (j - 0.5) * 0.3 - 0.1, i + (j - 0.5) * 0.3 + 0.1], [d.mean()] * 2, color=VC[v], lw=2)
ax.axhline(0, color=INK2, lw=0.7); ax.set_xticks(range(len(gens4))); ax.set_xticklabels(gens4); ax.set_ylabel("change in group attack AUC"); ax.set_title("(b) privacy cost")
h = [Line2D([], [], ls="", marker=m_, color=VC[v], label=v) for v, m_ in zip(VAR, ["o", "s", "^"])]
fig.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08)); fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1)); save(fig, "fig_x4_repair")
print("figures written; S2b panel:", has_b)
