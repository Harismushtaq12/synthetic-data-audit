import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
B = pd.read_csv(os.path.join(HERE, "S2b_addremove_audit.csv"))
B = B[B.releases_per_world > 2]
rows = []
for _, r in B.iterrows():
    lists = "from private data" if r.variant == "default" else "public schema"
    rows.append(f"{lists} & {r.eps_nominal:g} & {int(r.releases_per_world)} & {int(r.rare_in_D)}/{int(r.releases_per_world)} & "
                f"{int(r.rare_in_Dprime)}/{int(r.releases_per_world)} & {r.eps_lb:.2f} & {r.eps_lb_ceiling:.2f}\\\\")
tex = r"""\begin{table}[t]\centering\small
\caption{Add/remove canary audit of SmartNoise MST (stress test S2, second version).}\label{tab:s2b}
\begin{tabular}{lcccccc}\toprule
Category lists & Promised $\varepsilon$ & Releases & Rare in $M(D)$ & Rare in $M(D')$ & $\varepsilon_{\text{LB}}$ & Ceiling\\\midrule
""" + "\n".join(rows) + r"""
\botrule\end{tabular}
\par\smallskip{\footnotesize $D$: 499 Adult records without ``Married-AF-spouse'' or ``Preschool''; $D'=D$ plus the canary. Releases: per side. Test: does the release contain either value? $\delta=10^{-5}$; one-sided 97.5\% Clopper--Pearson limits.}
\end{table}
"""
open(os.path.join(HERE, "tab_s2b.tex"), "w").write(tex.replace("\\botrule", "\\bottomrule"))
print(tex)
