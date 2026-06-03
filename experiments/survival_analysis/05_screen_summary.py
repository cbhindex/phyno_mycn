# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
05_screen_summary.py — single-panel screen summary for Figure 6 Panel b.

Reads the 12 descriptive log-rank p-values from
  results/km_curves/km_by_gmm_median_split.csv
and renders a ranked horizontal bar chart of −log10(p), with:
  - bars coloured by component number (C1..C6),
  - hatching to distinguish `mean` vs `dom` statistic,
  - a dashed vertical line at the p = 0.05 reference (−log10 = 1.30).

This panel visually communicates the screen: "we tested all six components
across two statistics; only the MYCN-amp-enriched minor components rise above
the reference line."
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn"), "additional_exp/survival_analysis")
CSV  = os.path.join(BASE, "results/km_curves/km_by_gmm_median_split.csv")
OUT  = os.path.join(BASE, "results/km_curves")

# Pathology-review-derived component descriptions, aligned with Fig 3d.
# Short forms kept compact so bar-chart labels remain readable.
DESC = {
    1: "normal-like",
    2: "necrotic",
    3: "cellular tumour",
    4: "artefact",
    5: "dense tumour",
    6: "haemorrhagic",
}
COMP_COLOR = {
    1: "#9e9ac8",
    2: "#d94f3d",
    3: "#4a7ab8",
    4: "#9aa19a",
    5: "#3aa68d",
    6: "#e6a23c",
}

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} screen tests")

df["comp"]    = df.feature.str.extract(r"_c(\d+)$").astype(int)
df["stat"]    = df.feature.str.extract(r"^(mean|dom)").iloc[:, 0]
df["nlog10p"] = -np.log10(df.logrank_p.clip(lower=1e-12))
df["label"]   = df.apply(
    lambda r: f"C{int(r.comp)} ({DESC[int(r.comp)]}) — {r.stat}", axis=1,
)
df = df.sort_values("nlog10p", ascending=True).reset_index(drop=True)

# Rendering
fig, ax = plt.subplots(figsize=(7.8, 5.4))
ypos = np.arange(len(df))
colors  = [COMP_COLOR[c] for c in df.comp]
hatches = ["" if s == "mean" else "//" for s in df.stat]

for y, val, col, hat, plabel, p in zip(ypos, df.nlog10p, colors, hatches,
                                       df.label, df.logrank_p):
    ax.barh(y, val, color=col, edgecolor="black", linewidth=0.6,
            hatch=hat, alpha=0.92)
    # numeric p-value at bar tip
    ax.text(val + 0.05, y, f"p = {p:.3g}",
            va="center", ha="left", fontsize=9, color="#444")

# p = 0.05 reference
ref = -np.log10(0.05)
ax.axvline(ref, color="#999", linestyle="--", linewidth=1)
ax.text(ref + 0.04, -0.6, "p = 0.05", color="#555",
        fontsize=9, va="center", ha="left")

ax.set_yticks(ypos)
ax.set_yticklabels(df.label, fontsize=10)
ax.set_xlabel(r"$-\log_{10}$(log-rank p)")
ax.set_xlim(0, df.nlog10p.max() * 1.20 + 0.4)

# legend for hatching
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="white", edgecolor="black", label="mean responsibility"),
    Patch(facecolor="white", edgecolor="black", hatch="//",
          label="dominant tile fraction"),
]
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95,
          fontsize=9)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "screen_summary_logrank.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "screen_summary_logrank.png"), dpi=300,
            bbox_inches="tight")
plt.close()
print(f"Wrote {OUT}/screen_summary_logrank.pdf/png")
