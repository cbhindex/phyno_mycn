# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
04_km_gmm_components.py — Kaplan–Meier on per-slide GMM-component features.

For each of the 6 GMM components, runs a median-split KM on both:
  - mean_cK  (mean responsibility, per slide)
  - dom_cK   (dominant tile fraction, per slide)
All 12 stand-alone KM plots are written so the per-component screen is
complete (used by Supplementary Figure S6a). Also writes:
  - km_by_gmm_median_split.csv  — 12 descriptive log-rank p values
  - km_mycn_x_dom_c6.{pdf,png}  — 4-strata KM for the MYCN × C6 confounding view
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

BASE   = os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn")
SURV   = os.path.join(BASE, "additional_exp/survival_analysis")
COHORT = os.path.join(SURV, "data/survival_per_slide.csv")
GMM    = os.path.join(BASE, "additional_exp/gmm_responsibility/results/per_slide_stats.csv")
OUT    = os.path.join(SURV, "results/km_curves")
os.makedirs(OUT, exist_ok=True)

surv = pd.read_csv(COHORT)
g    = pd.read_csv(GMM)
df = surv.merge(
    g[["slide"] + [f"mean_c{i}" for i in range(1, 7)]
                 + [f"dom_c{i}"  for i in range(1, 7)]],
    left_on="slide_name", right_on="slide"
).dropna(subset=["OS_time_days", "event"])
df["OS_time_months"] = df.OS_time_days.astype(float) / 30.44
df["event"] = df.event.astype(int)
print(f"N = {len(df)}")

# ─── Median-split KM across all GMM features ─────────────────────────────── #
records = []
print(f"\n{'feature':<10} {'low_n':>5} {'high_n':>6} {'low_ev':>6} {'high_ev':>7} {'p':>10}")
for k in range(1, 7):
    for kind in ("mean", "dom"):
        col = f"{kind}_c{k}"
        if df[col].nunique() < 4:
            continue
        med = df[col].median()
        low  = df[df[col] <= med]
        high = df[df[col] >  med]
        if len(low) < 5 or len(high) < 5:
            continue
        lr = logrank_test(low.OS_time_months, high.OS_time_months,
                          event_observed_A=low.event, event_observed_B=high.event)
        records.append({"feature": col,
                        "low_n": len(low), "high_n": len(high),
                        "low_ev": int(low.event.sum()),
                        "high_ev": int(high.event.sum()),
                        "logrank_p": lr.p_value})
        print(f"{col:<10} {len(low):>5} {len(high):>6} {int(low.event.sum()):>6} "
              f"{int(high.event.sum()):>7} {lr.p_value:>10.3g}")
pd.DataFrame(records).to_csv(os.path.join(OUT, "km_by_gmm_median_split.csv"), index=False)

# ─── KM plots for ALL six components × {mean, dom}  (12 panels) ──────────── #
# Component labels aligned with Fig 3d "GMM component-to-pathology mapping".
COMPONENT_DESCRIPTIONS = {
    1: "normal-like / Schwannian",
    2: "necrotic",
    3: "cellular tumour morphology",
    4: "artefact",
    5: "dense tumour morphology",
    6: "haemorrhagic / erythrocyte-rich",
}

def plot_split(col, ttl, fname):
    med = df[col].median()
    low  = df[df[col] <= med]
    high = df[df[col] >  med]
    if low[col].nunique() == 0 or high[col].nunique() == 0:
        print(f"  skip {col}: degenerate median split")
        return
    lr = logrank_test(low.OS_time_months, high.OS_time_months,
                      event_observed_A=low.event, event_observed_B=high.event)
    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(low.OS_time_months, low.event,
            label=f"low {col} (n={len(low)}, ev={int(low.event.sum())})")
    kmf.plot_survival_function(ax=ax, color="#268bd2", ci_show=True)
    kmf.fit(high.OS_time_months, high.event,
            label=f"high {col} (n={len(high)}, ev={int(high.event.sum())})")
    kmf.plot_survival_function(ax=ax, color="#dc322f", ci_show=True)
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, fname + ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}.pdf/png  p={lr.p_value:.3g}")

print("\nRendering KM plots for all 6 components × {mean, dom}:")
for k in range(1, 7):
    desc = COMPONENT_DESCRIPTIONS[k]
    plot_split(f"dom_c{k}",  f"Dominant-tile fraction for C{k} ({desc})",
               f"km_by_dom_c{k}")
    plot_split(f"mean_c{k}", f"Mean responsibility for C{k} ({desc})",
               f"km_by_mean_c{k}")

# ─── MYCN-stratified KM for dom_c6 (within-stratum check) ────────────────── #
print("\nMYCN-stratified KM for dom_c6 (within each MYCN class):")
for mycn in (0, 1):
    sub = df[df.mycn_perslide == mycn]
    med = sub.dom_c6.median()
    low  = sub[sub.dom_c6 <= med]
    high = sub[sub.dom_c6 >  med]
    if len(low) < 5 or len(high) < 5:
        continue
    lr = logrank_test(low.OS_time_months, high.OS_time_months,
                      event_observed_A=low.event, event_observed_B=high.event)
    print(f"  MYCN={'amp' if mycn else 'non-amp'}: "
          f"n_low={len(low)} n_high={len(high)} log-rank p={lr.p_value:.3g}")

# ─── 4-strata KM (MYCN × dom_c6) ─────────────────────────────────────────── #
med = df.dom_c6.median()
df["strata"] = (
    "MYCN-" + df.mycn_perslide.map({0: "non", 1: "amp"})
    + " | C6-" + (df.dom_c6 > med).map({True: "hi", False: "lo"})
)
fig, ax = plt.subplots(figsize=(7, 5.5))
kmf = KaplanMeierFitter()
cmap = {"MYCN-non | C6-lo": "#a3b9d2", "MYCN-non | C6-hi": "#4F74C8",
        "MYCN-amp | C6-lo": "#f2a497", "MYCN-amp | C6-hi": "#D94F3D"}
for s, c in cmap.items():
    sub = df[df.strata == s]
    if len(sub) == 0:
        continue
    kmf.fit(sub.OS_time_months, sub.event,
            label=f"{s} (n={len(sub)}, ev={int(sub.event.sum())})")
    kmf.plot_survival_function(ax=ax, color=c, ci_show=False)
lr = multivariate_logrank_test(df.OS_time_months, df.strata, df.event)
ax.set_xlabel("Overall survival (months)")
ax.set_ylabel("Survival probability")
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "km_mycn_x_dom_c6.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "km_mycn_x_dom_c6.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  → km_mycn_x_dom_c6.pdf/png  multivariate log-rank p={lr.p_value:.3g}")
print("\nDone.")
