# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
03_cell_feature_survival.py — slide-level cell-feature × survival visualisation.

The master `cell_info.csv` (6,160 tiles × 351 cell-level features × 136 slides)
is aggregated per slide (mean of all tiles within the slide), giving 136
slide-level cell-feature vectors. All 136 slides are present in the survival
cohort. We then:

  - UMAP on the 136 × 351 matrix, coloured by MYCN, OS time, event, timepoint.
  - Scatter plots of the per-slide mean of the four key features highlighted
    in Section 2.4's LME analysis (closeness centrality, clustering coefficient,
    Simpson diversity, necrosis-cell count) versus OS time, coloured by event.

NB. This is exploratory: it broadens Section 2.4 from the 40 component-specific
slides (anonymised in the existing pipeline) to all 136 slides with cell-level
features. No survival model is fitted.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap

BASE   = os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn")
SURV   = os.path.join(BASE, "additional_exp/survival_analysis")
COHORT = os.path.join(SURV, "data/survival_per_slide.csv")
TILE   = os.path.join(BASE, "olga_refactered/results/cell_analysis/cell_info.csv")
GMM    = os.path.join(BASE, "additional_exp/gmm_responsibility/results/per_slide_stats.csv")
RES    = os.path.join(SURV, "results/cell_feature_maps")
os.makedirs(RES, exist_ok=True)

C_AMP = "#D94F3D"
C_NON = "#4F74C8"

# ─── 1. Load cohort + master cell file ───────────────────────────────────── #
surv = pd.read_csv(COHORT).dropna(subset=["OS_time_days", "event"]).copy()
surv["OS_time_months"] = surv.OS_time_days.astype(float) / 30.44
surv["event"] = surv.event.astype(int)
print(f"Survival cohort: {len(surv)} slides")

tile = pd.read_csv(TILE, index_col=0)
tile["slide"] = tile.index.str.split("|").str[0]
print(f"Master cell file: {len(tile)} tiles × {tile.shape[1]-1} feats × "
      f"{tile.slide.nunique()} slides")

# ─── 2. Aggregate per slide — mean of every numeric feature ──────────────── #
num_cols = [c for c in tile.columns if c != "slide" and pd.api.types.is_numeric_dtype(tile[c])]
slide_feats = tile.groupby("slide")[num_cols].mean()
slide_feats.index.name = "slide_name"
print(f"Per-slide aggregated features: {slide_feats.shape}")

# ─── 3. Join cell features ⨝ survival ────────────────────────────────────── #
df = surv.merge(slide_feats, left_on="slide_name", right_index=True, how="inner")
print(f"After join with cell features: {len(df)} slides "
      f"({df.mycn_perslide.value_counts().to_dict()})")

# Also pull C3 / C5 per-slide mean responsibilities for stratification
g = pd.read_csv(GMM)[["slide", "mean_c3", "mean_c5"]]
df = df.merge(g, left_on="slide_name", right_on="slide", how="left").drop(columns="slide")
print(f"Tagged C3 / C5 responsibilities for {df[['mean_c3','mean_c5']].notna().all(axis=1).sum()} slides")

# ─── 4. UMAP of the 351-D cell-feature space ─────────────────────────────── #
X = df[num_cols].copy()
# columns where everything is NaN drop entirely; impute the rest
X = X.dropna(axis=1, how="all")
imp = SimpleImputer(strategy="median").fit_transform(X)
Xs  = StandardScaler().fit_transform(imp)
print(f"Cell-feature matrix for embedding: {Xs.shape}")

um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Xs)
ts = TSNE(n_components=2, perplexity=min(30, len(Xs)//4), random_state=42,
          init="pca").fit_transform(Xs)

def four_panel(emb, title_prefix, save_prefix):
    """Title-stripped 4-panel embedding: MYCN / OS time / event × MYCN / timepoint."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    ax = axes[0]
    for lbl, c in [(0, C_NON), (1, C_AMP)]:
        m = df.mycn_perslide == lbl
        ax.scatter(emb[m, 0], emb[m, 1], c=c, s=28, alpha=0.8,
                   edgecolor="white", linewidth=0.4,
                   label="MYCN non-amp" if lbl == 0 else "MYCN-amp")
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    ax = axes[1]
    norm = Normalize(vmin=df.OS_time_months.min(), vmax=df.OS_time_months.quantile(0.95))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=df.OS_time_months, cmap="viridis",
                    norm=norm, s=28, alpha=0.85, edgecolor="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="OS (months)")
    ax = axes[2]
    for ev, marker in [(1, "o"), (0, "^")]:
        for lbl, c in [(0, C_NON), (1, C_AMP)]:
            m = (df.event == ev) & (df.mycn_perslide == lbl)
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], c=c, s=28, alpha=0.8, marker=marker,
                       edgecolor="white", linewidth=0.4,
                       label=f"{'deceased' if ev else 'censored'} | "
                             f"{'amp' if lbl else 'non-amp'}")
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=7)
    ax = axes[3]
    for tp, c in [("primary", "#2E8B57"), ("relapse", "#8B2E5E")]:
        m = df.timepoint == tp
        ax.scatter(emb[m, 0], emb[m, 1], c=c, s=28, alpha=0.8,
                   edgecolor="white", linewidth=0.4, label=tp)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, save_prefix + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(RES, save_prefix + ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_prefix}.pdf/png")

print("\nRendering 4-panel UMAPs / t-SNE...")
four_panel(um, "Slide-level cell features (UMAP)",  "umap_cell_features")
four_panel(ts, "Slide-level cell features (t-SNE)", "tsne_cell_features")

# ─── 5. LME-highlighted features vs OS — scatter ─────────────────────────── #
KEY_FEATURES = {
    "Closeness centrality (mean)": "mean of cells' closeness_centrality",
    "Clustering coefficient (mean)": "mean of cells' clustering_coefficient",
    "Simpson diversity (global)": "Gloabl Simpson index",   # NB original typo preserved
    "Necrosis cell count": "number of necrosis cells",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
norm = Normalize(vmin=df.OS_time_months.min(), vmax=df.OS_time_months.quantile(0.95))
for ax, (label, col) in zip(axes.flat, KEY_FEATURES.items()):
    if col not in df.columns:
        ax.set_visible(False); continue
    for ev, marker, name in [(1, "o", "deceased"), (0, "^", "censored")]:
        for mycn, c, mn in [(0, C_NON, "non-amp"), (1, C_AMP, "amp")]:
            m = (df.event == ev) & (df.mycn_perslide == mycn)
            if m.sum() == 0:
                continue
            ax.scatter(df.loc[m, col], df.loc[m, "OS_time_months"],
                       c=c, marker=marker, s=42, alpha=0.78,
                       edgecolor="black", linewidth=0.4,
                       label=f"{name} | MYCN-{mn}")
    ax.set_xlabel(label)
    ax.set_ylabel("OS (months)")
    ax.grid(alpha=0.25)
    ax.set_title(label, fontsize=10)
# legend deduped on first axis
axes[0, 0].legend(loc="best", fontsize=7, frameon=True, framealpha=0.9)
plt.suptitle("Section 2.4 LME features vs overall survival (slide-level)",
             fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(RES, "lme_features_vs_os.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "lme_features_vs_os.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  → lme_features_vs_os.pdf/png")

# ─── 6. C3 / C5 dominant slide stratification + KM (descriptive) ─────────── #
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# stratify by which of C3 vs C5 has the higher per-slide mean responsibility
df["dominant_phenotype"] = np.where(df.mean_c3 >= df.mean_c5, "C3-dominant", "C5-dominant")
c3d = df[df.dominant_phenotype == "C3-dominant"]
c5d = df[df.dominant_phenotype == "C5-dominant"]
lr = logrank_test(c3d.OS_time_months, c5d.OS_time_months,
                  event_observed_A=c3d.event, event_observed_B=c5d.event)
fig, ax = plt.subplots(figsize=(6, 5))
kmf = KaplanMeierFitter()
kmf.fit(c3d.OS_time_months, c3d.event,
        label=f"C3-dominant (n={len(c3d)}, ev={int(c3d.event.sum())})")
kmf.plot_survival_function(ax=ax, ci_show=True, color="#268bd2")
kmf.fit(c5d.OS_time_months, c5d.event,
        label=f"C5-dominant (n={len(c5d)}, ev={int(c5d.event.sum())})")
kmf.plot_survival_function(ax=ax, ci_show=True, color="#dc322f")
ax.set_xlabel("Overall survival (months)")
ax.set_ylabel("Survival probability")
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"KM by C3- vs C5-dominant slide phenotype\n"
             f"log-rank p = {lr.p_value:.3g}  (descriptive — slides not independent)")
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RES, "km_by_c3_vs_c5_dominant.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "km_by_c3_vs_c5_dominant.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  → km_by_c3_vs_c5_dominant.pdf/png  log-rank p={lr.p_value:.3g}")

# ─── 7. Save coordinates ─────────────────────────────────────────────────── #
out = pd.DataFrame({
    "slide_name": df.slide_name,
    "patient_id": df.patient_id,
    "mycn_perslide": df.mycn_perslide,
    "OS_time_months": df.OS_time_months,
    "event": df.event,
    "timepoint": df.timepoint,
    "mean_c3": df.mean_c3, "mean_c5": df.mean_c5,
    "dominant_phenotype": df.dominant_phenotype,
    "umap_cell_x": um[:, 0], "umap_cell_y": um[:, 1],
    "tsne_cell_x": ts[:, 0], "tsne_cell_y": ts[:, 1],
})
out.to_csv(os.path.join(RES, "cell_feature_coords.csv"), index=False)
print("Wrote cell_feature_coords.csv")
print("\nDone.")
