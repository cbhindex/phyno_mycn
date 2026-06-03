# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
02_slide_embedding_survival.py — slide-level embedding × survival visualisation.

Builds three slide-level representations and projects each to 2-D:
  1. Mean UNI            — average of UNI tile embeddings per slide (1024-D).
  2. Attention-weighted  — Σ_i a_i · u_i using fold_9 attention (1024-D).
  3. GMM responsibility  — 6-D mean per-slide GMM responsibilities (Sec 2.2).

Each 2-D map is rendered four times, coloured by:
  - MYCN per-slide
  - OS time (continuous)
  - Event status
  - Timepoint (primary vs relapse)

Also produces:
  - Direct C3 vs C5 mean-responsibility scatter coloured by OS time.
  - Optional KM by unsupervised KMeans cluster on the UNI-attn embedding.

Outputs go to results/slide_embedding_maps/.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE = os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn")
SURV = os.path.join(BASE, "additional_exp/survival_analysis")
COHORT = os.path.join(SURV, "data/survival_per_slide.csv")
UNI_DIR = os.path.join(BASE, "olga_refactered/data/wsi_embeddings/uni_feats/pt_files")
OUT_DIR = os.path.join(BASE, "olga_refactered/results/slide_inference/fold_9/pt_outputs")
GMM_CSV = os.path.join(BASE,
    "additional_exp/gmm_responsibility/results/per_slide_stats.csv")
RES = os.path.join(SURV, "results/slide_embedding_maps")
os.makedirs(RES, exist_ok=True)

C_AMP = "#D94F3D"
C_NON = "#4F74C8"

# ─── 1. Load cohort ───────────────────────────────────────────────────────── #
df = pd.read_csv(COHORT).dropna(subset=["OS_time_days", "event"]).reset_index(drop=True)
df["OS_time_days"] = df["OS_time_days"].astype(float)
df["event"] = df["event"].astype(int)
df["OS_time_months"] = df["OS_time_days"] / 30.44
print(f"Cohort: {len(df)} slides")

# ─── 2. Build slide-level UNI representations ─────────────────────────────── #
def build_uni_reps(df):
    mean_vecs = np.zeros((len(df), 1024), dtype=np.float32)
    attn_vecs = np.zeros((len(df), 1024), dtype=np.float32)
    for i, row in df.iterrows():
        uni  = torch.load(os.path.join(UNI_DIR, row.slide_name + ".pt"),
                          map_location="cpu", weights_only=False)
        att  = torch.load(os.path.join(OUT_DIR, row.slide_name + "_att.pt"),
                          map_location="cpu", weights_only=False).squeeze(0)
        # safety: trim/align if tile counts mismatch
        n = min(uni.shape[0], att.shape[0])
        uni, att = uni[:n], att[:n]
        mean_vecs[i] = uni.mean(dim=0).numpy()
        attn_vecs[i] = (att.unsqueeze(1) * uni).sum(dim=0).numpy()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} slides processed")
    return mean_vecs, attn_vecs

cache = os.path.join(SURV, "data/uni_slide_reps.npz")
if os.path.exists(cache):
    z = np.load(cache)
    mean_vecs, attn_vecs = z["mean_vecs"], z["attn_vecs"]
    print(f"Loaded cached UNI reps: mean={mean_vecs.shape}, attn={attn_vecs.shape}")
else:
    print("Building UNI slide-level representations (~1–2 min)...")
    mean_vecs, attn_vecs = build_uni_reps(df)
    np.savez_compressed(cache, mean_vecs=mean_vecs, attn_vecs=attn_vecs)
    print(f"Cached → {cache}")

# ─── 3. GMM 6-D responsibility per slide ──────────────────────────────────── #
gmm = pd.read_csv(GMM_CSV)
gmm_cols = ["mean_c1", "mean_c2", "mean_c3", "mean_c4", "mean_c5", "mean_c6"]
g_map = gmm.set_index("slide")[gmm_cols]
gmm_vecs = df.slide_name.map(lambda s: g_map.loc[s].values).tolist()
gmm_vecs = np.vstack(gmm_vecs).astype(np.float32)
print(f"GMM responsibilities: {gmm_vecs.shape}")

# ─── 4. 2-D embeddings (UMAP, t-SNE) ──────────────────────────────────────── #
def embed(X, name):
    Xs = StandardScaler().fit_transform(X)
    print(f"  UMAP on {name}: input {Xs.shape}")
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Xs)
    perp = min(30, max(5, len(Xs) // 4))
    print(f"  t-SNE on {name}: perplexity={perp}")
    ts = TSNE(n_components=2, perplexity=perp, random_state=42,
              init="pca").fit_transform(Xs)
    return um, ts

print("Embedding mean UNI...");        um_uni_mean, ts_uni_mean = embed(mean_vecs, "mean-UNI")
print("Embedding attention-UNI...");   um_uni_attn, ts_uni_attn = embed(attn_vecs, "attn-UNI")
print("Embedding GMM 6-D...");         um_gmm,      ts_gmm      = embed(gmm_vecs,  "GMM-6D")

# ─── 5. Plotting utilities ────────────────────────────────────────────────── #
def four_panel(emb, title_prefix, save_prefix):
    """Render an embedding 4× coloured by MYCN, OS-time, event, timepoint.

    Title-free: per-panel content discriminator is shown as an axis xlabel-like
    tag rather than a title so the panel composites cleanly in figure assembly.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    # 5a — MYCN
    ax = axes[0]
    for lbl, c in [(0, C_NON), (1, C_AMP)]:
        m = df.mycn_perslide == lbl
        ax.scatter(emb[m, 0], emb[m, 1], c=c, s=22, alpha=0.78,
                   edgecolor="white", linewidth=0.4,
                   label="MYCN non-amp" if lbl == 0 else "MYCN-amp")
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    # 5b — OS time continuous
    ax = axes[1]
    norm = Normalize(vmin=df.OS_time_months.min(), vmax=df.OS_time_months.quantile(0.95))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=df.OS_time_months, cmap="viridis",
                    norm=norm, s=22, alpha=0.85, edgecolor="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="OS (months)")
    # 5c — event status × MYCN as shape
    ax = axes[2]
    for ev, marker in [(1, "o"), (0, "^")]:
        for lbl, c in [(0, C_NON), (1, C_AMP)]:
            m = (df.event == ev) & (df.mycn_perslide == lbl)
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], c=c, s=22, alpha=0.78, marker=marker,
                       edgecolor="white", linewidth=0.4,
                       label=f"{'deceased' if ev else 'censored'} | "
                             f"{'amp' if lbl else 'non-amp'}")
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=7)
    # 5d — timepoint
    ax = axes[3]
    for tp, c in [("primary", "#2E8B57"), ("relapse", "#8B2E5E")]:
        m = df.timepoint == tp
        ax.scatter(emb[m, 0], emb[m, 1], c=c, s=22, alpha=0.78,
                   edgecolor="white", linewidth=0.4,
                   label=tp)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, save_prefix + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(RES, save_prefix + ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_prefix}.pdf/png")

print("\nRendering panels...")
four_panel(um_uni_mean, "Mean UNI (UMAP)",       "umap_uni_mean")
four_panel(ts_uni_mean, "Mean UNI (t-SNE)",      "tsne_uni_mean")
four_panel(um_uni_attn, "Attn UNI (UMAP)",       "umap_uni_attn")
four_panel(ts_uni_attn, "Attn UNI (t-SNE)",      "tsne_uni_attn")
four_panel(um_gmm,      "GMM 6-D (UMAP)",        "umap_gmm")
four_panel(ts_gmm,      "GMM 6-D (t-SNE)",       "tsne_gmm")

# ─── 6. C3 vs C5 scatter ──────────────────────────────────────────────────── #
fig, ax = plt.subplots(figsize=(6, 5.5))
norm = Normalize(vmin=df.OS_time_months.min(), vmax=df.OS_time_months.quantile(0.95))
for lbl, marker, name in [(1, "o", "MYCN-amp"), (0, "^", "non-amp")]:
    m = df.mycn_perslide == lbl
    sc = ax.scatter(gmm_vecs[m, 2], gmm_vecs[m, 4],
                    c=df.loc[m, "OS_time_months"], cmap="viridis", norm=norm,
                    s=55, alpha=0.85, marker=marker,
                    edgecolor="black", linewidth=0.5, label=name)
plt.colorbar(sc, ax=ax, label="OS (months)")
ax.set_xlabel("Mean C3 responsibility (per slide)")
ax.set_ylabel("Mean C5 responsibility (per slide)")
ax.set_title("Phenotype-axis: C3 × C5 × OS")
ax.legend(loc="best", frameon=True, framealpha=0.9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RES, "scatter_c3_c5_survival.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "scatter_c3_c5_survival.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  → scatter_c3_c5_survival.pdf/png")

# ─── 7. KM by unsupervised KMeans cluster ─────────────────────────────────── #
# Cluster on attn-UNI embedding to test if visual structure carries prognostic
# information. Use silhouette over K=2..4 to pick a sensible K.
from sklearn.metrics import silhouette_score
best_k, best_s = 2, -1
for k in (2, 3, 4):
    lbl = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(um_uni_attn)
    if len(set(lbl)) < 2:
        continue
    s = silhouette_score(um_uni_attn, lbl)
    if s > best_s:
        best_k, best_s = k, s
print(f"\nUnsupervised KMeans on UMAP(Attn-UNI): best K={best_k} (silhouette={best_s:.3f})")
clusters = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(um_uni_attn)
df["cluster"] = clusters

# Plot cluster + KM
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
cmap = plt.get_cmap("tab10")
for c in range(best_k):
    m = clusters == c
    ax.scatter(um_uni_attn[m, 0], um_uni_attn[m, 1], c=[cmap(c)], s=24, alpha=0.85,
               edgecolor="white", linewidth=0.4, label=f"C{c} (n={m.sum()})")
ax.set_title(f"Attn-UNI UMAP — KMeans K={best_k}")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(); ax.grid(alpha=0.2)

ax = axes[1]
kmf = KaplanMeierFitter()
for c in range(best_k):
    sub = df[df.cluster == c]
    kmf.fit(sub.OS_time_months, sub.event,
            label=f"Cluster {c} (n={len(sub)}, ev={int(sub.event.sum())})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=cmap(c))
# pairwise log-rank, or global multivariate
from lifelines.statistics import multivariate_logrank_test
lr = multivariate_logrank_test(df.OS_time_months, df.cluster, df.event)
ax.set_title(f"KM by attn-UNI cluster — log-rank p = {lr.p_value:.3g} (exploratory)")
ax.set_xlabel("Overall survival (months)")
ax.set_ylabel("Survival probability")
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RES, "km_by_attnuni_cluster.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "km_by_attnuni_cluster.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  → km_by_attnuni_cluster.pdf/png  (multivariate log-rank p={lr.p_value:.3g})")

# ─── 8. Save embedded coords + cluster assignments ────────────────────────── #
emb_df = pd.DataFrame({
    "slide_name": df.slide_name,
    "patient_id": df.patient_id,
    "mycn_perslide": df.mycn_perslide,
    "OS_time_months": df.OS_time_months,
    "event": df.event,
    "timepoint": df.timepoint,
    "umap_uni_mean_x": um_uni_mean[:, 0], "umap_uni_mean_y": um_uni_mean[:, 1],
    "umap_uni_attn_x": um_uni_attn[:, 0], "umap_uni_attn_y": um_uni_attn[:, 1],
    "umap_gmm_x":      um_gmm[:, 0],      "umap_gmm_y":      um_gmm[:, 1],
    "cluster_attnuni": df.cluster,
})
emb_df.to_csv(os.path.join(RES, "embedding_coords.csv"), index=False)
print("Wrote embedding_coords.csv")
print("\nDone.")
