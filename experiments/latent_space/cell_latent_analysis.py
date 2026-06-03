# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Latent space characterisation for Section 2.4 (Chris Bakal approach, 2026-05-27).

Strategy: describe each patient in a new feature space rather than leading with p-values.

For each component (C3=prototype_2, C5=prototype_4):

  2a. Soft labels — logistic regression P(MYCN-amp) per tile.
       Positive = tile looks MYCN-amp; near 0.5 = ambiguous; near 0 = non-amp.

  2b. UMAP of tile features — 2D embedding, coloured by class and by soft label.

  2c. Per-slide soft label violin — 20 slides (10 non-amp, 10 MYCN-amp), ordered
       non-amp left / MYCN-amp right. Shows patient-level phenotypic signature.

  2d. Patient PCA — per-slide mean feature vector (92-dim) projected to 2 PCs.
       Each dot = one patient; coloured by class. Shows latent patient separation.

Outputs (results/ subfolder):
  component3_umap_by_class.pdf
  component3_umap_by_softlabel.pdf
  component3_softlabel_violin.pdf
  component3_slide_pca.pdf
  component3_soft_labels.csv
  component5_*.pdf (same set)
  component5_soft_labels.csv
  summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn"), "olga_refactered")
CSVS = {
    "component3": os.path.join(
        BASE, "results/slide_inference/fold_9/images/prototype_2/features/cell_info_updated.csv"
    ),
    "component5": os.path.join(
        BASE, "results/slide_inference/fold_9/images/prototype_4/features/cell_info_updated.csv"
    ),
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

NAN_THRESH  = 0.50
COLOR_AMP   = "#D94F3D"
COLOR_NOAMP = "#4F74C8"

try:
    import umap
    HAS_UMAP = True
    print("umap-learn available — using UMAP")
except ImportError:
    HAS_UMAP = False
    print("umap-learn not available — falling back to PCA for 2-D embedding")


def parse_meta(index_series):
    split     = index_series.str.split("|", n=1, expand=True)
    class_str = split[0]
    slide_idx = split[1].str.split("_x_", n=1).str[0]
    # composite ID ensures non-amp slide 3 != MYCN-amp slide 3
    slide_id  = class_str + "_" + slide_idx
    return class_str, slide_id


def clean_label(name, max_len=40):
    for p in ["neuroblast cells: ", "immune cells: ", "necrosis cells: ", "neuroblast cells "]:
        if name.lower().startswith(p.lower()):
            name = name[len(p):]
            break
    name = name.replace("mean of their ", "").strip()
    return name[:max_len - 1] + "…" if len(name) > max_len else name


summary_lines = []

for comp_name, csv_path in CSVS.items():
    print(f"\n{'='*60}\n{comp_name}\n{'='*60}")

    df = pd.read_csv(csv_path, index_col=0)
    class_str, slide_idx = parse_meta(pd.Series(df.index))
    df["_class"]  = class_str.values
    df["_slide"]  = slide_idx.values
    df["_label"]  = (df["_class"] == "class_1").astype(int)  # 1=MYCN-amp

    meta = ["_class", "_slide", "_label"]
    feat_cols = [c for c in df.columns if c not in meta
                 and pd.api.types.is_numeric_dtype(df[c])]

    amp_m    = df["_label"] == 1
    nonamp_m = df["_label"] == 0
    feat_cols = [c for c in feat_cols
                 if df.loc[amp_m, c].isna().mean() <= NAN_THRESH
                 and df.loc[nonamp_m, c].isna().mean() <= NAN_THRESH]
    feat_cols = [c for c in feat_cols if df[c].std() > 1e-9]

    print(f"  Features: {len(feat_cols)}  | non-amp: {nonamp_m.sum()}, MYCN-amp: {amp_m.sum()}")

    X_raw  = df[feat_cols].values.astype(float)
    y      = df["_label"].values
    slides = df["_slide"].values

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    # ── 2a. Logistic regression + soft labels ─────────────────────────────────
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs")
    lr.fit(X_sc, y)
    soft_labels = lr.predict_proba(X_sc)[:, 1]   # P(MYCN-amp)
    print(f"  LR train accuracy: {lr.score(X_sc, y):.3f}")
    print(f"  Soft label range: [{soft_labels.min():.3f}, {soft_labels.max():.3f}]")

    # save per-tile soft labels
    sl_df = pd.DataFrame({
        "tile_index": df.index,
        "slide":      slides,
        "true_label": y,
        "class":      df["_class"].values,
        "soft_label_mycn_amp": soft_labels,
    })
    sl_df.to_csv(os.path.join(OUT_DIR, f"{comp_name}_soft_labels.csv"), index=False)

    # ── 2b. 2D embedding (UMAP or PCA) ────────────────────────────────────────
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42, verbose=False)
        embed   = reducer.fit_transform(X_sc)
        embed_label = "UMAP"
    else:
        pca2   = PCA(n_components=2, random_state=42)
        embed  = pca2.fit_transform(X_sc)
        embed_label = "PCA"

    print(f"  {embed_label} embedding computed: {embed.shape}")

    # figure: coloured by class
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, colour, label in [(0, COLOR_NOAMP, "non-amp"), (1, COLOR_AMP, "MYCN-amp")]:
        m = y == cls
        ax.scatter(embed[m, 0], embed[m, 1], c=colour, s=6, alpha=0.4,
                   linewidths=0, label=label, rasterized=True)
    ax.set_xlabel(f"{embed_label} 1", fontsize=9)
    ax.set_ylabel(f"{embed_label} 2", fontsize=9)
    ax.legend(fontsize=8, markerscale=2)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_umap_by_class.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {comp_name}_umap_by_class.pdf")

    # figure: coloured by soft label (continuous)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(embed[:, 0], embed[:, 1], c=soft_labels, cmap="RdBu_r",
                    vmin=0, vmax=1, s=6, alpha=0.5, linewidths=0, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0 (non-amp)", "0.5", "1 (MYCN-amp)"])
    ax.set_xlabel(f"{embed_label} 1", fontsize=9)
    ax.set_ylabel(f"{embed_label} 2", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_umap_by_softlabel.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {comp_name}_umap_by_softlabel.pdf")

    # ── 2c. Per-slide soft label violin ───────────────────────────────────────
    unique_slides = df["_slide"].unique()
    slide_class   = {s: int(df.loc[df["_slide"] == s, "_label"].iloc[0])
                     for s in unique_slides}

    # order: non-amp slides left, MYCN-amp slides right; sort by numeric suffix
    noamp_slides = sorted([s for s in unique_slides if slide_class[s] == 0],
                          key=lambda s: int(s.split("_")[-1]))
    amp_slides   = sorted([s for s in unique_slides if slide_class[s] == 1],
                          key=lambda s: int(s.split("_")[-1]))
    ordered_slides = noamp_slides + amp_slides
    n_noamp = len(noamp_slides)
    n_amp   = len(amp_slides)

    slide_data   = [soft_labels[df["_slide"].values == s] for s in ordered_slides]
    slide_colors = [COLOR_NOAMP if slide_class[s] == 0 else COLOR_AMP for s in ordered_slides]

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_slides) * 0.55), 4.5))
    parts = ax.violinplot(slide_data, positions=range(len(ordered_slides)),
                          showmedians=True, showextrema=False, widths=0.7)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(slide_colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)

    # dividing line between classes
    ax.axvline(n_noamp - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    # x-tick labels: "S0", "S1", … coloured by class
    ax.set_xticks(range(len(ordered_slides)))
    tick_labels = [f"N{s.split('_')[-1]}" if slide_class[s] == 0
                   else f"A{s.split('_')[-1]}" for s in ordered_slides]
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    for tick, s in zip(ax.get_xticklabels(), ordered_slides):
        tick.set_color(COLOR_NOAMP if slide_class[s] == 0 else COLOR_AMP)

    ax.set_ylabel("P(MYCN-amp) soft label", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)

    ax.legend(handles=[
        mpatches.Patch(color=COLOR_NOAMP, alpha=0.7, label="non-amp"),
        mpatches.Patch(color=COLOR_AMP,   alpha=0.7, label="MYCN-amp"),
    ], fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_softlabel_violin.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {comp_name}_softlabel_violin.pdf")

    # per-slide soft label stats for summary
    slide_medians = {s: float(np.median(soft_labels[df["_slide"].values == s]))
                     for s in ordered_slides}
    noamp_med = np.mean([slide_medians[s] for s in noamp_slides])
    amp_med   = np.mean([slide_medians[s] for s in amp_slides])
    print(f"  Mean per-slide median soft label: non-amp={noamp_med:.3f}, MYCN-amp={amp_med:.3f}")

    # ── 2d. Patient-level PCA (slide feature profiles) ────────────────────────
    # Per-slide mean of each scaled feature → 92-dim slide vector
    slide_vecs  = []
    slide_labels_pca = []
    slide_ids   = []
    for s in ordered_slides:
        m = df["_slide"].values == s
        slide_vecs.append(X_sc[m].mean(axis=0))
        slide_labels_pca.append(slide_class[s])
        slide_ids.append(s)

    S = np.stack(slide_vecs)   # (20, n_feats)
    pca_slide = PCA(n_components=2, random_state=42)
    S_pc      = pca_slide.fit_transform(S)
    var_exp   = pca_slide.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for cls, colour, label in [(0, COLOR_NOAMP, "non-amp"), (1, COLOR_AMP, "MYCN-amp")]:
        m = np.array(slide_labels_pca) == cls
        ax.scatter(S_pc[m, 0], S_pc[m, 1], c=colour, s=80, edgecolors="white",
                   linewidths=0.5, label=label, zorder=3)

    # label each slide dot
    for i, (sid, slabel) in enumerate(zip(slide_ids, slide_labels_pca)):
        tick_lbl = f"N{sid.split('_')[-1]}" if slabel == 0 else f"A{sid.split('_')[-1]}"
        ax.annotate(tick_lbl, (S_pc[i, 0], S_pc[i, 1]),
                    fontsize=6, ha="center", va="bottom",
                    color=COLOR_NOAMP if slabel == 0 else COLOR_AMP,
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_slide_pca.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {comp_name}_slide_pca.pdf")

    # ── summary ───────────────────────────────────────────────────────────────
    summary_lines.append(f"\n{comp_name.upper()} — Latent Space Characterisation")
    summary_lines.append(f"  Tiles: non-amp={int(nonamp_m.sum())}, MYCN-amp={int(amp_m.sum())}")
    summary_lines.append(f"  Slides: non-amp={n_noamp}, MYCN-amp={n_amp}")
    summary_lines.append(f"  LR train accuracy: {lr.score(X_sc, y):.3f}")
    summary_lines.append(f"  Soft label — mean per-slide median: non-amp={noamp_med:.3f}, MYCN-amp={amp_med:.3f}")
    summary_lines.append(f"  Slide PCA: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}% variance explained")
    summary_lines.append(f"  Embedding method: {embed_label}")


# ── write summary ─────────────────────────────────────────────────────────────
header = [
    "=" * 70,
    "LATENT SPACE CHARACTERISATION — Chris Bakal approach (2026-05-27)",
    "Soft label = P(MYCN-amp) from logistic regression (C=0.1, balanced)",
    "UMAP/PCA of tile features; per-slide violin; patient PCA",
    "=" * 70,
]
summary = "\n".join(header + summary_lines)
print("\n" + summary)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary + "\n")

print(f"\nAll outputs written to: {OUT_DIR}")
