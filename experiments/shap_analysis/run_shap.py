# Pheno-MYCN — additional experiments.
# Author:  Dr Olga Fourkioti  (https://github.com/olgarithmics)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
SHAP analysis.

Convention:
  class_0 = 0 = non-amp
  class_1 = 1 = MYCN-amp

Logistic regression with class_weight='balanced' (handles tile count imbalance).
SHAP LinearExplainer for exact SHAP values.

Positive SHAP → pushes toward MYCN-amp prediction (class 1).
Negative SHAP → pushes toward non-amp prediction (class 0).

Outputs (results/):
  component{3,5}_shap_heatmap.pdf     — tiles × top-features SHAP heatmap
  component{3,5}_shap_beeswarm.pdf    — feature importance with direction
  component{3,5}_shap_values.npy      — SHAP values array [n_tiles × n_feats]
  component{3,5}_feature_names.csv    — feature names for the above array
  shap_summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap

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

TOP_N_HEAT = 20   # features in heatmap
TOP_N_BEE  = 20   # features in beeswarm
NAN_THRESH  = 0.50


def parse_meta(index_series):
    split     = index_series.str.split("|", n=1, expand=True)
    class_str = split[0]
    slide_idx = split[1].str.split("_x_", n=1).str[0]
    return class_str, slide_idx


def clean_label(name, max_len=50):
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
    df["_class"]     = class_str.values
    df["_slide"]     = slide_idx.values
    df["_label"]     = (df["_class"] == "class_1").astype(int)   # 1=MYCN-amp, 0=non-amp

    meta = ["_class", "_slide", "_label"]
    feat_cols = [c for c in df.columns if c not in meta
                 and pd.api.types.is_numeric_dtype(df[c])]

    amp_m    = df["_label"] == 1
    nonamp_m = df["_label"] == 0
    feat_cols = [c for c in feat_cols
                 if df.loc[amp_m, c].isna().mean() <= NAN_THRESH
                 and df.loc[nonamp_m, c].isna().mean() <= NAN_THRESH]

    # drop near-zero variance features (e.g. all-zero columns)
    feat_cols = [c for c in feat_cols if df[c].std() > 1e-9]
    print(f"  Features: {len(feat_cols)}  | non-amp: {nonamp_m.sum()}, MYCN-amp: {amp_m.sum()}")

    X_raw = df[feat_cols].values.astype(float)
    y     = df["_label"].values

    # ── preprocessing ─────────────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    # ── logistic regression ───────────────────────────────────────────────────
    lr = LogisticRegression(
        C=0.1,                    # L2 regularisation (avoids perfect separation at n=271)
        class_weight="balanced",  # counteract 1194:271 imbalance
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X_sc, y)
    train_acc = lr.score(X_sc, y)
    print(f"  LR train accuracy: {train_acc:.3f}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    # LinearExplainer: exact SHAP for linear models, uses background = training data mean
    background = shap.maskers.Independent(X_sc, max_samples=200)
    explainer  = shap.LinearExplainer(lr, background)
    shap_vals  = explainer.shap_values(X_sc)  # [n_tiles, n_feats] for class 1 (MYCN-amp)
    # For binary classification, LinearExplainer returns shap values for class 1 (MYCN-amp)
    # Positive SHAP → pushes toward MYCN-amp prediction ✓
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # class 1 = MYCN-amp

    print(f"  SHAP values shape: {shap_vals.shape}")

    # save arrays
    np.save(os.path.join(OUT_DIR, f"{comp_name}_shap_values.npy"), shap_vals)
    pd.DataFrame({"feature": feat_cols}).to_csv(
        os.path.join(OUT_DIR, f"{comp_name}_feature_names.csv"), index=False
    )

    # ── feature importance ────────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feat_imp      = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs_shap})
    feat_imp      = feat_imp.sort_values("mean_abs_shap", ascending=False)
    feat_imp.to_csv(os.path.join(OUT_DIR, f"{comp_name}_shap_importance.csv"), index=False)

    top_features = feat_imp["feature"].head(TOP_N_HEAT).tolist()
    top_idx      = [feat_cols.index(f) for f in top_features]

    # ── SHAP heatmap ──────────────────────────────────────────────────────────
    # Sort tiles: non-amp first (sorted by SHAP sum ascending), then MYCN-amp (ascending)
    shap_sum  = shap_vals.sum(axis=1)
    sort_order = np.concatenate([
        np.where(y == 0)[0][np.argsort(shap_sum[y == 0])],
        np.where(y == 1)[0][np.argsort(shap_sum[y == 1])],
    ])
    n_nonamp = int((y == 0).sum())
    n_amp    = int((y == 1).sum())

    shap_plot = shap_vals[sort_order, :][:, top_idx]   # [n_tiles, top_feats]
    shap_plot = shap_plot.T                             # [top_feats, n_tiles]

    feat_labels = [clean_label(top_features[i]) for i in range(len(top_features))]

    # robust color scale: clip at 5th/95th percentile
    vmax = np.percentile(np.abs(shap_plot), 95)
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(shap_plot, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="SHAP value\n(+ve → MYCN-amp, −ve → non-amp)",
                 fraction=0.03, pad=0.02)

    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=7)
    ax.set_xlabel("Tiles  (non-amp ← | → MYCN-amp)", fontsize=9)
    ax.axvline(n_nonamp - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    # class labels at top
    mid_nonamp = n_nonamp // 2
    mid_amp    = n_nonamp + n_amp // 2
    ax.text(mid_nonamp, -0.8, "non-amp", ha="center", va="bottom", fontsize=8,
            color="#4F74C8", fontweight="bold", transform=ax.transData)
    ax.text(mid_amp, -0.8, "MYCN-amp", ha="center", va="bottom", fontsize=8,
            color="#D94F3D", fontweight="bold", transform=ax.transData)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_shap_heatmap.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {comp_name}_shap_heatmap.pdf")

    # ── SHAP beeswarm / summary ───────────────────────────────────────────────
    # Normalise feature values to [0, 1] per feature for colour mapping
    top_shap_vals  = shap_vals[:, top_idx[:TOP_N_BEE]]     # [n_tiles, TOP_N_BEE]
    top_feat_vals  = X_sc[:, top_idx[:TOP_N_BEE]]          # scaled feature values
    top_feat_norms = np.zeros_like(top_feat_vals)
    for j in range(top_feat_vals.shape[1]):
        col = top_feat_vals[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            top_feat_norms[:, j] = (col - cmin) / (cmax - cmin)
        else:
            top_feat_norms[:, j] = 0.5

    cmap = plt.cm.RdBu_r
    fig, ax = plt.subplots(figsize=(8, 10))

    jitter_rng = np.random.default_rng(42)
    for rank in range(min(TOP_N_BEE, len(top_idx))):
        sv   = top_shap_vals[:, rank]
        fv   = top_feat_norms[:, rank]
        jitter = jitter_rng.uniform(-0.25, 0.25, size=len(sv))
        y_pos = (TOP_N_BEE - 1 - rank) + jitter
        colours_pts = cmap(fv)
        ax.scatter(sv, y_pos, c=colours_pts, s=4, alpha=0.5, linewidths=0)

    ax.set_yticks(range(TOP_N_BEE))
    ax.set_yticklabels(
        [clean_label(top_features[TOP_N_BEE - 1 - i]) for i in range(TOP_N_BEE)],
        fontsize=8,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (+ve → MYCN-amp, −ve → non-amp)", fontsize=10)

    # colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature value (scaled low → high)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    ax.text(0.01, 0.01, "← lower in MYCN-amp    higher in MYCN-amp →",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_shap_beeswarm.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Beeswarm saved: {comp_name}_shap_beeswarm.pdf")

    # summary
    top5 = feat_imp["feature"].head(5).tolist()
    summary_lines.append(f"\n{comp_name.upper()} — SHAP (LogReg, class_weight=balanced, C=0.1)")
    summary_lines.append(f"  Train accuracy : {train_acc:.3f}")
    summary_lines.append(f"  Top 5 features by mean |SHAP|:")
    for i, feat in enumerate(top5):
        d = "MYCN-amp >" if shap_vals[:, feat_cols.index(feat)][y == 1].mean() > 0 else "non-amp >"
        summary_lines.append(
            f"    {i+1}. {feat[:55]} | mean|SHAP|={mean_abs_shap[feat_cols.index(feat)]:.4f} | direction: {d}"
        )

# ── write summary ─────────────────────────────────────────────────────────────
header = [
    "=" * 70,
    "SHAP ANALYSIS — corrected class labels (class_1=MYCN-amp, class_0=non-amp)",
    "Logistic regression, class_weight=balanced, C=0.1",
    "LinearExplainer SHAP values for class 1 (MYCN-amp)",
    "Positive SHAP → pushes toward MYCN-amp prediction",
    "=" * 70,
]
summary = "\n".join(header + summary_lines)
print("\n" + summary)
with open(os.path.join(OUT_DIR, "shap_summary.txt"), "w") as f:
    f.write(summary + "\n")

print(f"\nAll SHAP outputs written to: {OUT_DIR}")
