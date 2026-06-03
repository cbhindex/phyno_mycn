# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Slide-level aggregation diagnostic for Section 2.4 pseudo-replication assessment.

The original t-tests in the manuscript treated ~1,194/271 tiles as independent observations
(Component 3) and ~1,245/329 tiles (Component 5), but all tiles come from 10 slides per class.
This script recomputes the analysis at the correct unit: slide-level means (n=10 per class).

For each component (C3=prototype_2, C5=prototype_4):
  1. Load cell_info_updated.csv
  2. Parse class and slide index from index column (format: class_X|SLIDE_IDX_x_X_y_Y)
  3. Drop columns with >50% NaN overall
  4. Compute per-slide mean for each feature → 10 slide means per class
  5. Two-sample Welch's t-test on 10 vs 10 slide means
  6. Cohen's d effect size
  7. FDR correction (Benjamini-Hochberg)
  8. Save stats CSV, slide means CSV, PDF figure

Outputs (results/ subfolder):
  component3_slide_level_stats.csv
  component5_slide_level_stats.csv
  component3_slide_means.csv
  component5_slide_means.csv
  component3_top_features.pdf
  component5_top_features.pdf
  summary.txt
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests

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

TOP_N = 20          # features to show in figure
NAN_THRESH = 0.50   # drop columns with >50% NaN in either class


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_class_slide(index_series):
    """Return (class_str, slide_idx) Series pair from index like 'class_X|SLIDE_x_...'."""
    split = index_series.str.split("|", n=1, expand=True)
    class_str = split[0]                          # 'class_0' or 'class_1'
    slide_idx  = split[1].str.split("_x_", n=1).str[0]  # '0'–'9'
    return class_str, slide_idx


def cohens_d(a, b):
    """Cohen's d: (mean_a - mean_b) / pooled SD. Positive → a > b."""
    n_a, n_b = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd


def clean_label(name, max_len=45):
    """Shorten feature label for y-axis."""
    prefixes = [
        "neuroblast cells: ",
        "immune cells: ",
        "necrosis cells: ",
        "neuroblast cells ",
    ]
    for p in prefixes:
        if name.lower().startswith(p.lower()):
            name = name[len(p):]
            break
    name = name.replace("mean of their ", "").strip()
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return name


def sig_stars(p_fdr):
    if p_fdr < 0.001:
        return "***"
    elif p_fdr < 0.01:
        return "**"
    elif p_fdr < 0.05:
        return "*"
    return ""


# ── main loop ─────────────────────────────────────────────────────────────────

summary_lines = []

for comp_name, csv_path in CSVS.items():
    print(f"\n{'='*60}")
    print(f"Processing {comp_name}  →  {csv_path}")
    print("="*60)

    # ── load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)
    class_str, slide_idx = parse_class_slide(pd.Series(df.index))
    df["_class"]     = class_str.values
    df["_slide_idx"] = slide_idx.values

    amp_mask    = df["_class"] == "class_1"
    nonamp_mask = df["_class"] == "class_0"
    print(f"  Tiles: non-amp={nonamp_mask.sum()}  MYCN-amp={amp_mask.sum()}")
    print(f"  Slides non-amp: {sorted(df.loc[nonamp_mask, '_slide_idx'].unique())}")
    print(f"  Slides MYCN-amp: {sorted(df.loc[amp_mask, '_slide_idx'].unique())}")

    # ── feature columns ───────────────────────────────────────────────────────
    meta_cols = ["_class", "_slide_idx"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    # drop non-numeric
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    # drop high-NaN columns
    keep = []
    for c in feat_cols:
        nan_amp    = df.loc[amp_mask,    c].isna().mean()
        nan_nonamp = df.loc[nonamp_mask, c].isna().mean()
        if nan_amp <= NAN_THRESH and nan_nonamp <= NAN_THRESH:
            keep.append(c)
    print(f"  Feature columns after NaN filter: {len(keep)} / {len(feat_cols)}")
    feat_cols = keep

    # ── slide-level means ─────────────────────────────────────────────────────
    slide_means = (
        df.groupby(["_class", "_slide_idx"])[feat_cols]
        .mean()
        .reset_index()
    )
    slide_means_amp    = slide_means[slide_means["_class"] == "class_1"].set_index("_slide_idx")[feat_cols]
    slide_means_nonamp = slide_means[slide_means["_class"] == "class_0"].set_index("_slide_idx")[feat_cols]

    print(f"  Slide means shape: amp={slide_means_amp.shape}, non-amp={slide_means_nonamp.shape}")

    # save slide means
    slide_means.to_csv(os.path.join(OUT_DIR, f"{comp_name}_slide_means.csv"), index=False)

    # ── t-tests + effect sizes ────────────────────────────────────────────────
    rows = []
    for feat in feat_cols:
        a = slide_means_amp[feat].dropna().values
        b = slide_means_nonamp[feat].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)
        rows.append({
            "feature":          feat,
            "mean_amp":         np.mean(a),
            "sd_amp":           np.std(a, ddof=1),
            "mean_nonamp":      np.mean(b),
            "sd_nonamp":        np.std(b, ddof=1),
            "n_amp":            len(a),
            "n_nonamp":         len(b),
            "t_stat":           t_stat,
            "p_val":            p_val,
            "cohens_d":         d,
        })

    stats_df = pd.DataFrame(rows)

    # FDR correction
    reject, p_fdr, _, _ = multipletests(stats_df["p_val"].values, method="fdr_bh")
    stats_df["p_fdr"]    = p_fdr
    stats_df["fdr_sig"]  = reject
    stats_df["stars"]    = stats_df["p_fdr"].apply(sig_stars)

    stats_df = stats_df.sort_values("t_stat", ascending=False).reset_index(drop=True)
    stats_df.to_csv(os.path.join(OUT_DIR, f"{comp_name}_slide_level_stats.csv"), index=False)

    n_sig   = int(stats_df["fdr_sig"].sum())
    n_pos   = int((stats_df["fdr_sig"] & (stats_df["t_stat"] > 0)).sum())
    n_neg   = int((stats_df["fdr_sig"] & (stats_df["t_stat"] < 0)).sum())
    top_pos = stats_df[stats_df["t_stat"] > 0].head(3)["feature"].tolist()
    top_neg = stats_df[stats_df["t_stat"] < 0].tail(3)["feature"].tolist()

    summary_lines.append(f"\n{comp_name.upper()} (slide-level Welch t-test, n=10 per class, FDR-BH)")
    summary_lines.append(f"  Features tested      : {len(stats_df)}")
    summary_lines.append(f"  FDR-significant (<0.05): {n_sig}  ({n_pos} higher in MYCN-amp, {n_neg} higher in non-amp)")
    summary_lines.append(f"  Top features higher in MYCN-amp (by t-stat): {top_pos}")
    summary_lines.append(f"  Top features higher in non-amp  (by |t-stat|): {list(reversed(top_neg))}")

    # print top 10
    print(f"\n  Top 10 by |t-stat|:")
    top10 = stats_df.reindex(stats_df["t_stat"].abs().nlargest(10).index)
    for _, r in top10.iterrows():
        direction = "↑amp" if r["t_stat"] > 0 else "↓amp"
        print(f"    {direction}  t={r['t_stat']:+.2f}  p_fdr={r['p_fdr']:.3f}{r['stars']}  "
              f"d={r['cohens_d']:+.2f}  [{r['feature'][:50]}]")

    # ── figure ────────────────────────────────────────────────────────────────
    # Pick top N by |t-stat|
    plot_df = stats_df.reindex(stats_df["t_stat"].abs().nlargest(TOP_N).index)
    plot_df = plot_df.sort_values("t_stat")   # ascending → highest at top in horizontal bar

    labels  = [clean_label(f) for f in plot_df["feature"]]
    t_vals  = plot_df["t_stat"].values
    p_fdrs  = plot_df["p_fdr"].values
    stars   = [sig_stars(p) for p in p_fdrs]
    colours = ["#D94F3D" if t > 0 else "#4F74C8" for t in t_vals]   # red / blue

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(range(len(labels)), t_vals, color=colours, edgecolor="none", height=0.7)

    # significance markers
    for i, (t, s) in enumerate(zip(t_vals, stars)):
        if s:
            offset = 0.15 if t >= 0 else -0.15
            ha = "left" if t >= 0 else "right"
            ax.text(t + offset, i, s, va="center", ha=ha, fontsize=9, color="black")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Slide-level t-statistic  (Welch, n=10 per class)", fontsize=10)

    red_patch  = mpatches.Patch(color="#D94F3D", label="Higher in MYCN-amp")
    blue_patch = mpatches.Patch(color="#4F74C8", label="Higher in non-amp")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"{comp_name}_top_features.pdf")
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

# ── summary text ──────────────────────────────────────────────────────────────
summary_header = [
    "=" * 70,
    "SLIDE-LEVEL AGGREGATION DIAGNOSTIC — Section 2.4 pseudo-replication check",
    "Effective n = 10 slides per class (not tile counts)",
    "=" * 70,
]
summary_text = "\n".join(summary_header + summary_lines)
print("\n" + summary_text)

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_text + "\n")

print(f"\nAll outputs written to: {OUT_DIR}")
