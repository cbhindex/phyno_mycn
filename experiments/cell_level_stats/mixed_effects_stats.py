# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Linear mixed-effects model for Section 2.4 (as suggested by line manager).

Model: feature_value ~ class + (1 | slide_idx)
  - Fixed effect 'class': estimated difference MYCN-amp vs non-amp
    accounting for within-slide correlation via random intercept.
  - Positive coefficient → feature higher in MYCN-amp (class_1)
  - Negative coefficient → feature higher in non-amp (class_0)

This is the statistically correct approach: uses all tile-level data while
properly modelling the hierarchical structure (tiles nested within slides).

Components: C3 = prototype_2, C5 = prototype_4.

Outputs (results/ subfolder):
  component3_mixed_effects_stats.csv
  component5_mixed_effects_stats.csv
  component3_mixed_effects_top.pdf    — z-stat bar chart, top 20 features
  component5_mixed_effects_top.pdf
  mixed_effects_summary.txt
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

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

TOP_N    = 20
NAN_THRESH = 0.50


def parse_meta(index_series):
    split     = index_series.str.split("|", n=1, expand=True)
    class_str = split[0]
    slide_idx = split[1].str.split("_x_", n=1).str[0]
    return class_str, slide_idx


def cohens_d_lme(coef, resid_var, rand_var):
    """Approximate Cohen's d from mixed model variance components."""
    total_sd = math.sqrt(resid_var + rand_var) if (resid_var + rand_var) > 0 else 1e-9
    return coef / total_sd


def clean_label(name, max_len=45):
    for p in ["neuroblast cells: ", "immune cells: ", "necrosis cells: ", "neuroblast cells "]:
        if name.lower().startswith(p.lower()):
            name = name[len(p):]
            break
    name = name.replace("mean of their ", "").strip()
    return name[:max_len - 1] + "…" if len(name) > max_len else name


def sig_stars(p_fdr):
    if p_fdr < 0.001: return "***"
    if p_fdr < 0.01:  return "**"
    if p_fdr < 0.05:  return "*"
    return ""


summary_lines = []

for comp_name, csv_path in CSVS.items():
    print(f"\n{'='*60}\n{comp_name}\n{'='*60}")

    df = pd.read_csv(csv_path, index_col=0)
    class_str, slide_idx = parse_meta(pd.Series(df.index))
    df["_class"]     = class_str.values
    df["_slide"]     = slide_idx.values
    df["_class_num"] = (df["_class"] == "class_1").astype(float)  # 1=MYCN-amp, 0=non-amp

    meta_cols = ["_class", "_slide", "_class_num"]
    feat_cols = [c for c in df.columns if c not in meta_cols
                 and pd.api.types.is_numeric_dtype(df[c])]

    # filter high-NaN
    amp_m    = df["_class"] == "class_1"
    nonamp_m = df["_class"] == "class_0"
    feat_cols = [c for c in feat_cols
                 if df.loc[amp_m, c].isna().mean() <= NAN_THRESH
                 and df.loc[nonamp_m, c].isna().mean() <= NAN_THRESH]
    print(f"  Features: {len(feat_cols)}  |  Tiles: non-amp={nonamp_m.sum()}, MYCN-amp={amp_m.sum()}")

    rows = []
    for feat in feat_cols:
        sub = df[["_class_num", "_slide", feat]].dropna()
        if len(sub) < 10 or sub["_class_num"].nunique() < 2:
            continue
        endog  = sub[feat].values.astype(float)
        exog   = np.column_stack([np.ones(len(sub)), sub["_class_num"].values])
        groups = sub["_slide"].values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = MixedLM(endog, exog, groups=groups).fit(
                    reml=True, method="bfgs", maxiter=200
                )
            coef = float(result.params[1])
            se   = float(result.bse[1])
            z    = float(result.tvalues[1])
            p    = float(result.pvalues[1])
            # variance components for approx Cohen's d
            cre = result.cov_re
            rv  = float(cre[0, 0] if isinstance(cre, np.ndarray) else cre.values[0, 0]) \
                  if hasattr(result, "cov_re") and result.cov_re is not None else 0.0
            ev = float(result.scale)
            d  = cohens_d_lme(coef, ev, rv)
        except Exception:
            coef = se = z = p = d = float("nan")
        rows.append({"feature": feat, "coef": coef, "se": se, "z_stat": z,
                     "p_val": p, "cohens_d_approx": d})

    stats_df = pd.DataFrame(rows).dropna(subset=["p_val"])
    reject, p_fdr, _, _ = multipletests(stats_df["p_val"].values, method="fdr_bh")
    stats_df["p_fdr"]   = p_fdr
    stats_df["fdr_sig"] = reject
    stats_df["stars"]   = stats_df["p_fdr"].apply(sig_stars)
    stats_df = stats_df.sort_values("z_stat", ascending=False).reset_index(drop=True)
    stats_df.to_csv(os.path.join(OUT_DIR, f"{comp_name}_mixed_effects_stats.csv"), index=False)

    n_sig = int(stats_df["fdr_sig"].sum())
    n_pos = int((stats_df["fdr_sig"] & (stats_df["z_stat"] > 0)).sum())
    n_neg = int((stats_df["fdr_sig"] & (stats_df["z_stat"] < 0)).sum())

    print(f"  FDR-significant: {n_sig}  ({n_pos} higher MYCN-amp, {n_neg} higher non-amp)")
    top10 = stats_df.reindex(stats_df["z_stat"].abs().nlargest(10).index)
    for _, r in top10.sort_values("z_stat").iterrows():
        direction = "↑amp" if r["z_stat"] > 0 else "↓amp"
        print(f"    {direction}  z={r['z_stat']:+.2f}  p_fdr={r['p_fdr']:.3f}{r['stars']}  "
              f"d={r['cohens_d_approx']:+.2f}  [{r['feature'][:55]}]")

    summary_lines.append(
        f"\n{comp_name.upper()} — LME (feature ~ class + (1|slide), FDR-BH)"
    )
    summary_lines.append(f"  Features tested : {len(stats_df)}")
    summary_lines.append(
        f"  FDR-significant : {n_sig}  ({n_pos} higher MYCN-amp, {n_neg} higher non-amp)"
    )

    # ── figure ────────────────────────────────────────────────────────────────
    plot_df = stats_df.reindex(stats_df["z_stat"].abs().nlargest(TOP_N).index)
    plot_df = plot_df.sort_values("z_stat")

    labels  = [clean_label(f) for f in plot_df["feature"]]
    z_vals  = plot_df["z_stat"].values
    p_fdrs  = plot_df["p_fdr"].values
    stars   = [sig_stars(p) for p in p_fdrs]
    colours = ["#D94F3D" if z > 0 else "#4F74C8" for z in z_vals]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(range(len(labels)), z_vals, color=colours, edgecolor="none", height=0.7)
    for i, (z, s) in enumerate(zip(z_vals, stars)):
        if s:
            offset = 0.12 if z >= 0 else -0.12
            ax.text(z + offset, i, s, va="center",
                    ha="left" if z >= 0 else "right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mixed-effects z-statistic  (class fixed effect, n_slides=10 per class)",
                  fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(color="#D94F3D", label="Higher in MYCN-amp"),
        mpatches.Patch(color="#4F74C8", label="Higher in non-amp"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{comp_name}_mixed_effects_top.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {comp_name}_mixed_effects_top.pdf")

# ── summary ───────────────────────────────────────────────────────────────────
header = [
    "=" * 70,
    "LINEAR MIXED-EFFECTS MODEL — feature ~ class + (1|slide)",
    "Correct statistical approach: tiles nested within slides",
    "Positive coefficient / z > 0 → feature HIGHER in MYCN-amp (class_1)",
    "=" * 70,
]
summary = "\n".join(header + summary_lines)
print("\n" + summary)
with open(os.path.join(OUT_DIR, "mixed_effects_summary.txt"), "w") as f:
    f.write(summary + "\n")

print(f"\nOutputs written to: {OUT_DIR}")
