# Pheno-MYCN — additional experiments.
# Author:  Dr Olga Fourkioti  (https://github.com/olgarithmics)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Compute mean GMM responsibility per component per MYCN class (amp vs non-amp),
separately for train / val / test splits of fold 9.

Outputs
-------
results/mean_responsibility_by_split.csv   — per-split, per-component, per-class stats
results/mean_responsibility_all.csv        — aggregated over all splits combined
results/summary.txt                        — human-readable summary for manuscript
"""

import os
import math
import torch
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn"), "olga_refactered")
PT_DIR  = os.path.join(BASE, "results/slide_inference/fold_9/pt_outputs")
CV_CSV  = os.path.join(BASE, "data/cv_splits/neuroblastoma/fold9.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_COMPONENTS = 6  # GMM K=6; Python idx 0-5 = manuscript Component 1-6

# ── load CV splits ────────────────────────────────────────────────────────────
df = pd.read_csv(CV_CSV)

# Build a flat dict: slide_name -> (label, split)
# val and test columns only populated in the first N rows (one per val/test slide)
slide_info = {}
for _, row in df.iterrows():
    slide_info[row["train"]] = (int(row["train_label"]), "train")
    if pd.notna(row["val"]):
        slide_info[row["val"]]  = (int(row["val_label"]),  "val")
    if pd.notna(row["test"]):
        slide_info[row["test"]] = (int(row["test_label"]), "test")

# ── load per-slide mean responsibilities ──────────────────────────────────────
records = []
missing = []

for slide_name, (label, split) in slide_info.items():
    pt_file = os.path.join(PT_DIR, slide_name + "_gmm.pt")
    if not os.path.exists(pt_file):
        missing.append(slide_name)
        continue
    # shape: [1, num_tiles, 6]
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    resp = data[0]  # [num_tiles, 6]
    # per-slide mean responsibility for each component
    mean_resp = resp.mean(dim=0).tolist()  # list of 6 floats
    rec = {"slide": slide_name, "label": label, "split": split,
           "label_str": "MYCN-amp" if label == 1 else "non-amp",
           "n_tiles": resp.shape[0]}
    for c in range(N_COMPONENTS):
        rec[f"comp{c+1}"] = mean_resp[c]
    records.append(rec)

if missing:
    print(f"WARNING: {len(missing)} slides not found in pt_outputs — skipped.")

slide_df = pd.DataFrame(records)
print(f"Loaded {len(slide_df)} slides  "
      f"(MYCN-amp: {(slide_df.label==1).sum()}, non-amp: {(slide_df.label==0).sum()})")

# ── helper: stats per group ───────────────────────────────────────────────────
def group_stats(sub_df):
    rows = []
    for label_str, grp in sub_df.groupby("label_str"):
        for c in range(1, N_COMPONENTS + 1):
            vals = grp[f"comp{c}"].values
            m = vals.mean()
            s = vals.std(ddof=1) if len(vals) > 1 else float("nan")
            rows.append({"label": label_str, "component": c,
                         "n_slides": len(vals), "mean": m, "sd": s})
    return pd.DataFrame(rows)

# ── per-split stats ───────────────────────────────────────────────────────────
split_rows = []
for split in ["train", "val", "test"]:
    sub = slide_df[slide_df.split == split]
    stats = group_stats(sub)
    stats.insert(0, "split", split)
    split_rows.append(stats)

split_stats = pd.concat(split_rows, ignore_index=True)
split_stats.to_csv(os.path.join(OUT_DIR, "mean_responsibility_by_split.csv"), index=False)

# ── all-slides stats (for manuscript sentence) ────────────────────────────────
all_stats = group_stats(slide_df)
all_stats.insert(0, "split", "all")
all_stats.to_csv(os.path.join(OUT_DIR, "mean_responsibility_all.csv"), index=False)

# ── build summary text ────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("GMM RESPONSIBILITY ANALYSIS — fold_9 model, all 189 slides")
lines.append("=" * 70)
lines.append("")
lines.append("Mean slide-level responsibility per component per MYCN class")
lines.append("(each value = mean ± SD across slides; n per class shown)")
lines.append("")

for split in ["train", "val", "test", "all"]:
    sub = slide_df if split == "all" else slide_df[slide_df.split == split]
    amp_n    = int((sub.label == 1).sum())
    nonamp_n = int((sub.label == 0).sum())
    lines.append(f"--- {split.upper()} SPLIT  (MYCN-amp n={amp_n}, non-amp n={nonamp_n}) ---")
    lines.append(f"{'Component':<12} {'MYCN-amp mean±SD':>20} {'non-amp mean±SD':>20}  {'Difference':>12}")
    lines.append("-" * 68)
    for c in range(1, N_COMPONENTS + 1):
        amp_vals    = sub[sub.label == 1][f"comp{c}"].values
        nonamp_vals = sub[sub.label == 0][f"comp{c}"].values
        amp_m  = amp_vals.mean()
        amp_s  = amp_vals.std(ddof=1) if len(amp_vals) > 1 else float("nan")
        na_m   = nonamp_vals.mean()
        na_s   = nonamp_vals.std(ddof=1) if len(nonamp_vals) > 1 else float("nan")
        diff   = amp_m - na_m
        marker = " <-- MYCN-associated" if c in (3, 5) else ""
        lines.append(f"  Comp {c} (C{c}):  {amp_m:.3f}±{amp_s:.3f}          {na_m:.3f}±{na_s:.3f}    {diff:+.3f}{marker}")
    lines.append("")

# Key finding for section 2.2
amp_df    = slide_df[slide_df.label == 1]
nonamp_df = slide_df[slide_df.label == 0]

lines.append("=" * 70)
lines.append("KEY VALUES FOR MANUSCRIPT (all 189 slides, fold_9 model)")
lines.append("=" * 70)
lines.append("")
for c in [3, 5]:
    amp_m  = amp_df[f"comp{c}"].mean()
    amp_s  = amp_df[f"comp{c}"].std(ddof=1)
    na_m   = nonamp_df[f"comp{c}"].mean()
    na_s   = nonamp_df[f"comp{c}"].std(ddof=1)
    lines.append(f"Component {c} (manuscript Component {c}):")
    lines.append(f"  MYCN-amplified : {amp_m:.3f} ± {amp_s:.3f}  (n={len(amp_df)})")
    lines.append(f"  non-amplified  : {na_m:.3f} ± {na_s:.3f}  (n={len(nonamp_df)})")
    lines.append(f"  difference     : {amp_m - na_m:+.3f}")
    lines.append("")

# TEST SPLIT specifically (most relevant for results)
test_sub = slide_df[slide_df.split == "test"]
test_amp    = test_sub[test_sub.label == 1]
test_nonamp = test_sub[test_sub.label == 0]
lines.append("--- TEST SPLIT ONLY (as reported in paper) ---")
lines.append("")
for c in [3, 5]:
    amp_m  = test_amp[f"comp{c}"].mean()
    amp_s  = test_amp[f"comp{c}"].std(ddof=1) if len(test_amp) > 1 else float("nan")
    na_m   = test_nonamp[f"comp{c}"].mean()
    na_s   = test_nonamp[f"comp{c}"].std(ddof=1) if len(test_nonamp) > 1 else float("nan")
    lines.append(f"Component {c}: MYCN-amp {amp_m:.3f}±{amp_s:.3f} vs non-amp {na_m:.3f}±{na_s:.3f}  "
                 f"(n_amp={len(test_amp)}, n_nonamp={len(test_nonamp)})")
lines.append("")

summary_text = "\n".join(lines)
print(summary_text)

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_text)

print(f"\nOutputs written to: {OUT_DIR}")
