# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
00_build_cohort.py — rebuild slide-level survival cohort for Section 2.5.

Input:
  - pheno_mycn_paper/Book 6(Sheet1).csv  (clinical registry; encoding latin-1)
  - olga_refactered/data/cv_splits/neuroblastoma/fold0.csv  (canonical 189 MIL slides + MYCN)

Output:
  - data/survival_per_slide.csv  (189 rows; one per slide)
  - data/cohort_flow.txt         (build log)

Conventions:
  - Analysis unit = SLIDE (multi-slide patients enter multiple times).
  - MYCN per slide preserved from fold CSV (no patient-level adjudication).
  - OS reference start = `date_of_biopsy_resection_or_surgery_from_which_ffpe_taken` (Book 6 per-patient).
  - event=1 if Book 6 `date_of_death` non-null; OS_time = death − biopsy.
  - event=0 if Book 6 `current_patient_status == "Patient alive"`;
              OS_time = study cut-off − biopsy, where study cut-off = max date_of_death across this 86-patient subset.
"""

import os
import re
import sys
import pandas as pd

BASE = os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn")
BOOK6 = os.path.join(BASE, "pheno_mycn_paper/Book 6(Sheet1).csv")
FOLD0 = os.path.join(BASE, "olga_refactered/data/cv_splits/neuroblastoma/fold0.csv")
OUT_DIR = os.path.join(BASE, "additional_exp/survival_analysis/data")
os.makedirs(OUT_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 70)
log("Slide-level survival cohort build — Section 2.5")
log("=" * 70)

# --------------------------------------------------------------------------- #
# 1. Canonical 189-slide cohort from fold0.csv (train+val+test concatenated)  #
# --------------------------------------------------------------------------- #
fold = pd.read_csv(FOLD0)
parts = []
for split in ("train", "val", "test"):
    lbl = f"{split}_label"
    sub = fold[[split, lbl]].dropna().rename(
        columns={split: "slide_name", lbl: "mycn_perslide"}
    )
    sub["split"] = split
    parts.append(sub)
slides = pd.concat(parts, ignore_index=True).drop_duplicates("slide_name")
slides["mycn_perslide"] = slides["mycn_perslide"].astype(int)
slides["patient_id"] = slides.slide_name.str.extract(r"^(SMP\d+)")

# timepoint extracted from slide name
def get_timepoint(name):
    n = name.lower()
    if "primary diagnosis" in n:
        return "primary"
    if "relapse" in n:
        return "relapse"
    return "unknown"
slides["timepoint"] = slides.slide_name.apply(get_timepoint)

log(f"Loaded {len(slides)} unique slides from fold0.csv")
log(f"  MYCN per-slide: {dict(slides.mycn_perslide.value_counts())}")
log(f"  Timepoints: {dict(slides.timepoint.value_counts())}")
log(f"  Unique patients: {slides.patient_id.nunique()}")

# --------------------------------------------------------------------------- #
# 2. Book 6 → per-patient survival                                            #
# --------------------------------------------------------------------------- #
b6 = pd.read_csv(BOOK6, encoding="latin-1")
sub = b6[b6.patient_id.isin(slides.patient_id)].copy()
log(f"\nBook 6 rows for these 86 patients: {len(sub)} "
    f"(={sub.patient_id.nunique()} unique pid)")

# Parse dates
for c in ("date_of_death", "date_of_biopsy_resection_or_surgery_from_which_ffpe_taken"):
    sub[c] = pd.to_datetime(sub[c], errors="coerce", format="mixed")

# Per-patient dedup: prefer row with non-null death; else first.
def dedup(g):
    death = g.dropna(subset=["date_of_death"])
    return death.iloc[0] if len(death) else g.iloc[0]
patient = sub.groupby("patient_id", as_index=False).apply(dedup).reset_index(drop=True)
log(f"After per-patient dedup: {len(patient)} patient rows "
    f"({patient.date_of_death.notna().sum()} deceased, "
    f"{patient.date_of_death.isna().sum()} alive/censored)")

# Study cut-off = max death date in cohort (proxy for last observed event)
study_cutoff = patient.date_of_death.max()
log(f"Study cut-off for censoring = max(date_of_death) = {study_cutoff.date()}")

# Compute OS_time and event
def compute_os(row):
    biopsy = row.date_of_biopsy_resection_or_surgery_from_which_ffpe_taken
    if pd.isna(biopsy):
        return pd.Series([None, None, "no_biopsy_date"])
    if pd.notna(row.date_of_death):
        days = (row.date_of_death - biopsy).days
        return pd.Series([days, 1, "deceased"])
    # censored
    if row.current_patient_status == "Patient alive":
        days = (study_cutoff - biopsy).days
        return pd.Series([days, 0, "censored_alive"])
    return pd.Series([None, None, "unknown_status"])

patient[["OS_time_days", "event", "note"]] = patient.apply(compute_os, axis=1)
log(f"\nOS computation result:")
log(f"  {dict(patient.note.value_counts())}")
# drop patients with no biopsy date or no derivable OS
keep = patient.OS_time_days.notna() & (patient.OS_time_days >= 0)
log(f"  Patients with valid OS (>=0 days): {keep.sum()} / {len(patient)}")
neg = patient[(patient.OS_time_days.notna()) & (patient.OS_time_days < 0)]
if len(neg):
    log(f"  Negative-OS patients (death before biopsy date): {neg.patient_id.tolist()}")
patient = patient[keep].copy()

# --------------------------------------------------------------------------- #
# 3. Join Book 6 patient survival back to 189 slides                          #
# --------------------------------------------------------------------------- #
keep_cols = [
    "patient_id", "OS_time_days", "event",
    "date_of_death", "date_of_biopsy_resection_or_surgery_from_which_ffpe_taken",
    "current_patient_status", "patient_age_at_biopsy_months", "gender",
    "detailed_diagnosis", "disease_category",
]
out = slides.merge(patient[keep_cols], on="patient_id", how="left")
log(f"\nJoined to slides: {len(out)} slide rows; "
    f"slides with OS: {out.OS_time_days.notna().sum()}")

# Order columns
final = out[
    ["slide_name", "patient_id", "mycn_perslide", "timepoint", "split",
     "OS_time_days", "event",
     "date_of_death", "date_of_biopsy_resection_or_surgery_from_which_ffpe_taken",
     "current_patient_status", "patient_age_at_biopsy_months", "gender",
     "detailed_diagnosis", "disease_category"]
].copy()
final["event"] = final["event"].astype("Int64")
final["OS_time_days"] = final["OS_time_days"].astype("Int64")

# --------------------------------------------------------------------------- #
# 4. Summary stats                                                             #
# --------------------------------------------------------------------------- #
log("\n" + "-" * 70)
log("FINAL slide-level cohort summary")
log("-" * 70)
log(f"  Total slides:        {len(final)}")
log(f"  Unique patients:     {final.patient_id.nunique()}")
log(f"  Per-slide MYCN:      "
    f"amp={int((final.mycn_perslide==1).sum())}, "
    f"non-amp={int((final.mycn_perslide==0).sum())}")
log(f"  Per-slide event:     "
    f"deceased={int((final.event==1).sum())}, "
    f"censored={int((final.event==0).sum())}, "
    f"missing={int(final.event.isna().sum())}")
log(f"  Per-slide timepoint: {dict(final.timepoint.value_counts())}")
log(f"  OS days (event=1):   "
    f"median={int(final.loc[final.event==1, 'OS_time_days'].median())}, "
    f"range=[{int(final.loc[final.event==1, 'OS_time_days'].min())}, "
    f"{int(final.loc[final.event==1, 'OS_time_days'].max())}]")
log(f"  OS days (event=0):   "
    f"median={int(final.loc[final.event==0, 'OS_time_days'].median())}, "
    f"range=[{int(final.loc[final.event==0, 'OS_time_days'].min())}, "
    f"{int(final.loc[final.event==0, 'OS_time_days'].max())}]")

# Patient-level summary for transparency
plog = final.drop_duplicates("patient_id")
log(f"\n  Patient-level (deduplicated): {len(plog)} patients; "
    f"events={int((plog.event==1).sum())}, "
    f"censored={int((plog.event==0).sum())}")

# --------------------------------------------------------------------------- #
# 5. Save                                                                      #
# --------------------------------------------------------------------------- #
out_csv = os.path.join(OUT_DIR, "survival_per_slide.csv")
final.to_csv(out_csv, index=False)
log(f"\nWrote {out_csv}")

with open(os.path.join(OUT_DIR, "cohort_flow.txt"), "w") as f:
    f.write("\n".join(log_lines) + "\n")
print(f"\nWrote cohort_flow.txt")
