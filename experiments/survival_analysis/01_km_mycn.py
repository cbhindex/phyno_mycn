# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
01_km_mycn.py — Kaplan–Meier + log-rank by MYCN status (slide-level).

Panel a — cohort flow summary text.
Panel b — KM curves stratified by per-slide MYCN with log-rank descriptive p.

Note: slides from the same patient are not independent observations; the
log-rank p is descriptive only, not a formal significance test. Reported
patient-level KM is included as a supplementary sanity check.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401  (applies global font / rcParams)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

BASE = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/phyno_mycn"), "additional_exp/survival_analysis")
COHORT = os.path.join(BASE, "data/survival_per_slide.csv")
OUT = os.path.join(BASE, "results/km_curves")
os.makedirs(OUT, exist_ok=True)

C_AMP = "#D94F3D"
C_NON = "#4F74C8"

df = pd.read_csv(COHORT)
df = df.dropna(subset=["OS_time_days", "event"]).copy()
df["OS_time_days"] = df["OS_time_days"].astype(float)
df["event"] = df["event"].astype(int)
df["OS_time_months"] = df["OS_time_days"] / 30.44
print(f"Loaded {len(df)} slides with valid OS")

# ─── Panel a — cohort flow text ───────────────────────────────────────────── #
flow_lines = ["Cohort flow (slide-level survival cohort, Section 2.5)", "=" * 60]
flow_lines.append(f"Source clinical registry:  Book 6 (820 patients)")
flow_lines.append(f"  → 90 neuroblastic-tumour rows in Book 6")
flow_lines.append(f"  → 86 patients matched to 189-slide MIL cohort")
flow_lines.append(f"  → 85 patients with valid OS (1 patient dropped: death before biopsy date)")
flow_lines.append("")
flow_lines.append(f"Slide-level survival cohort (analysis unit):")
flow_lines.append(f"  Total slides:        {len(df)}")
flow_lines.append(f"  MYCN-amp:            {int((df.mycn_perslide==1).sum())}")
flow_lines.append(f"  MYCN non-amp:        {int((df.mycn_perslide==0).sum())}")
flow_lines.append(f"  Events (deceased):   {int((df.event==1).sum())}")
flow_lines.append(f"  Censored (alive):    {int((df.event==0).sum())}")
flow_lines.append(f"  Primary diagnosis:   {int((df.timepoint=='primary').sum())}")
flow_lines.append(f"  Current relapse:     {int((df.timepoint=='relapse').sum())}")
flow_lines.append("")
flow_lines.append(f"  OS days (event=1):   median={int(df.loc[df.event==1,'OS_time_days'].median())} "
                  f"[IQR {int(df.loc[df.event==1,'OS_time_days'].quantile(0.25))}–"
                  f"{int(df.loc[df.event==1,'OS_time_days'].quantile(0.75))}]")
flow_lines.append(f"  OS days (censored):  median={int(df.loc[df.event==0,'OS_time_days'].median())} "
                  f"[IQR {int(df.loc[df.event==0,'OS_time_days'].quantile(0.25))}–"
                  f"{int(df.loc[df.event==0,'OS_time_days'].quantile(0.75))}]")
with open(os.path.join(OUT, "cohort_summary.txt"), "w") as f:
    f.write("\n".join(flow_lines) + "\n")
print("Wrote cohort_summary.txt")

# ─── Panel b — KM by per-slide MYCN ──────────────────────────────────────── #
amp  = df[df.mycn_perslide == 1]
non  = df[df.mycn_perslide == 0]

lr = logrank_test(non.OS_time_months, amp.OS_time_months,
                  event_observed_A=non.event, event_observed_B=amp.event)

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(6, 5))
kmf.fit(non.OS_time_months, non.event, label=f"MYCN non-amp (n={len(non)})")
kmf.plot_survival_function(ax=ax, ci_show=True, color=C_NON)
median_non = kmf.median_survival_time_
kmf.fit(amp.OS_time_months, amp.event, label=f"MYCN-amp (n={len(amp)})")
kmf.plot_survival_function(ax=ax, ci_show=True, color=C_AMP)
median_amp = kmf.median_survival_time_

ax.set_xlabel("Overall survival (months)")
ax.set_ylabel("Survival probability")
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "km_by_mycn_slide.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "km_by_mycn_slide.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Wrote km_by_mycn_slide.pdf/png   log-rank p={lr.p_value:.3g}")

# ─── Patient-level supplementary KM ─────────────────────────────────────── #
# Per-patient row: prefer event=1 if any, else event=0; MYCN by majority of slides (tie→amp)
def patient_mycn(g):
    counts = g.mycn_perslide.value_counts()
    if 1 in counts and 0 in counts and counts[1] >= counts[0]:
        return 1
    if 1 in counts and counts.get(0, 0) == 0:
        return 1
    return 0
def patient_row(g):
    g_event = g.iloc[0]
    return pd.Series({
        "patient_id": g.name,
        "OS_time_months": float(g.OS_time_months.iloc[0]),
        "event": int(g.event.iloc[0]),
        "mycn_patient": int(patient_mycn(g)),
    })
pat = df.groupby("patient_id").apply(patient_row).reset_index(drop=True)
amp_p = pat[pat.mycn_patient == 1]
non_p = pat[pat.mycn_patient == 0]
lr_p = logrank_test(non_p.OS_time_months, amp_p.OS_time_months,
                    event_observed_A=non_p.event, event_observed_B=amp_p.event)

fig, ax = plt.subplots(figsize=(6, 5))
kmf.fit(non_p.OS_time_months, non_p.event, label=f"MYCN non-amp (n={len(non_p)})")
kmf.plot_survival_function(ax=ax, ci_show=True, color=C_NON)
kmf.fit(amp_p.OS_time_months, amp_p.event, label=f"MYCN-amp (n={len(amp_p)})")
kmf.plot_survival_function(ax=ax, ci_show=True, color=C_AMP)
ax.set_xlabel("Overall survival (months)")
ax.set_ylabel("Survival probability")
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "km_by_mycn_patient.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "km_by_mycn_patient.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Wrote km_by_mycn_patient.pdf/png  log-rank p={lr_p.p_value:.3g}")

# ─── Numeric outputs ─────────────────────────────────────────────────────── #
res = pd.DataFrame([
    {"level": "slide", "group": "MYCN-amp",    "n": len(amp), "events": int(amp.event.sum()),
     "median_OS_months": median_amp},
    {"level": "slide", "group": "MYCN-non-amp", "n": len(non), "events": int(non.event.sum()),
     "median_OS_months": median_non},
    {"level": "slide", "group": "log-rank p",   "n": None, "events": None,
     "median_OS_months": lr.p_value},
    {"level": "patient", "group": "MYCN-amp",    "n": len(amp_p), "events": int(amp_p.event.sum()),
     "median_OS_months": None},
    {"level": "patient", "group": "MYCN-non-amp", "n": len(non_p), "events": int(non_p.event.sum()),
     "median_OS_months": None},
    {"level": "patient", "group": "log-rank p",   "n": None, "events": None,
     "median_OS_months": lr_p.p_value},
])
res.to_csv(os.path.join(OUT, "km_mycn_stats.csv"), index=False)
print("\nDone.\n", res.to_string(index=False))
