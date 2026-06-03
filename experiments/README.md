# Additional experiments

Downstream analyses that build on the trained Pheno-MYCN model, reproducing the
secondary results in the manuscript. **Author: Dr Binghao Chai.**

These scripts consume intermediate outputs of the main pipeline — per-tile GMM
responsibilities/attention, cell-level feature tables, cross-validation splits
and the clinical registry. Those inputs are patient-derived and/or too large to
distribute, so they are **not** included in this repository (see the top-level
`README.md` and `data/README.md`). Each script documents the exact inputs it
expects in its module docstring.

## Configuration

All scripts resolve paths from a single project root, set via an environment
variable (see [`_common/paths.py`](_common/paths.py)):

```bash
export PHENO_MYCN_ROOT=/path/to/phyno_mycn
pip install -r requirements-experiments.txt
```

## Contents

| Folder | Script(s) | What it does |
|--------|-----------|--------------|
| `gmm_responsibility/` | `compute_responsibilities.py` | Mean per-component GMM responsibility per MYCN class across train/val/test (Section 2.2). |
| `cell_level_stats/` | `slide_level_stats.py`, `mixed_effects_stats.py` | Slide-level Welch t-tests (pseudo-replication-corrected) and linear mixed-effects models of cell features by MYCN status, for Components 3 and 5 (Section 2.4). |
| `shap_analysis/` | `run_shap.py` | Logistic-regression + SHAP feature attribution distinguishing MYCN-amp vs non-amp tiles within a component. |
| `latent_space/` | `cell_latent_analysis.py` | Soft-label, UMAP, per-slide violin and patient PCA characterisation of the cell-feature latent space. |
| `survival_analysis/` | `00_build_cohort.py` … `05_screen_summary.py`, `_style.py` | Slide-level survival cohort build and exploratory Kaplan–Meier / log-rank analyses by MYCN status and by GMM component (Section 2.5). Run in numeric order. |

## Dependencies

Install the core requirements plus the experiment extras:

```bash
pip install -r ../requirements.txt
pip install -r requirements-experiments.txt
```

Key extras: `lifelines` (survival), `statsmodels` (mixed-effects models, FDR),
`shap` (attribution), `umap-learn`, `scikit-learn`, `seaborn`.

## Notes

- These are exploratory analyses run on a small single-centre cohort; the
  survival results in particular are descriptive (see the script docstrings and
  the manuscript for the appropriate caveats).
- Outputs are written under each experiment's `results/` subfolder (or, for the
  survival scripts, under `$PHENO_MYCN_ROOT/additional_exp/survival_analysis/`).
