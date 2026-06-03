# Bundled weights

## `pheno_mycn_k6_fold9.ckpt`

- **Model:** Pheno-MYCN — CLAM-SB attention-MIL backbone + auxiliary GMM
  phenotype branch.
- **GMM components:** K = 6 (the configuration selected in the manuscript).
- **Cross-validation fold:** 9 of 10.
- **Format:** PyTorch-Lightning weights-only checkpoint (`save_weights_only=True`).
  The network weights live under the `model.` prefix; the plug-and-play loader
  ([`pheno_mycn/inference.py`](../../pheno_mycn/inference.py)) strips this prefix
  and loads them into a fresh `CLAM_SB`, so no Lightning runtime is required.
- **Committed directly** in the repository (~7 MB); no Git LFS required.

### Held-out test metrics (this fold)

See `fold9_result.csv`:

| accuracy | F1 | precision | recall | AUC |
|----------|------|-----------|--------|------|
| 0.778 | 0.750 | 0.750 | 0.750 | 0.722 |

These are single-fold numbers. The manuscript reports mean ± SD across all ten
folds (Pheno-MYCN: AUC 0.90 ± 0.12, accuracy 0.89 ± 0.13).

### Provenance

This checkpoint is the model used for the GMM-responsibility and phenotype
analyses in the paper, so its six components correspond to the manuscript's
Components 1–6. The remaining folds and the K = 2…9 component sweep are not
redistributed in this repository.
