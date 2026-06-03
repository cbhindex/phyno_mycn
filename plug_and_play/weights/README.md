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
- **Availability:** released with the publication. Until then the weight file and
  its per-fold metrics are withheld from the repository (see the pre-publication
  `.gitignore` and `../../PUBLICATION.md`).

### Held-out test metrics (this fold)

Per-fold and cross-validation metrics are reported in the manuscript and will be
included here (as `fold9_result.csv`) at publication.

### Provenance

This checkpoint is the model used for the GMM-responsibility and phenotype
analyses in the paper, so its six components correspond to the manuscript's
Components 1–6. The remaining folds and the K = 2…9 component sweep are not
redistributed in this repository.
