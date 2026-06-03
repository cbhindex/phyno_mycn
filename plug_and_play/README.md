# Pheno-MYCN — plug-and-play predictor

A ready-to-use wrapper around the **pretrained** Pheno-MYCN model. Give it one
slide's tile embeddings and it returns the slide-level MYCN-amplification
prediction and the interpretable per-tile GMM phenotype assignment — no
training, no PyTorch Lightning required.

> **Pre-publication note:** the trained model-weight files are withheld until the
> manuscript is published, so `weights/pheno_mycn_k6_fold9.ckpt` may be absent
> from a public checkout. The code is complete and works as soon as a checkpoint
> is present in `weights/`.

## What's included

| File | Description |
|------|-------------|
| `weights/pheno_mycn_k6_fold9.ckpt` | Representative pretrained Pheno-MYCN model: **K = 6** GMM components, cross-validation **fold 9** (released with the publication). |
| `predict.py` | Command-line demo (forwards to the `pheno-mycn` CLI). |

**Why fold 9?** K=6 is the configuration selected in the manuscript, and fold 9
is the model on which the GMM-responsibility / phenotype analysis was run, so its
six components correspond to the Components 1–6 discussed in the paper. (The full
set of 10 folds and the K = 2…9 sweep are not distributed here.)

## Inputs

Pheno-MYCN consumes **UNI tile embeddings**: a tensor of shape
`[n_tiles, 1024]` per slide, the same features used for training. Produce them
by tiling a colour-normalised H&E WSI and encoding each tile with the
[UNI encoder](https://github.com/mahmoodlab/UNI). Accepted file formats: `.pt`
(torch) or `.npy` (numpy).

## Quick start

```bash
# 1. install the package (editable)
pip install -e .

# 2. predict on one slide's tile embeddings (installs the `pheno-mycn` command)
pheno-mycn predict --features /path/to/SLIDE_uni.pt --output SLIDE_phenotypes.csv
```

Equivalent forms: `python -m pheno_mycn predict ...`, or the legacy
`python plug_and_play/predict.py --features ...` (kept for backwards
compatibility — it forwards to `pheno-mycn predict`).

## Python API

```python
import torch
from pheno_mycn import PhenoMYCNPredictor

predictor = PhenoMYCNPredictor.from_pretrained()      # bundled K=6 fold-9 weights
feats = torch.load("SLIDE_uni.pt")                    # [n_tiles, 1024]
out = predictor.predict(feats)

out["mycn_probability"]   # float: P(MYCN-amplified)
out["predicted_label"]    # 0 = non-amplified, 1 = MYCN-amplified
out["responsibilities"]   # [n_tiles, 6] soft GMM responsibilities
out["hard_components"]     # [n_tiles] dominant component, 1-indexed (Components 1..6)
out["attention"]          # [n_tiles] MIL attention weights
out["anomaly_score"]      # float: slide-level GMM free-energy
```

To use a different checkpoint, pass `PhenoMYCNPredictor.from_pretrained(ckpt_path=...)`.

## Notes

- The component indices are **1-indexed** to match the manuscript figures.
- This tool is a research artefact for H&E neuroblastoma WSIs and is **not** a
  diagnostic device.
