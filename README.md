# Pheno-MYCN

**Interpretable histological phenotypes associated with MYCN amplification in
paediatric neuroblastoma.**

Pheno-MYCN is a weakly supervised digital-pathology framework that couples
attention-based multiple-instance learning (MIL) for slide-level MYCN-amplification
prediction with an auxiliary Gaussian mixture model (GMM) branch for
**interpretable tile-level phenotype discovery** in H&E whole-slide images (WSIs)
of paediatric neuroblastoma.

Rather than treating MYCN prediction as a black-box classification task,
Pheno-MYCN learns a MYCN-associated phenotype space (K = 6 components) that lets
you compare morphological heterogeneity between MYCN-amplified and non-amplified
tumours — components that expert pathology review mapped to recognisable
neuroblastoma morphologies (high-cellularity tumour, nuclear moulding,
karyorrhexis, neuropil-rich / ganglionic tissue, necrosis, haemorrhage, and
technical artefact).

On a cohort of 189 H&E slides, Pheno-MYCN outperformed CLAM-SB and TransMIL
baselines (mean across 10 cross-validation folds: accuracy ≈ 0.89, F1 ≈ 0.87,
AUC ≈ 0.90).

> **Research use only.** This is a research artefact for H&E neuroblastoma WSIs,
> not a diagnostic device.

---

## Highlights

- 🧩 **Plug-and-play predictor** — run the pretrained model on a slide's tile
  embeddings via the `pheno-mycn` command line **or** the Python API; no training,
  no PyTorch Lightning at inference. See [`plug_and_play/`](plug_and_play/).
- 🔬 **Interpretable phenotypes** — per-tile soft GMM responsibilities + hard
  component labels (Components 1–6), not just a slide-level score.
- 🏋️ **Full training pipeline** — reproduce the model and the CLAM-SB / TransMIL /
  MILNet baselines.
- 📊 **Reproducible experiments** — survival, SHAP, latent-space, GMM-responsibility
  and cell-level analyses from the manuscript.
- 🖼️ **Visualization** — whole-slide GMM/attention/energy heatmaps and
  phenotype-space maps.

## Repository layout

```
pheno_mycn/        Installable package: model, data pipeline, training interface,
                   optimisers/losses, and the Lightning-free inference API.
  models/          CLAM_SB (= Pheno-MYCN) + GMM branch (ad_loss.py); TransMIL,
                   MILNet_multi baselines; the Lightning ModelInterface.
  datasets/        Per-slide MIL dataset + data module.
  optimizers/      RAdam + Lookahead (trimmed from timm).
  losses/          Loss factory (CrossEntropyLoss / BCEWithLogitsLoss).
  configs/         pheno_mycn_k6.yaml and baseline configs.
  inference.py     PhenoMYCNPredictor — plug-and-play API.
scripts/           train.py / train.sh entry points.
plug_and_play/     Pretrained K=6 (fold-9) checkpoint + CLI demo + docs.
visualization/     WSI heatmap / phenotype-map figure scripts (CLAM-derived).
experiments/       Additional analyses (survival, SHAP, latent space, etc.).
data/              Data-format documentation + a synthetic example (no patient data).
```

## How it works

1. **Tiling + encoding.** An H&E WSI is tissue-detected, tiled, colour-normalised,
   and each tile is encoded with the [UNI](https://github.com/mahmoodlab/UNI)
   encoder → `[n_tiles, 1024]` embeddings (the model input).
2. **MIL branch.** A gated-attention CLAM-SB head aggregates tiles into a
   slide-level representation and predicts MYCN status (+ attention map).
3. **GMM phenotype branch.** A diagonal-covariance GMM (K = 6) is fitted on
   projected tile embeddings from MYCN-amplified training tiles, then applied to
   all tiles to produce per-tile soft *responsibilities* and hard component
   labels — the interpretable phenotype space analysed in the paper.

## Getting started

### 1. Install

Requirements: Python ≥ 3.8 and PyTorch. The pretrained checkpoint (~7 MB) is
committed directly in the repo — nothing extra to fetch.

```bash
git clone <this-repo> && cd <this-repo>

# (recommended) an isolated environment
conda create -n pheno_mycn python=3.10 -y
conda activate pheno_mycn

# install the package + run the pretrained model
pip install -e .
```

`pip install -e .` pulls only the **minimal** runtime dependencies — `torch`,
`numpy`, `pyyaml`, `addict` — which is all you need for prediction.

For GPU acceleration, install the PyTorch build that matches **your** CUDA driver
**before** `pip install -e .`, using the selector at https://pytorch.org. For
example, on a CUDA 12.1 driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Pick the `cuXXX` channel for your driver (`nvidia-smi` shows the maximum CUDA
version it supports). The model then uses the GPU automatically — verify with
`python -c "import torch; print(torch.cuda.is_available())"`. (A bare
`pip install -e .` also works, but may fetch a CUDA build newer than your driver
supports, in which case CUDA falls back to unavailable.)

For the **full pipeline** (training, baselines, metrics) add the extras, or use
the all-in-one conda environment that also covers visualization and experiments:

```bash
pip install -e ".[train]"          # or: pip install -r requirements.txt
conda env create -f environment.yml   # all-in-one alternative
```

### 2. Predict — command line

Installing the package adds a `pheno-mycn` command:

```bash
# slide-level MYCN prediction + per-tile phenotype assignment
pheno-mycn predict --features SLIDE_uni.pt

# also write the per-tile responsibilities / attention to CSV
pheno-mycn predict --features SLIDE_uni.pt --output SLIDE_phenotypes.csv

pheno-mycn --version
```

`--features` is one slide's UNI tile embeddings (`[n_tiles, 1024]`, a `.pt` or
`.npy` file). The GPU is used automatically when available; pass
`--device cuda` or `--device cpu` to choose explicitly. `pheno-mycn predict
--help` lists all options. The same command is also available as
`python -m pheno_mycn predict ...`.

### 3. Predict — Python

```python
import torch
from pheno_mycn import PhenoMYCNPredictor

predictor = PhenoMYCNPredictor.from_pretrained()   # bundled K=6 fold-9 weights
feats = torch.load("SLIDE_uni.pt")                 # [n_tiles, 1024] UNI embeddings
out = predictor.predict(feats)

print(out["mycn_probability"])   # P(MYCN-amplified)
print(out["hard_components"])    # per-tile phenotype, 1-indexed (Components 1..6)
```

See [`plug_and_play/README.md`](plug_and_play/README.md) for the full output schema.

## Training

Edit the data paths in [`pheno_mycn/configs/pheno_mycn_k6.yaml`](pheno_mycn/configs/pheno_mycn_k6.yaml),
then:

```bash
# single fold
python scripts/train.py --stage train --fold 0 \
    --config pheno_mycn/configs/pheno_mycn_k6.yaml --path pheno_mycn_k6 --l 6

# all 10 folds
bash scripts/train.sh
```

Baselines use the same script with `configs/TransMIL.yaml` or `configs/DSMIL.yaml`.
The model class is chosen by `Model.name` in the config; `--l` sets the number of
GMM components K.

## Visualization & experiments

- [`visualization/`](visualization/README.md) — whole-slide heatmaps and
  phenotype-space maps (run from that directory; paths are placeholders).
- [`experiments/`](experiments/README.md) — survival, SHAP, latent-space,
  GMM-responsibility and cell-level analyses (set `PHENO_MYCN_ROOT`).

## Data

No patient data is included. [`data/README.md`](data/README.md) documents the
expected formats (UNI tile embeddings, cross-validation split CSVs, the survival
cohort table, and per-tile model outputs), and [`data/example/`](data/example/)
holds a synthetic split illustrating the layout.

## Authors

- **Dr Binghao Chai** — additional experiments; code review and refactoring of
  the core Pheno-MYCN pipeline.
  [bhchai.com](https://bhchai.com/) · [github.com/cbhindex](https://github.com/cbhindex)
- **Dr Olga Fourkioti** — core Pheno-MYCN model and training/inference pipeline.
  [github.com/olgarithmics](https://github.com/olgarithmics)

## License & attribution

Released under the **GNU General Public License v3.0** (see [`LICENSE`](LICENSE)).
GPL-3.0 is required because the attention-MIL backbone and the WSI utilities
derive from [CLAM](https://github.com/mahmoodlab/CLAM) (GPL-3.0). Third-party
components (CLAM, TransMIL, DSMIL, timm, nystrom-attention, UNI, HoverNet) and
their licenses are credited in [`NOTICE`](NOTICE).
