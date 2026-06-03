#!/usr/bin/env bash
# Train Pheno-MYCN (K=6) across the 10 cross-validation folds.
# Edit the Data paths in the config first (see pheno_mycn/configs/pheno_mycn_k6.yaml).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
STAGE=train                         # 'train' to fit, 'test' to evaluate
CONFIG=pheno_mycn/configs/pheno_mycn_k6.yaml
RUN=pheno_mycn_k6
K=6                                 # number of GMM components

for fold in $(seq 0 9); do
    python scripts/train.py \
        --stage "${STAGE}" \
        --gpus 0 \
        --fold "${fold}" \
        --path "${RUN}" \
        --config "${CONFIG}" \
        --l "${K}"
done
