"""
Per-slide MIL dataset for the neuroblastoma cohort.

``CamelData`` serves one whole-slide bag per item: a tensor of pre-extracted
tile embeddings (UNI features, ``[n_tiles, 1024]``) loaded from ``<slide_id>.pt``
together with the slide-level MYCN label. Train / val / test membership and
labels are read from a per-fold cross-validation CSV (see ``data/README.md`` for
the expected schema).

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                  Dr Olga Fourkioti  (https://github.com/olgarithmics)
Code review & refactor:  Dr Binghao Chai    (https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import random
from pathlib import Path

import torch
import pandas as pd
import torch.utils.data as data


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        self.dataset_cfg = dataset_cfg

        # ---- data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        # ---- order
        self.shuffle = self.dataset_cfg.data_shuffle

        # ---- split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        features = torch.load(full_path)

        # ---- optional within-bag shuffle
        if self.shuffle is True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label, slide_id
