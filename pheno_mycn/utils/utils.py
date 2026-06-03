"""
Training utilities for Pheno-MYCN: config loading, loggers and callbacks.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                  Dr Olga Fourkioti  (https://github.com/olgarithmics)
Code review & refactor:  Dr Binghao Chai    (https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

from pathlib import Path

import yaml
from addict import Dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def read_yaml(fpath=None):
    """Read a YAML config into an attribute-style ``addict.Dict``."""
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def load_loggers(cfg, args):
    """Create a CSV logger and set ``cfg.log_path`` to the per-fold run directory."""
    log_path = cfg.General.log_path
    path = args.path

    Path(log_path).mkdir(exist_ok=True, parents=True)

    version_name = Path(cfg.config).name[:-5]
    cfg.log_path = Path(log_path) / version_name / f'{path}' / f'fold{cfg.Data.fold}'
    print(f'---->Log dir: {cfg.log_path}')

    csv_logger = pl_loggers.CSVLogger(
        Path(log_path) / version_name, name=f'{path}', version=f'fold{cfg.Data.fold}',
    )
    return [csv_logger]


def load_callbacks(cfg):
    """Early-stopping and (during training) best/last model-checkpoint callbacks."""
    Mycallbacks = []
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.00, patience=cfg.General.patience, verbose=True, mode='min',
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train':
        Mycallbacks.append(ModelCheckpoint(
            monitor='val_loss',
            dirpath=str(cfg.log_path),
            filename='{epoch:02d}-{val_loss:.4f}',
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode='min',
            save_weights_only=True,
        ))
    return Mycallbacks
