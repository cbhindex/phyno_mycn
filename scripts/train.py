"""
Train / evaluate Pheno-MYCN (or a baseline) on the neuroblastoma cohort.

Examples
--------
Train Pheno-MYCN (K=6) on fold 0::

    python scripts/train.py \\
        --stage train --gpus 0 --fold 0 \\
        --config pheno_mycn/configs/pheno_mycn_k6.yaml \\
        --path pheno_mycn_k6 --l 6

Evaluate the checkpoints written for a fold::

    python scripts/train.py --stage test --fold 0 \\
        --config pheno_mycn/configs/pheno_mycn_k6.yaml --path pheno_mycn_k6 --l 6

The model class is selected by ``Model.name`` in the YAML config (CLAM_SB =
Pheno-MYCN, TransMIL, MILNet_multi). ``--l`` sets the number of GMM components K
(the manuscript uses K=6).

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pheno_mycn.datasets import DataInterface
from pheno_mycn.models import ModelInterface
from pheno_mycn.utils.utils import read_yaml, load_loggers, load_callbacks


def make_parse():
    parser = argparse.ArgumentParser(description="Train/evaluate Pheno-MYCN and baselines.")
    parser.add_argument('--stage', default='train', type=str, choices=['train', 'test'],
                        help="'train' to fit, 'test' to evaluate saved checkpoints.")
    parser.add_argument('--config', default='pheno_mycn/configs/pheno_mycn_k6.yaml', type=str,
                        help="Path to the YAML config.")
    parser.add_argument('--gpus', default=[0], nargs='+', type=int, help="GPU id(s).")
    parser.add_argument('--fold', default=0, type=int, help="Cross-validation fold index.")
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--path', default='pheno_mycn_k6', type=str,
                        help="Run name used to build the log/checkpoint directory.")
    parser.add_argument('--l', type=int, default=6, help="Number of GMM components K.")
    return parser.parse_args()


def main(cfg, args):
    # ---- reproducibility
    pl.seed_everything(cfg.General.seed)

    # ---- loggers and callbacks
    cfg.load_loggers = load_loggers(cfg, args)
    cfg.callbacks = load_callbacks(cfg)

    # ---- data
    DataInterface_dict = {
        'train_batch_size': cfg.Data.train_dataloader.batch_size,
        'train_num_workers': cfg.Data.train_dataloader.num_workers,
        'test_batch_size': cfg.Data.test_dataloader.batch_size,
        'test_num_workers': cfg.Data.test_dataloader.num_workers,
        'dataset_name': cfg.Data.dataset_name,
        'dataset_cfg': cfg.Data,
    }
    dm = DataInterface(**DataInterface_dict)

    # ---- model
    ModelInterface_dict = {
        'model': cfg.Model,
        'loss': cfg.Loss,
        'optimizer': cfg.Optimizer,
        'data': cfg.Data,
        'log': cfg.log_path,
        'epochs': cfg.General.epochs,
        'lr': cfg.Optimizer.lr,
        'path': args.path,
    }
    model = ModelInterface(**ModelInterface_dict)

    # ---- trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        gpus=cfg.General.gpus,
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=False,
        check_val_every_n_epoch=1,
    )

    # ---- train or test
    if cfg.General.server == 'train':
        trainer.fit(model=model, datamodule=dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(p) for p in model_paths if 'epoch' in str(p)]
        for path in model_paths:
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)


if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)

    cfg.config = args.config
    cfg.Model.thresh = args.l
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.Model.l = args.l

    main(cfg, args)
