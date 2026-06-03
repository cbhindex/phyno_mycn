"""
PyTorch-Lightning data module for Pheno-MYCN.

``DataInterface`` dynamically loads a dataset class by name from the config
(``Data.dataset_name``) and exposes train / val / test dataloaders. For the
neuroblastoma cohort the dataset is :class:`pheno_mycn.datasets.camel_data.CamelData`,
which serves per-slide bags of pre-extracted UNI tile embeddings.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import inspect
import importlib

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,
                 dataset_name=None, **kwargs):
        """
        Args:
            train_batch_size: training batch size (1 for MIL bags).
            train_num_workers: dataloader workers for train/val.
            test_batch_size: test batch size (1 for MIL bags).
            test_num_workers: dataloader workers for test.
            dataset_name: name of the dataset module/class to load.
        """
        super().__init__()
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = self.instancialize(state='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size,
                          num_workers=self.train_num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                          num_workers=self.test_num_workers, shuffle=False)

    def load_data_module(self):
        """Import the dataset class ``pheno_mycn.datasets.<dataset_name>.<CamelName>``."""
        camel_name = ''.join([i.capitalize() for i in self.dataset_name.split('_')])
        try:
            self.data_module = getattr(
                importlib.import_module(f'pheno_mycn.datasets.{self.dataset_name}'), camel_name)
        except Exception:
            raise ValueError('Invalid Dataset File Name or Invalid Class Name!')

    def instancialize(self, **other_args):
        """Instantiate the dataset with matching parameters from ``self.kwargs``."""
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
