"""Dataset definitions and the Lightning data module for Pheno-MYCN."""

from pheno_mycn.datasets.data_interface import DataInterface
from pheno_mycn.datasets.camel_data import CamelData

__all__ = ["DataInterface", "CamelData"]
