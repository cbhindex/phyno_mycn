"""
Pheno-MYCN
==========

A weakly supervised digital-pathology framework that couples attention-based
multiple-instance learning for slide-level MYCN-amplification prediction with an
auxiliary Gaussian mixture model branch for interpretable tile-level phenotype
discovery in H&E whole-slide images of paediatric neuroblastoma.

The lightweight, Lightning-free inference API for the bundled pretrained model
is exposed here::

    from pheno_mycn import PhenoMYCNPredictor

``PhenoMYCNPredictor`` is resolved lazily, so importing it only requires
``torch`` (not PyTorch Lightning).
"""

import importlib

__version__ = "1.0.0"

__all__ = ["PhenoMYCNPredictor", "__version__"]


def __getattr__(name):
    if name == "PhenoMYCNPredictor":
        return getattr(importlib.import_module("pheno_mycn.inference"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
