"""Model definitions for Pheno-MYCN and its baselines.

Names are resolved lazily (PEP 562) so that importing a single model — or the
Lightning-free inference path — does not pull in PyTorch Lightning via
``ModelInterface``. Models are also discovered dynamically by name from the
config (``Model.name``) in
:meth:`pheno_mycn.models.model_interface.ModelInterface.load_model`.
"""

import importlib

_EXPORTS = {
    "ModelInterface": "pheno_mycn.models.model_interface",  # requires pytorch-lightning
    "CLAM_SB": "pheno_mycn.models.CLAM_SB",
    "TransMIL": "pheno_mycn.models.TransMIL",
    "MILNet": "pheno_mycn.models.MILNet",
    "MILNet_multi": "pheno_mycn.models.MILNet_multi",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
