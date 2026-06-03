""" Loss factory for Pheno-MYCN.

Pheno-MYCN and its baselines are trained with standard ``torch.nn``
classification losses (the published config uses ``CrossEntropyLoss``;
``BCEWithLogitsLoss`` is also supported). ``create_loss`` therefore instantiates
the loss named in the config directly from ``torch.nn``.

Note:
    The original research code routed through a large segmentation-loss zoo and
    the ``pytorch_toolbelt`` dependency, none of which is used by this
    classification task. That machinery has been removed; the supported losses
    are any zero-argument ``torch.nn`` loss (e.g. ``CrossEntropyLoss``,
    ``BCEWithLogitsLoss``).
"""
import torch.nn as nn


def create_loss(args):
    """Instantiate the loss named by ``args.base_loss`` from ``torch.nn``."""
    conf_loss = args.base_loss
    if hasattr(nn, conf_loss):
        return getattr(nn, conf_loss)()
    raise ValueError(
        f"Invalid loss '{conf_loss}'. Expected the name of a torch.nn loss, "
        f"e.g. 'CrossEntropyLoss' or 'BCEWithLogitsLoss'."
    )
