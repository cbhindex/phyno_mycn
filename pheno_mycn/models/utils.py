"""
Shared model utilities for Pheno-MYCN.

Currently this provides weight initialisation used by the attention-MIL
backbone. Adapted from CLAM (Mahmood Lab, GPL-3.0).

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import torch.nn as nn


def initialize_weights(module):
    """Xavier-initialise ``nn.Linear`` weights and reset batch-norm parameters."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
