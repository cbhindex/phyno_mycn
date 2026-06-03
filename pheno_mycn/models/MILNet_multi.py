"""
Multi-class dual-stream MIL (MILNet_multi) baseline classifier.

The multi-class variant of the dual-stream MIL head used as the slide-level
MYCN-prediction baseline in the manuscript (config: ``DSMIL.yaml``). Reuses the
``FCLayer`` / ``BClassifier`` streams from :mod:`pheno_mycn.models.MILNet`.
Adapted from dual-stream MIL (Li et al., CVPR 2021):
https://github.com/binli123/dsmil-wsi.

The returned dict mirrors the Pheno-MYCN model so the shared training/inference
loop can drive it interchangeably; auxiliary-GMM fields are placeholders.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                  Dr Olga Fourkioti  (https://github.com/olgarithmics)
Code review & refactor:  Dr Binghao Chai    (https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pheno_mycn.models.MILNet import BClassifier, FCLayer


class MILNet_multi(nn.Module):
    def __init__(self, n_classes, thresh):
        super(MILNet_multi, self).__init__()
        self.i_classifier = FCLayer(1024, n_classes)
        self.b_classifier = BClassifier(1024, n_classes)
        self.n_classes = n_classes
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=1024)

    def forward(self, feats, label):
        feats = feats.squeeze(0)

        cl_feats, classes = self.i_classifier(feats)
        A, bag_vector, V = self.b_classifier(cl_feats, classes)

        C = self.fcc(bag_vector)
        logits = C.view(1, -1)

        max_prediction, index = torch.max(classes, 0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'anomaly_loss': 0,
            'max_prediction': max_prediction.unsqueeze(0),
            'scores': logits,
            'gmm_sores': logits,
        }
        return results_dict
