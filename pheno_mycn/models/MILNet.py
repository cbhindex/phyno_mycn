"""
Dual-stream MIL (DSMIL) building blocks and a binary MILNet baseline.

Adapted from the dual-stream MIL implementation (Li et al., CVPR 2021):
https://github.com/binli123/dsmil-wsi. Provides ``FCLayer`` and ``BClassifier``
(reused by ``MILNet_multi``) and a binary ``MILNet`` head. ``MILNet_multi`` is
the variant used as the slide-level MYCN-prediction baseline in the manuscript.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    """Instance-level linear classifier stream."""

    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    """Bag-level attention classifier stream."""

    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU(),
            )
        else:
            self.v = nn.Identity()

        # 1D convolution that can handle multiple classes (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # Sort class scores along the instance dimension and select critical instances.
        _, m_indices = torch.sort(c, 0, descending=True)  # N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # C x K
        q_max = self.q(m_feats)  # C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # N x C, unnormalised attention
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        B = torch.mm(A.transpose(0, 1), V)  # C x V, bag representation
        return A, B, V


class MILNet(nn.Module):
    """Binary dual-stream MIL head."""

    def __init__(self, n_classes, thresh):
        super(MILNet, self).__init__()
        self.n_classes = n_classes
        self.i_classifier = FCLayer(1024, 1)
        self.b_classifier = BClassifier(1024, 1)
        self.fcc = nn.Conv1d(1, 1, kernel_size=1024)

    def forward(self, feats, label):
        feats = feats.squeeze(0)

        feats, classes = self.i_classifier(feats)
        A, bag_vector, V = self.b_classifier(feats, classes)

        max_prediction, index = torch.max(classes, 0)
        C = self.fcc(bag_vector)

        Y_prob = torch.sigmoid(C)
        threshold = 0.5
        Y_hat = (Y_prob >= threshold).long()

        results_dict = {
            'logits': C,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'anomaly_loss': 0,
            'max_prediction': max_prediction.unsqueeze(0),
            'scores': C,
            'gmm_sores': C,
        }
        return results_dict
