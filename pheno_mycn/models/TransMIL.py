"""
TransMIL baseline classifier.

A transformer-based correlated multiple-instance learning model used as a
slide-level MYCN-prediction baseline against Pheno-MYCN. Adapted from the
official TransMIL implementation (Shao et al., NeurIPS 2021):
https://github.com/szc19990412/TransMIL.

The ``forward`` signature and the returned dict mirror the Pheno-MYCN model so
the shared training/inference loop (``model_interface``) can drive either model
interchangeably. The auxiliary-GMM fields are returned as placeholders here
because the baseline has no phenotype branch.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,       # Moore-Penrose iterations for the pseudo-inverse
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x), return_attn=False)
        return x


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator."""

    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, thresh):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, h, label):
        h = self._fc1(h)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # cls_token
        h = self.norm(h)[:, 0]

        # predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'anomaly_loss': 0,
            'scores': logits,
            'gmm_sores': logits,
            'feats': h,
        }
        return results_dict
