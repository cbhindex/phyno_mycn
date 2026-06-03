"""
CLAM-SB backbone with the auxiliary GMM phenotype branch — the Pheno-MYCN model.

This is the core Pheno-MYCN network. It couples a single-branch
Clustering-constrained Attention MIL (CLAM-SB) classifier for slide-level MYCN
prediction with the auxiliary Gaussian mixture model branch
(``MILWithLearnableAnomalyDetection``) that defines the interpretable,
MYCN-associated tile-level phenotype space.

The attention-MIL backbone is adapted from CLAM (Mahmood Lab, GPL-3.0):
https://github.com/mahmoodlab/CLAM. The GMM phenotype branch and its
integration are the Pheno-MYCN contribution.

The ``forward`` pass returns, in a single dict, the slide-level logits /
probabilities, the attention scores, the per-tile GMM responsibilities
(``gmm_sores``), the projected GMM features, and the auxiliary GMM energy.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
Portions adapted from CLAM (Mahmood Lab, GPL-3.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pheno_mycn.models.utils import initialize_weights
from pheno_mycn.models.ad_loss import MILWithLearnableAnomalyDetection


class Attn_Net(nn.Module):
    """Attention network without gating (2 fc layers).

    Args:
        L: input feature dimension.
        D: hidden layer dimension.
        dropout: whether to use dropout (p = 0.25).
        n_classes: number of attention branches.
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """Attention network with sigmoid gating (3 fc layers).

    Args:
        L: input feature dimension.
        D: hidden layer dimension.
        dropout: whether to use dropout (p = 0.25).
        n_classes: number of attention branches.
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    """Pheno-MYCN: CLAM single-branch attention MIL + auxiliary GMM phenotype branch.

    Args:
        l: number of GMM components, K (the manuscript uses K=6). Also sizes the
            auxiliary phenotype space.
        gate: whether to use the gated attention network.
        size_arg: network-size preset ("small" or "big").
        dropout: whether to apply dropout (p = 0.25).
        k_sample: number of positive/negative tiles sampled for instance-level
            clustering supervision.
        n_classes: number of output classes (2 for MYCN-amp vs non-amp).
        instance_loss_fn: loss for instance-level clustering supervision.
        subtyping: whether this is a subtyping problem.

    Note:
        The original research code hard-coded ``l = 6`` inside ``__init__``,
        which silently ignored the constructor argument. That line has been
        removed so that ``l`` (the number of GMM components) is honoured as
        passed from the config. With the published config (``l: 6``) the model
        is identical to the one that produced the bundled K=6 weights.
    """

    def __init__(self, l, gate=True, size_arg="small", dropout=False, k_sample=2, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        # Auxiliary GMM phenotype branch over the raw (1024-d) tile embeddings,
        # with K = l components.
        self.ad_layer = MILWithLearnableAnomalyDetection(feature_dim=size[0], num_components=l)

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        """Instance-level evaluation for the in-the-class attention branch."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        """Instance-level evaluation for the out-of-the-class attention branch."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, feats, label=None, c=None, instance_eval=False, return_features=False, attention_only=False):
        device = feats.device
        feats = feats.squeeze(0)

        A, h = self.attention_net(feats)  # NxK

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarise label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # Auxiliary GMM phenotype branch over the raw tile embeddings.
        energy, qq, feats = self.ad_layer(feats.unsqueeze(0), label)

        results_dict = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'anomaly_loss': energy,
            'scores': A,
            'gmm_sores': qq,
            'feats': feats,
        }
        return results_dict
