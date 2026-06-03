"""
Auxiliary Gaussian mixture model (GMM) phenotype branch for Pheno-MYCN.

This module implements ``MILWithLearnableAnomalyDetection``, the auxiliary branch
that turns a bag of tile embeddings into an interpretable, MYCN-associated
phenotype space. A diagonal-covariance Gaussian mixture with ``num_components``
components (K) is fitted on projected tile embeddings from MYCN-amplified
training cases via a small number of MAP Expectation-Maximisation (EM) steps,
and then applied to tiles from both molecular subtypes:

  * during training (label == 1): returns the GMM free-energy (used as an
    auxiliary regulariser) together with the per-tile soft responsibilities;
  * at inference: returns the per-tile anomaly score (free-energy) and the
    per-tile soft responsibilities ``qq`` over the K components.

The soft responsibilities ``qq`` and their hard arg-max labels are the
phenotype representation analysed throughout the manuscript (Components 1..K).
The mixture parameters (``mu``, ``V_``, ``phi``) are registered as buffers-like
parameters with ``requires_grad=False``: they are updated in closed form by the
MAP M-step rather than by gradient descent, and they are initialised from
ProtoDiv clustering in the training loop (see ``model_interface``).

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILWithLearnableAnomalyDetection(nn.Module):
    """Auxiliary diagonal-covariance GMM branch over projected tile embeddings.

    Args:
        feature_dim: dimensionality of the (projected) tile embeddings, D.
        epsilon: numerical floor added to the covariance for stability.
        num_components: number of mixture components, K (the manuscript uses K=6).
    """

    def __init__(self, feature_dim, epsilon=1e-6, num_components=4):
        super(MILWithLearnableAnomalyDetection, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)

        # Mixture parameters. These are not trained by back-prop (requires_grad
        # is False); they are set by ProtoDiv initialisation and updated in
        # closed form by the MAP M-step below.
        # mu: component means [K, D]
        self.mu = nn.Parameter(0.1 * torch.randn(num_components, feature_dim), requires_grad=False)
        # V_: unconstrained pre-activation; the positive variance is softplus(V_)
        self.V_ = nn.Parameter(
            torch.log(torch.exp(torch.ones((num_components, feature_dim))) - 1),
            requires_grad=False,
        )
        # phi: mixture weights [K]
        self.phi = nn.Parameter(F.softmax(torch.randn(num_components), dim=-1), requires_grad=False)
        self.eps = epsilon

    def forward(self, feats, label=None):
        """Return the GMM free-energy (training) or anomaly scores (inference).

        Args:
            feats: tile embeddings of shape [B, N, D].
            label: slide-level label; the EM fit is only run for MYCN-amplified
                bags (label == 1) during training.

        Returns:
            (energy_or_score, responsibilities, projected_feats)
        """
        self.V = torch.clamp(F.softplus(self.V_), min=1e-6)  # ensure positive variance
        feats = self.fc(feats)

        if self.training and (label.item() == 1 if isinstance(label, torch.Tensor) else label == 1):
            pi, mu, Sigma, responsibilities, energy = self.compute_energy(feats, num_iters=5)
            return energy.mean() if energy.dim() > 0 else energy, responsibilities, feats
        else:
            anomaly_scores, qq = self.compute_anomaly_score(feats)
            return anomaly_scores if anomaly_scores.dim() > 0 else anomaly_scores.unsqueeze(0), qq, feats

    def map_m_step(self, feats, weight, tau=1):
        """MAP M-step: update the mixture parameters from responsibilities."""
        wsum = weight.sum(dim=1)  # [B, K]
        wsum_reg = wsum + tau  # regularisation term

        # Weighted sums
        wxsum = torch.bmm(weight.permute(0, 2, 1), feats)  # [B, K, D]
        wxxsum = torch.bmm(weight.permute(0, 2, 1), feats ** 2)  # [B, K, D]

        # Update parameters
        phi = wsum_reg / wsum_reg.sum(dim=1, keepdim=True)  # [B, K]
        mu = (wxsum + self.mu.unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1)  # [B, K, D]

        Sigma = (wxxsum + (self.V + self.mu ** 2).unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1) - mu ** 2  # [B, K, D]
        Sigma = Sigma + self.eps + 1e-3  # stronger regularisation

        # In-place closed-form parameter updates (no gradient).
        with torch.no_grad():
            self.mu.copy_(mu.mean(dim=0))
            self.V_.copy_(torch.log1p(torch.exp(Sigma.mean(dim=0)) - 1))

        return phi, self.mu, Sigma

    def compute_energy(self, data, num_iters=1):
        """Run ``num_iters`` MAP-EM steps; return parameters, responsibilities and energy."""
        B, N, d = data.shape

        for _ in range(num_iters):
            phi = self.phi / self.phi.sum()
            mu = self.mu.unsqueeze(0)
            Sigma = self.V.unsqueeze(0)

            mu = mu.expand(B, -1, -1)
            Sigma = Sigma.expand(B, -1, -1)

            z_mu = data.unsqueeze(2) - mu.unsqueeze(1)

            mahalanobis = torch.sum((z_mu ** 2) / (Sigma.unsqueeze(1)), dim=-1)
            log_det = torch.sum(torch.log(torch.clamp(Sigma, min=1e-6)), dim=-1)

            log_prob = -0.5 * (
                mahalanobis + log_det.unsqueeze(1)
                + d * torch.log(torch.tensor(2 * torch.pi, device=Sigma.device))
            )

            log_prob += torch.log(phi.unsqueeze(0))

            qq = F.softmax(log_prob / 20.0, dim=-1)  # temperature scaling

            energy = -torch.logsumexp(log_prob, dim=-1)

            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=1.0)

        return pi, mu, Sigma, qq, energy

    def compute_anomaly_score(self, data):
        """Compute per-tile responsibilities and the bag-level anomaly (free-energy) score."""
        B, N, d = data.shape

        phi = self.phi / self.phi.sum()  # [K]
        mu = self.mu.unsqueeze(0)  # [1, K, D]
        Sigma = self.V.unsqueeze(0)  # [1, K, D]

        # Expand for broadcasting
        mu = mu.expand(B, -1, -1)  # [B, K, D]
        Sigma = Sigma.expand(B, -1, -1)  # [B, K, D]

        # Mahalanobis distance and log-determinant
        z_mu = data.unsqueeze(2) - mu.unsqueeze(1)  # [B, N, K, D]
        mahalanobis = torch.sum((z_mu ** 2) / Sigma.unsqueeze(1), dim=-1)  # [B, N, K]
        log_det = torch.sum(torch.log(Sigma), dim=-1)  # [B, K]

        # Log-probabilities
        log_prob = -0.5 * (
            mahalanobis + log_det.unsqueeze(1)
            + d * torch.log(torch.tensor(2 * torch.pi, device=Sigma.device))
        )
        qq = F.softmax(log_prob / 20.0, dim=-1)  # temperature scaling

        # Energy (anomaly score)
        energy = -torch.logsumexp(log_prob, dim=-1)  # [B]

        return energy.mean(), qq
