"""
Plug-and-play inference API for Pheno-MYCN.

``PhenoMYCNPredictor`` wraps the trained Pheno-MYCN model (CLAM-SB backbone +
auxiliary K=6 GMM phenotype branch) behind a small, framework-light interface.
It loads weights directly from the published PyTorch-Lightning checkpoint
without instantiating Lightning, so a prediction needs only ``torch`` at
runtime.

Given a slide's tile embeddings (UNI features, ``[n_tiles, 1024]``) it returns:

  * ``mycn_probability``  — slide-level P(MYCN-amplified);
  * ``predicted_label``   — 0 (non-amplified) / 1 (MYCN-amplified);
  * ``responsibilities``  — per-tile soft GMM responsibilities, ``[n_tiles, K]``
                            (the columns are manuscript Components 1..K);
  * ``hard_components``    — per-tile arg-max component, 1-indexed to match the
                            manuscript;
  * ``attention``         — per-tile MIL attention weights, ``[n_tiles]``;
  * ``anomaly_score``     — slide-level GMM free-energy.

Example
-------
>>> import torch
>>> from pheno_mycn import PhenoMYCNPredictor
>>> predictor = PhenoMYCNPredictor.from_pretrained()          # bundled K=6 fold-9 weights
>>> feats = torch.load("SLIDE_uni.pt")                        # [n_tiles, 1024]
>>> out = predictor.predict(feats)
>>> out["mycn_probability"], out["hard_components"][:5]

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                  Dr Olga Fourkioti  (https://github.com/olgarithmics)
Code review & refactor:  Dr Binghao Chai    (https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

from pathlib import Path

import numpy as np
import torch

from pheno_mycn.models.CLAM_SB import CLAM_SB

# Number of GMM components in the published model.
DEFAULT_K = 6


def default_checkpoint_path():
    """Return the path to the checkpoint bundled with the repository, if present.

    The representative weights live at
    ``<repo>/plug_and_play/weights/pheno_mycn_k6_fold9.ckpt``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = repo_root / "plug_and_play" / "weights" / "pheno_mycn_k6_fold9.ckpt"
    return ckpt


def _extract_model_state_dict(checkpoint):
    """Pull the CLAM-SB sub-state-dict out of a Lightning checkpoint.

    The training LightningModule stores the network as ``self.model``, so its
    weights appear under the ``model.`` prefix alongside (discarded) torchmetric
    buffers. This strips the prefix and keeps only the network weights.
    """
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
    if not model_state:
        # Already a bare CLAM-SB state dict.
        model_state = dict(state_dict)
    return model_state


class PhenoMYCNPredictor:
    """Lightweight predictor around a trained Pheno-MYCN (CLAM-SB + GMM) model."""

    def __init__(self, model, device, num_components=DEFAULT_K):
        self.model = model
        self.device = device
        self.num_components = num_components

    @classmethod
    def from_pretrained(cls, ckpt_path=None, device=None, num_components=DEFAULT_K, n_classes=2):
        """Build a predictor from a Pheno-MYCN checkpoint.

        Args:
            ckpt_path: path to the ``.ckpt`` file. Defaults to the bundled
                K=6 fold-9 weights.
            device: torch device string (e.g. ``"cuda"`` or ``"cpu"``).
                Defaults to CUDA when available.
            num_components: number of GMM components K (must match the checkpoint).
            n_classes: number of output classes (2: non-amp vs MYCN-amp).
        """
        if ckpt_path is None:
            ckpt_path = default_checkpoint_path()
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Pass an explicit `ckpt_path`, "
                f"or check out the repository so the bundled weights are present."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # The published checkpoint is a PyTorch-Lightning weights-only file; it
        # pickles config objects, so it must be loaded with weights_only=False.
        try:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch without the `weights_only` argument.
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")

        model_state = _extract_model_state_dict(checkpoint)

        model = CLAM_SB(l=num_components, n_classes=n_classes)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"[PhenoMYCNPredictor] WARNING: {len(missing)} missing keys, "
                  f"e.g. {missing[:4]}")
        if unexpected:
            print(f"[PhenoMYCNPredictor] WARNING: {len(unexpected)} unexpected keys, "
                  f"e.g. {unexpected[:4]}")

        model.to(device)
        model.eval()
        return cls(model, device, num_components=num_components)

    @staticmethod
    def _to_bag(features):
        """Coerce input tile embeddings to a 2-D float tensor ``[n_tiles, 1024]``."""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        features = features.float()
        if features.dim() == 3:  # [1, n_tiles, 1024]
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(
                f"Expected tile embeddings of shape [n_tiles, 1024] (or "
                f"[1, n_tiles, 1024]); got {tuple(features.shape)}."
            )
        return features

    @torch.no_grad()
    def predict(self, features):
        """Run Pheno-MYCN on one slide's tile embeddings.

        Args:
            features: tile embeddings, ``[n_tiles, 1024]`` (numpy array or torch
                tensor). These are the UNI features used to train the model.

        Returns:
            dict with keys ``mycn_probability``, ``predicted_label``,
            ``responsibilities``, ``hard_components``, ``attention`` and
            ``anomaly_score`` (numpy arrays / Python scalars).
        """
        bag = self._to_bag(features).to(self.device)

        out = self.model(bag)  # label is unused at inference

        y_prob = out["Y_prob"].squeeze(0)              # [n_classes]
        responsibilities = out["gmm_sores"].squeeze(0)  # [n_tiles, K]
        attention = out["scores"].squeeze(0)            # [n_tiles]
        energy = out["anomaly_loss"]

        hard_components = torch.argmax(responsibilities, dim=1) + 1  # 1-indexed

        return {
            "mycn_probability": float(y_prob[1].item()),
            "predicted_label": int(torch.argmax(y_prob).item()),
            "responsibilities": responsibilities.detach().cpu().numpy(),
            "hard_components": hard_components.detach().cpu().numpy(),
            "attention": attention.detach().cpu().numpy(),
            "anomaly_score": float(energy.item()) if torch.is_tensor(energy) else float(energy),
        }
