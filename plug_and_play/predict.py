"""
Command-line demo for the Pheno-MYCN plug-and-play predictor.

Runs the bundled, pretrained Pheno-MYCN model on one slide's tile embeddings
(UNI features, ``[n_tiles, 1024]``) and prints the slide-level MYCN prediction
together with the per-tile GMM phenotype assignment. Optionally writes the
per-tile responsibilities / attention to CSV.

Examples
--------
    # use the bundled K=6 fold-9 weights
    python plug_and_play/predict.py --features SLIDE_uni.pt

    # explicit checkpoint, save per-tile output, force CPU
    python plug_and_play/predict.py \\
        --features SLIDE_uni.pt \\
        --checkpoint plug_and_play/weights/pheno_mycn_k6_fold9.ckpt \\
        --output SLIDE_phenotypes.csv --device cpu

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                  Dr Olga Fourkioti  (https://github.com/olgarithmics)
Code review & refactor:  Dr Binghao Chai    (https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from pheno_mycn import PhenoMYCNPredictor


def load_features(path):
    """Load tile embeddings from a ``.pt`` (torch) or ``.npy`` (numpy) file."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="Pheno-MYCN plug-and-play prediction.")
    parser.add_argument("--features", required=True,
                        help="Path to one slide's tile embeddings (.pt or .npy), shape [n_tiles, 1024].")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a Pheno-MYCN .ckpt (default: bundled K=6 fold-9 weights).")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--output", default=None,
                        help="Optional CSV path for per-tile responsibilities / attention.")
    args = parser.parse_args()

    predictor = PhenoMYCNPredictor.from_pretrained(ckpt_path=args.checkpoint, device=args.device)

    features = load_features(args.features)
    result = predictor.predict(features)

    label_str = "MYCN-amplified" if result["predicted_label"] == 1 else "non-amplified"
    n_tiles = result["responsibilities"].shape[0]

    print("=" * 60)
    print(f"Slide features:        {args.features}")
    print(f"Tiles:                 {n_tiles}")
    print(f"P(MYCN-amplified):     {result['mycn_probability']:.4f}")
    print(f"Predicted MYCN status: {label_str}")
    print(f"GMM free-energy:       {result['anomaly_score']:.4f}")
    print("-" * 60)
    # Distribution of tiles over the K phenotype components (1-indexed).
    comps, counts = np.unique(result["hard_components"], return_counts=True)
    print("Per-tile dominant phenotype component (manuscript Components 1..K):")
    for c, n in zip(comps.tolist(), counts.tolist()):
        print(f"  Component {c}: {n:5d} tiles ({100.0 * n / n_tiles:5.1f}%)")
    print("=" * 60)

    if args.output:
        import csv
        K = result["responsibilities"].shape[1]
        with open(args.output, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["tile_index", "dominant_component", "attention"] + [f"resp_component_{k + 1}" for k in range(K)]
            writer.writerow(header)
            for i in range(n_tiles):
                row = [i, int(result["hard_components"][i]), float(result["attention"][i])]
                row += [float(x) for x in result["responsibilities"][i]]
                writer.writerow(row)
        print(f"Wrote per-tile output to {args.output}")


if __name__ == "__main__":
    main()
