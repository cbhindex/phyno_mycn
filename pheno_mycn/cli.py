"""
Command-line interface for Pheno-MYCN.

Installed as the ``pheno-mycn`` console command (and runnable as
``python -m pheno_mycn``). Currently exposes a single ``predict`` subcommand
that runs the pretrained model on one slide's tile embeddings and prints the
slide-level MYCN prediction plus the per-tile GMM phenotype assignment.

Examples
--------
    pheno-mycn predict --features SLIDE_uni.pt
    pheno-mycn predict --features SLIDE_uni.npy --output SLIDE_phenotypes.csv --device cpu
    pheno-mycn predict --features SLIDE_uni.pt --checkpoint path/to/model.ckpt
    pheno-mycn --version

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)
Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import argparse
import csv
from pathlib import Path

from pheno_mycn import __version__


def load_features(path):
    """Load tile embeddings from a ``.pt`` (torch) or ``.npy`` (numpy) file."""
    path = Path(path)
    if path.suffix == ".npy":
        import numpy as np
        return np.load(path)
    import torch
    return torch.load(path, map_location="cpu")


def run_prediction(features, checkpoint=None, device=None, output=None):
    """Load the predictor, run it on ``features`` and print a summary.

    Args:
        features: path to a slide's tile embeddings (.pt or .npy), [n_tiles, 1024].
        checkpoint: optional path to a Pheno-MYCN .ckpt (default: bundled weights).
        device: 'cuda' or 'cpu' (default: auto).
        output: optional CSV path for the per-tile responsibilities / attention.

    Returns:
        The result dict from :meth:`PhenoMYCNPredictor.predict`.
    """
    import numpy as np
    from pheno_mycn import PhenoMYCNPredictor

    predictor = PhenoMYCNPredictor.from_pretrained(ckpt_path=checkpoint, device=device)
    result = predictor.predict(load_features(features))

    label_str = "MYCN-amplified" if result["predicted_label"] == 1 else "non-amplified"
    n_tiles = result["responsibilities"].shape[0]

    print("=" * 60)
    print(f"Slide features:        {features}")
    print(f"Tiles:                 {n_tiles}")
    print(f"P(MYCN-amplified):     {result['mycn_probability']:.4f}")
    print(f"Predicted MYCN status: {label_str}")
    print(f"GMM free-energy:       {result['anomaly_score']:.4f}")
    print("-" * 60)
    comps, counts = np.unique(result["hard_components"], return_counts=True)
    print("Per-tile dominant phenotype component (manuscript Components 1..K):")
    for c, n in zip(comps.tolist(), counts.tolist()):
        print(f"  Component {c}: {n:5d} tiles ({100.0 * n / n_tiles:5.1f}%)")
    print("=" * 60)

    if output:
        K = result["responsibilities"].shape[1]
        with open(output, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["tile_index", "dominant_component", "attention"]
                + [f"resp_component_{k + 1}" for k in range(K)]
            )
            for i in range(n_tiles):
                row = [i, int(result["hard_components"][i]), float(result["attention"][i])]
                row += [float(x) for x in result["responsibilities"][i]]
                writer.writerow(row)
        print(f"Wrote per-tile output to {output}")

    return result


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pheno-mycn",
        description="Pheno-MYCN: interpretable MYCN-amplification prediction and "
                    "tile-level phenotype discovery for H&E neuroblastoma WSIs.",
    )
    parser.add_argument("--version", action="version", version=f"pheno-mycn {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_predict = sub.add_parser(
        "predict",
        help="Run the pretrained model on a slide's tile embeddings.",
        description="Run Pheno-MYCN on one slide's UNI tile embeddings ([n_tiles, 1024]).",
    )
    p_predict.add_argument(
        "--features", required=True,
        help="Path to the slide's tile embeddings (.pt or .npy), shape [n_tiles, 1024].")
    p_predict.add_argument(
        "--checkpoint", default=None,
        help="Path to a Pheno-MYCN .ckpt (default: bundled K=6 fold-9 weights).")
    p_predict.add_argument(
        "--device", default=None, help="'cuda' or 'cpu' (default: auto-detect).")
    p_predict.add_argument(
        "--output", default=None,
        help="Optional CSV path for per-tile responsibilities / attention.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        run_prediction(args.features, checkpoint=args.checkpoint,
                       device=args.device, output=args.output)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
