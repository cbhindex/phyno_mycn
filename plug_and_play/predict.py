"""
Backwards-compatible launcher for the Pheno-MYCN plug-and-play predictor.

The canonical command-line interface is now `pheno-mycn` (installed with the
package; see `pheno_mycn/cli.py`). This script remains so that

    python plug_and_play/predict.py --features SLIDE_uni.pt [--checkpoint ...] [--device ...] [--output ...]

keeps working; it simply forwards to `pheno-mycn predict`.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import sys

from pheno_mycn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict"] + sys.argv[1:]))
