# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""Shared matplotlib style — Arial-like sans-serif for all survival panels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Arial-like sans family. Liberation Sans is the metric-equivalent open-source
# replacement for Arial; Nimbus Sans is the Helvetica equivalent. Both are
# preinstalled in the dpath conda env.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    # embed TrueType fonts in PDFs / PSs so vector text is editable downstream
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "svg.fonttype": "none",
    # readable defaults across all panels
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.titlesize": 13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
