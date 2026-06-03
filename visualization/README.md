# Visualization

Figure-generation scripts for the Pheno-MYCN manuscript: whole-slide heatmaps,
phenotype-space maps and cell-level feature maps. These render the per-tile
outputs produced at test time (GMM responsibilities, attention, free-energy,
projected features) over the original H&E WSIs and cell-feature tables.

## Scripts

| Script | Purpose |
|--------|---------|
| `create_map.py` | Whole-slide heatmaps of GMM-component responsibility, MIL attention and GMM free-energy, rendered over the WSI from saved per-tile `.pt` outputs and the per-slide `.h5` patch graph. |
| `prototype_map.py` | t-SNE map of representative tiles across the six GMM prototypes (phenotype space). |
| `prototype_analysis.py` | Representative-tile montage / embedding for a single GMM component, using its cell-level features. |
| `cell_level_maps.py` | Cell-level feature correlation maps relating projected tile features to HoverNet-derived cell descriptors for a component. |

## Support packages (CLAM-derived)

`vis_utils/`, `wsi_core/`, `clam_utils/`, `clam_datasets/` are the whole-slide
image utilities these scripts depend on. They are adapted from
[CLAM](https://github.com/mahmoodlab/CLAM) (Mahmood Lab, GPL-3.0) — see the
top-level `NOTICE`. Run the scripts from this directory so the packages are
importable, e.g.:

```bash
cd visualization
python create_map.py --path_file ... --path_WSI ... --path_graph ... --vis_folder ...
```

## Important: paths are placeholders

All input/output paths in these scripts (and in `heatmap_config_camelyon.yaml`)
have been replaced with `/path/to/...` placeholders. They expect:

- WSIs readable by OpenSlide;
- per-slide `.h5` patch graphs (tile coordinates);
- the per-tile outputs (`<slide>_gmm.pt`, `<slide>_gmm_feats.pt`, `<slide>_att.pt`,
  `<slide>_energy.pt`) — export these from the test loop (see the note in
  `pheno_mycn/models/model_interface.py`);
- HoverNet-derived `cell_info` tables for the cell-level scripts.

None of these patient-derived inputs are distributed in this repository (see the
top-level `README.md` and `data/README.md`). Edit the paths to point at your own
data before running. These are research figure scripts, provided for
transparency and reproducibility rather than as a turn-key tool.
