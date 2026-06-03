# Pre-publication / confidentiality notes (maintainer)

This file is for the repository maintainers. It records what is withheld from the
**current** repository state until the manuscript is published, and how to
release it afterwards.

## What is withheld before publication

| Item | Where | How it is withheld |
|------|-------|--------------------|
| Trained model weights | `plug_and_play/weights/pheno_mycn_k6_fold9.ckpt` | untracked (`git rm --cached`) + ignored by the pre-publication `.gitignore` |
| Per-fold test metrics | `plug_and_play/weights/fold9_result.csv` | untracked + ignored by the pre-publication `.gitignore` |
| Headline quantitative results | `README.md`, `plug_and_play/README.md`, `plug_and_play/weights/README.md` | softened in prose ("reported in the manuscript") |

The **code** (model, training, inference, CLI, experiments, visualization) is not
confidential and stays in the repository. The weight/metric files remain **on
disk** (so local inference still works); they are just no longer tracked, so they
drop out of the latest commit. Keeping them in earlier Git history is acceptable.

## The two `.gitignore` variants

- **`.gitignore`** — *pre-publication* (active). Ignores the checkpoint and its
  metrics so they are not re-added.
- **`.gitignore.release`** — *post-publication*. Re-includes the checkpoint.

## Omit the files from the current tree (already staged)

The weights and metrics have been untracked with:

```bash
git rm --cached plug_and_play/weights/pheno_mycn_k6_fold9.ckpt \
                plug_and_play/weights/fold9_result.csv
```

Commit when ready to make the omission take effect in the latest state:

```bash
git commit -m "Withhold model weights and metrics until publication"
git push
```

## At publication

```bash
cp .gitignore.release .gitignore                 # re-include weights going forward
git add -f plug_and_play/weights/                # stage the withheld files
# optionally restore the exact metric numbers in the README files
git commit -m "Release pretrained Pheno-MYCN weights and metrics"
git push
```
