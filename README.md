# Narcissus Recreation Proposal Repo

This repository is set up as a starting point for a one- to two-page brief proposal about recreating the Narcissus research project.

## What Is Included

- `docs/brief_proposal.md`: a fill-in template with section headers and talking points
- `docs/source_notes.md`: short notes about the paper, repo, and likely reproduction focus
- `references/`: downloaded source material and a simple manifest
- `data/README.md`: notes on what dataset material is still needed

## Source Materials

This repo currently includes:

- the paper PDF for arXiv `2204.05255`
- a zip snapshot of the upstream `reds-lab/Narcissus` repository

## Intended Use

Use the proposal template as the file you actually fill in for the class submission. The rest of the repo is just there to keep the research context organized.

## Figure 3 Recreation

The Colab entry point is `code/colab_driver.ipynb`. It mounts Drive, runs smoke/validation checks, then launches a resumable CIFAR-10 sweep for the Figure 3 methods: BadNets, Blend, Label-Consistent, and NARCISSUS.

The same sweep can be run from the command line on a CUDA machine:

```bash
python3 code/run_grid.py \
  --root ./data \
  --out results/figure3_results.csv \
  --epochs 100 \
  --target-class 2 \
  --ratios 0,0.5,5,10,20,30,70 \
  --seeds 0,1,2 \
  --methods BadNets,Blend,Label-Consistent,Ours

python3 code/plot_figure3.py \
  --csv results/figure3_results.csv \
  --out figures/figure3.png
```

`run_grid.py` appends each completed run and skips existing rows on restart, so interrupted Colab sessions can continue from the last completed configuration.

## Publishing Status

The local git repository is initialized and ready for commit. A GitHub remote has not been created yet in this environment.
