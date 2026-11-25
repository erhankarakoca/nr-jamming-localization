# Paper revision v9: one U-Net, clearer text, smoother curves, explanatory figures

25 November 2026. Agreed with the author in conversation.

## Why

The author wants a six-page conference paper that he can read and defend easily. The current
draft (`conference-v8.tex` on Overleaf) mixes two U-Nets, its tables are hard to read, its
captions and results section are long, some BLER curves are irregular because a few points rest
on very few block errors, and the method has no figure that shows what the equations do.

## Goals

- The general U-Net (trained on `train_gen`) is the paper's method, and the paper does not
  mention the U-Net trained only on block jammers.
- Every result in the paper comes from one consistent set of runs on this machine, with the
  general U-Net.
- BLER curves are smooth: each point runs until 200 block errors or 30 000 slots.
- Tables have few rows, plain column names, and a one-sentence explanation of the metric.
- Captions have one or two sentences. The results section keeps only the main comparisons.
- Two new explanatory figures, both simulated from real slots:
  1. A worked example of shrinkage. One PRB row filled by a block jammer and one row hit by a
     hopping jammer on some symbols, with each cell's own estimate, the row mean and the shrunk
     value.
  2. Three maps of one slot under pilot jamming: where the jammer transmitted, what the general
     U-Net detects, and what the EVM marks.
- The text follows plain, grammatical, connected sentences (the author's style rules) and the
  practices of strong IEEE wireless papers (Codex review, `docs/writing-guide.md`).
- Six pages.

## Non-goals

- No new method and no new attack types.
- The earlier sections of `docs/results.md` stay as a record of past runs.

## Approach

- Code. `experiments.recovery` gets a `--unet` option, `general` by default, that chooses which
  training the `p_r_e_*` receivers' U-Net comes from (thresholds from `val_gen`). `nominal`
  remains for reproducing older results. The new figures go into `experiments.figures`.
- Runs. Rerun the paper's decoding table, the 16-QAM table, the BLER-versus-JSR curves and the
  data-aided passes on this GPU with the general U-Net, pooled over three seeds.
- Paper. A new file `conference-v9.tex` next to v8 on Overleaf. I write the text, and Codex
  reviews it against the writing rules and checks every number against the run outputs.

## Left to the implementation

- Which rows and columns each table keeps, decided once the numbers are in.
- Whether the existing figure of three jamming estimates on one hopping slot stays, depending on
  space.
