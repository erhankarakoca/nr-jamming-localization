# Plan v7: what the paper should claim

24 November 2026. Summary of the drafts and the discussion after the Codex review. Read this
first when picking the work up on another device.

## Where things stand

- Two conference drafts are in [paper/drafts/v7](../paper/drafts/v7).
  - `draft-A.pdf` tells the receiver story. A learned map of the attacked cells is used softly by a
    standard receiver.
  - `draft-C.pdf` tells both stories and leads with the DMRS residual. Text that only C has is
    printed in teal.
  - Both come from `draft.tex`. Run `./build.sh` to rebuild them.
- An independent audit checked every number in the drafts against [results.md](results.md). The
  wrong statements it found are fixed.
- Hopping is left out of the drafts for now.

## What the reviews concluded

- Both drafts would likely get a weak reject from a demanding reviewer.
- The reason is novelty, not errors. Every part has precedent, for example Li et al. 2012
  (detect, erase, interpolate), Örnek and Kartal 2024 (EVM maps), Mehrabian and Kaddoum 2026
  (pilot-subspace detection) and DeepRx.
- No new method or simulator is needed. The claim has to change.
- Codex's full report is in [review-v7-codex.md](review-v7-codex.md).

## The proposed claim

To decode through selective jamming, the receiver must know where the jammer transmitted, cell by
cell. A map of where the damage shows, like EVM, or a map of whole PRBs is not enough.

## Two experiments to prove it (about 4 days)

1. Pilot jamming. The same soft receiver once with an EVM map and once with the U-Net map.
2. Hopping. The per-cell map against the best per-PRB receiver. The current per-PRB test may be
   unfair, because it averages the jamming power over the whole row
   (`coarsen_to_prb` in `src/jamloc/experiments/recovery.py`).

If experiment 2 shows the hopping gap comes only from that averaging, drop "cell by cell" and keep
the injection-versus-damage claim. If neither experiment holds up, submit the sharpened draft A to
a workshop.

## Facts that shape the claim

- The residual raises weak pilot detection (Pd 0.39 to 0.57 at -15 dB) but barely changes the
  block error rate.
- The residual is not new as an idea. A patent application and earlier papers use pilot residuals,
  and Liu et al. 2026 use residual diagnostics against pilot jamming. Its per-PRB false-alarm law
  is the defensible part.
- That law holds only with known noise power and a channel inside the delay subspace. A noise power
  2 dB too low gives 14.5% false alarms, and 1000 ns delay spread gives 62.9%.
- Under data jamming the classical block-test map used softly does as well as the U-Net (0.04%).
  The U-Net wins under partial-band jamming (0.02% against 7.7%).
- The nominal U-Net learned whole rows from its training data. On hopping it paints rows (F1
  0.595, 12.9% false alarms). The same network trained with hopping reaches 0.918.
- A map with one decision per PRB loses 52 to 57 points of block error rate under hopping.
- When 8 of 12 PRBs are jammed, too few clean coded bits remain, and even the true map loses 23%
  and 38% of the blocks.
- The full U-Net needs 0.68 ms per slot on one CPU core, more than the 0.5 ms slot. The 7,657
  parameter version needs 0.48 ms.

## Ideas considered and parked

- Normalize the residual by the median of the slot's other PRBs. It removes the known-noise
  condition (about 1 day). Needed only if the residual is the headline.
- Compare the two DMRS symbols to cancel the channel without a delay model. Ma et al. 2026 use a
  similar temporal idea.
- Codex's other directions (noise calibration under jammed references, predicting reliability
  instead of support, repairing partial DMRS replay, early warning under link adaptation). See
  [review-v7-codex.md](review-v7-codex.md).

## Open items

- Add citations for the prior residual work. Nguyen 2007, Sheng 2014, the patent application
  Zhong WO2022151467A1, and Liu et al. 2026,
  [10.1016/j.vehcom.2026.101086](https://doi.org/10.1016/j.vehcom.2026.101086).
- `docs/references.bib` has broken entries around lines 740 to 768 (missing commas, for example
  `amini2025deep`).
- Choose between draft A and draft C after the experiments.

## Next step

Run the two experiments, then choose the claim and rewrite one draft.
