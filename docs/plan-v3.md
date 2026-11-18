# Plan v3: fair baselines, calibration and detection before damage

18 November 2026. After plan v2 (theory, soft recovery, generalization, size) the weakest points a
reviewer would find are:

1. **The threshold baselines only score scheduled cells.** Jamming outside the allocation is
   invisible to them, so part of the gap to the U-Net (F1 0.95 vs 0.30) is by construction.
   Nothing shows how much of the U-Net's gain comes from the features and how much from looking
   at neighbouring cells.
2. **The U-Net's false-alarm rate changes with SNR** (2.0% at 5-15 dB, 0.3% at 15-25 dB with one
   threshold), although the introduction argues that a cell-level detector must hold its
   false-alarm rate at every SNR. The noise power is assumed known.
3. **No statement about the attacker's best option**: can a jammer damage the link and stay
   undetected?

| # | Work | Answers | Compute |
| --- | --- | --- | --- |
| 1 | **Block test**, a training-free detector that uses the attack structure and covers every cell: power averaged over the PRB's data symbols outside the allocation (exact Gamma null), the residual's exact p-value on scheduled DMRS cells, PRB-averaged EVM on scheduled data cells (null from benign validation slots per SNR bin). One threshold on `-log10 p`. | Is the deep network needed, or does a well-designed classical test do as well? | CPU, minutes |
| 2 | **Per-cell MLP**: the same inputs as the fused U-Net, 1x1 convolutions only (no neighbouring cells), same training. | How much of the gain is spatial context? | CPU, ~30 min |
| 3 | **Detection before damage**: slot-level Pd of the map at 1% slot false alarm vs the BLER of the unprotected link, for a jammer on the user's PRBs at 15 dB SNR. | Is there a JSR range where the attack hurts but is not seen? How much earlier than a CRC/KPI-based detector does the map see it? | CPU, minutes |
| 4 | **Calibration**: per-cell false alarm vs SNR with one threshold; thresholds per SNR bin (the gNB knows its SNR); false alarm and Pd when the noise power is misestimated by up to +-2 dB (theory for the residual, the trained U-Nets for the rest). | Does the false-alarm rate hold at every SNR and under noise-estimation error? | CPU, minutes |
| 5 | **K-slot integration** of the residual for a persistent jammer: exact threshold (the phase-type law with every rate repeated K times), Pd vs JNR for K = 1, 2, 4, 8, checked by Monte Carlo. | How far can a gNB push the residual's sensitivity by trading detection delay? | CPU, minutes |
| 6 | Results, figures, paper and README updated. | | |

Not in this round: a jammer that is not aligned to the OFDM symbols (needs a time-domain jammer
and a labelling rule for partly-hit symbols) and multiple receive antennas (map-aware IRC). Both
stay in the paper's limitations.

## Status (18 November 2026, morning)

- [x] 1. Block test: `baselines.block_test_scores`; F1 0.872 (U-Net 0.951; literature thresholds
  up to 0.30). Results section 11.
- [x] 2. Per-cell MLP: `model.PixelMLP`, `train --arch pixel`; F1 0.694. Section 11.
- [x] 3. Detection before damage: `experiments/detection_margin`, `margin` split; every attack
  flagged in 90% of slots at least 7.7 dB before 10% BLER. Section 12.
- [x] 4. Calibration: `experiments/calibration`. Thresholds per SNR bin hold 1% at every SNR; the
  learned detectors are very sensitive to an underestimated noise power, the residual follows
  its theory. Added: training with noise-estimate jitter (`train --noise-jitter-db 1`).
  Section 13.
- [x] 5. K-slot residual: `theory.slot_average`, `experiments/multislot`. Section 14.
- [x] 6. Results, figures gallery, README and paper updated.
