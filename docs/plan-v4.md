# Plan v4: smarter jammers, related work, independent review

18 November 2026, after plan v3. The threat model says the jammer knows the numerology and the
DMRS positions, but every attack so far jams whole PRBs on fixed symbols. Both the block test and
the U-Net exploit that structure, so an attacker that changes it is the most important honest
test left.

| # | Work | Answers | Compute |
| --- | --- | --- | --- |
| 1 | **Pilot-comb jammer** (`SimConfig.jammer_pilot_comb`): on the DMRS symbol only the pilot subcarriers are jammed. Same damage to the channel estimate for half the energy. | Does detection rely on the power of the whole DMRS cell, or on what actually hurts the estimate? | CPU, minutes |
| 2 | **Hopping jammer** (`SimConfig.jammer_hopping`): a new PRB block on every OFDM symbol. | Does the learned whole-PRB structure become a liability? | CPU, minutes |
| 3 | **Attack-mixed training** (`train_atk` / `val_atk`, `experiments/attack_training`): the fused U-Net retrained on batches cycling through nominal, hopping and comb jammers. | Can diversity restore the detector, as mixed channels did for unseen channels? | CPU, ~1 h |
| 4 | **Literature search**: noise uncertainty / SNR wall, RE-selective pilot jamming, hopping-jammer detection, any PRB x symbol interference localization that challenges the novelty claim. | Missing citations and novelty check. | independent reviewer |
| 5 | **Independent code review** of the round-3 code. | Bugs and statistical mistakes before the numbers go into the paper. | independent reviewer |
| 6 | Results, paper, README. | | |

Still out of scope: multiple receive antennas (map-aware IRC) and a jammer that is not aligned to
the OFDM symbols.

## Status (18 November 2026, morning)

- [x] 1-2. Pilot-comb and hopping jammers (`combjam`, `hopjam`). The residual keeps the fused
  detector sensitive to the comb jammer; the hopping jammer breaks every structure-based detector
  (U-Net F1 0.59, block test 0.47; per-cell MLP 0.68). Results section 15.
- [x] 3. Attack-mixed training: hopping 0.914, comb 0.950, matched 0.953 (3 seeds).
- [x] 4. Literature: 8 references added, every identifier resolved (`docs/related_work.md`, second
  search). The closest precedent at PRB x symbol resolution is a 2022 MSc thesis on unallocated
  resources; the contribution statement now says "including the scheduled allocation".
- [x] 5. Independent code review: 9 findings, all fixed or documented (margin split regenerated
  with small batches and split by batch; block-test vetoed cells keep score 0; Gamma tail; null
  split required; BLER read at the matching SNR; calibration edge cases and theory cutoff).
- [x] 6. Results, paper, README and gallery updated.
