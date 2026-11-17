# Design: map-aware recovery and interpretation figures

## Why

1. **Recovery.** The detector is accurate, but a reviewer's first question will be what the map is
   good for. A receiver that uses the map to recover the link answers it with a BLER curve.
2. **Figures.** The project will be presented and the code shared. The current figures are mostly
   result curves (BER/BLER, Pd vs JSR). What is missing are figures that *explain*: what each attack
   does to the slot, how the DMRS residual works, why EVM reports pilot jamming at the wrong symbols,
   and what each detector actually outputs.

## Goals

- A map-aware receiver that (a) re-estimates the channel from the pilots the map says are clean,
  using the same channel model as the residual feature, and (b) erases the soft bits of flagged
  data cells.
- BLER vs JSR with no protection, with the detectors' maps, and with the true map.
- A set of interpretation figures produced by one command into `results/figures/` (PDF for the
  paper, PNG for slides), each with a short caption saying what it shows and how to read it.
- One local page that shows all figures with their captions.

## Non-goals and limits

- No new detector, no soft (probability-weighted) use of the map, no multi-antenna processing.
- Pilot jamming of the *whole* allocation cannot be repaired: no clean pilot is left. This is
  reported as a limit, not worked around.

## Anti-goals

- Code that is hard to explain in the presentation.
- Figures whose numbers cannot be regenerated from code and seeds.
- Showing recovery only with the true map: the detector's own maps must be in every comparison.

## Constraints

Same as `docs/plan.md`: Sionna tutorial style (one block per step), Windows/Linux/macOS/Colab,
tensors on `sionna.phy.config.device`, outputs only through `results_dir()`, figures through
`jamloc.plotting` (Okabe-Ito colours plus a marker and line style per curve, IEEE widths).

## Approaches considered (recovery)

1. **Erase only.** Erase flagged data cells, and all data in PRBs whose DMRS is flagged. Simple,
   but the data in pilot-jammed PRBs is lost.
2. **Erase and repair (chosen).** Also re-estimate the channel in pilot-jammed PRBs from the clean
   pilots, by LMMSE interpolation under the same channel model the residual feature uses (delays
   spread uniformly up to 3 µs). Larger gain against pilot jamming, and one channel model both
   detects the attack and repairs it.
3. Robust estimation without the map (e.g. outlier-rejecting pilot fit). Rejected: it would hide
   how much the map contributes.

## Recovery: the receiver

`src/jamloc/recovery.py`, class `MapAwareReceiver(tx)`, written like `receiver.py`:

```text
y, no, map -> repaired channel estimate -> LMMSE equalizer -> demapper
           -> erase LLRs of flagged data cells -> layer demapper -> TB decoder
```

- **Map input:** boolean cells `[B, n, 14]` over the allocation (cut from the 24 × 14 carrier map).
- **Channel repair (LMMSE interpolation):** raw LS estimates at the DMRS pilots
  (`features.dmrs_ls`). Clean pilots are those in PRBs whose DMRS cell is not flagged. With `R` the
  frequency correlation of the uniform-delay model (`tau_max` = 3 µs) over the allocation's
  subcarriers and `s² = N0 / |x_p|²` the LS noise variance:
  `h_hat = R[:, clean] (R[clean, clean] + s² I)⁻¹ h_ls[clean]` on every subcarrier, the same on
  all 14 symbols (one DMRS symbol, low mobility), with error variance per subcarrier
  `e_k = R[k, k] − R[k, clean] (R[clean, clean] + s² I)⁻¹ R[clean, k]`, passed to the LMMSE
  equalizer as `err_var`.
  *Why LMMSE and not a least-squares fit in the subspace:* removing a block of pilots makes that
  fit badly conditioned (noise gain up to ~10⁵ inside the allocation and ~10¹³ at its edge, checked
  numerically), and a single error variance would be wrong in the gap. With the per-subcarrier
  error variance, subcarriers that cannot be predicted get small LLRs by themselves, so repair
  turns smoothly into erasure where the gap is wider than the channel's coherence bandwidth. The
  uniform 3 µs model is a conservative prior (it assumes a short coherence bandwidth), so repair
  will reach further into a gap on TDL-A than on TDL-C; the results report that honestly.
- **No clean pilot at all** (every DMRS cell of the allocation flagged): there is nothing to
  interpolate from, so the receiver uses Sionna's LS estimate exactly as the unprotected receiver
  does, and only erases flagged data cells. The map-aware receiver is therefore never worse than
  no protection in this case.
- **Empty map:** the same LMMSE interpolation from all pilots. This is plotted as its own reference
  line ("LMMSE estimator, no map"), so the gain from the estimator and the gain from the map can be
  told apart.
- **Erasure:** LLRs of all data REs in flagged data cells are set to 0.
- **Maps from detectors:** scores of the seed-0 models thresholded at their validation 1%-false-
  alarm threshold. True map = labels.

## Recovery: the experiment

`python -m jamloc.experiments.recovery --channel A` (and `C`):

- UE on PRBs 6–17 (12 PRBs), SNR 15 dB. Jammer block of 4 PRBs at a random position inside the
  allocation. Attacks pilot / data / partial-band, JSR −15 … +9 dB in 3 dB steps.
- Same Monte-Carlo settings as the BER/BLER sweep (`sim_ber`, batch 64, up to 20 batches or 200
  block errors per point, `early_stop=False`).
- Five receivers: standard (no map); subspace estimator, no map; map from U-Net P+E; map from
  U-Net P+R+E; true map.
- Output: `results/recovery/recovery_{A,C}.csv` and a figure with one BLER-vs-JSR panel per attack.
- Limit case, reported as numbers: pilot jamming of the whole allocation (standard vs map-aware
  receiver with the true map; they should match, by the rule above).
- Needs the trained models in `results/models/` and the `val` split for thresholds.

## Interpretation figures

`python -m jamloc.experiments.figures` writes into `results/figures/`:

1. **attack_patterns.** Schematic 24 × 14 grids: allocation, DMRS row, and the jammed cells for
   no jammer, pilot, data and partial-band jamming. No simulation; this is "what the attacker does".
2. **system_diagram.** Block diagram of the link and the detector (UE, channels, jammer, receiver
   steps, the three features, U-Net, jamming map), for slides and the paper.
3. **residual_explained.** One slot, TDL-C, SNR 20 dB, without and with pilot jamming on 4 PRBs:
   magnitude of the LS pilot estimates across subcarriers with the subspace fit drawn over them,
   and below it the residual per PRB. Shows why the user's own channel cancels and the jammer
   does not.
4. **constellations.** Equalized QPSK symbols for no jammer, pilot jamming and data jamming
   (JSR 0 dB), with symbols from jammed PRBs and clean PRBs in different colours and markers.
   Shows that pilot jamming damages data symbols that were never jammed.
5. **feature_distributions.** From the test split: distributions of P, R and E (dB) for
   non-jammed and jammed scheduled cells, one column per attack, at moderate JSR (−6 … 0 dB).
   Shows which feature separates which attack and where each one is blind.
6. **detector_outputs.** Three example slots (pilot, data, partial; JSR −6 dB) × columns: true
   map, P+R+E, P+E, power only, EVM threshold, residual threshold, each thresholded at 1% false
   alarm. Cells coloured as hit, miss or false alarm.
7. **training_curves.** Training and validation loss per epoch for each variant and seed.
   `train.py` starts saving a per-run history CSV; the figure appears once histories exist (the
   next training run, ~7 min on Colab).
8. **recovery** (from the recovery experiment).

The existing figures (BER/BLER, feature maps, Pd vs JSR) stay where they are.

**Gallery:** the same command writes `results/figures/index.html`, one local page that shows
every figure (new and existing) with its caption. It needs no internet connection and opens in any
browser. Captions live in the code next to each figure, so they regenerate with it.

## Run order (one consistent reference run)

The local datasets predate the residual fix, and a new training run is needed anyway for the
training-curve histories. So:

1. Build and test everything locally on small runs: history saving in `train.py`, the recovery
   receiver and experiment, the figures and the gallery.
2. Extend `full_run` to also run the recovery experiment (both channels) and the figures command,
   so that its zip contains models, histories, evaluation, recovery curves, all figures and the
   gallery.
3. One Colab run of `full_run --seeds 0 1 2` (about 10–15 minutes) becomes the reference: every
   number in `docs/results.md` and every figure comes from that run's data and models.
4. Only the CPU latency benchmark and the BER/BLER sweep stay local (they don't depend on the
   trained models).

## Left to the builder

Panel sizes and styling within `jamloc.plotting`, the example slots' seeds and JSR/SNR, histogram
bins, whether the recovery receivers share Monte-Carlo batches, how the existing figures are pulled
into the gallery, and the exact layout of the gallery page.
