# Plan for paper v11

Written on 26 November 2026, after the author read v10 and asked for five changes. This note
records what v11 does about each and how the runs are set up, so that the work can continue on
another machine.

## What the author asked for, and why

1. **Two DMRS symbols.** v10 uses one front-loaded DMRS symbol only, and its conclusion lists this
   as a limit. NR often adds a second DMRS symbol, and the U-Net should be trained for it.
2. **64-QAM**, if it does not take long. v10 has QPSK and 16-QAM.
3. **Drop Tables II and III.** Fig. 5 already shows the BLER, so the tables repeat it.
4. **The pulsed jammer in training?** Asked as a question.
5. **Fig. 5 looks irregular in panels (a) to (c)**, and the zero-error points look stuck at 1e-4.
   The author also wants the true map without shrinkage.

The author chose one U-Net for every configuration over separate side studies. A gNB runs one
detector for all its users, whatever DMRS configuration and modulation each one has, so the paper
can then claim one detector for all of them.

## Why Fig. 5 looks irregular

A triangle is not a measured BLER. When a receiver has no block error in 30 720 slots, the figure
draws the 95% upper bound, which is 1.2e-4 for every such point, so these points line up and look
like a floor. QPSK at 15 dB is so easy that the receivers that estimate the jamming power make 0
to 3 errors in 30 720 slots, so their curves follow chance. A point with one error (3e-5) even
lands below the triangles. More slots would not fix this at a sensible cost, since a BLER of 1e-5
needs about a million slots per point.

The fix is a harder operating point. A first try at 10 dB, where the baseline fails 0.23% of the
blocks without a jammer on random allocations (`results/ber_bler/benign_C.csv`), still gave 0 to 5
errors per seed at -9 dB JSR. The decoding runs use a fixed allocation of 12 PRBs, which fails less
often than the random ones, and the receivers that estimate the jamming power use the LMMSE
interpolator, which is better than the baseline's linear one. That run was stopped after 20
minutes. Each modulation now runs where the baseline fails about 0.8% of the blocks without a
jammer on random allocations, QPSK at 8 dB, 16-QAM at 15 dB and 64-QAM at 18 dB. These are
points where link adaptation would plausibly use these MCS.

## Decisions

| Item | Decision | Whose call |
| --- | --- | --- |
| U-Net | one U-Net for one and two DMRS symbols and QPSK, 16-QAM and 64-QAM | author |
| Pulsed jammer | stays out of training, as the one jammer the U-Net never saw | recommended, author did not object |
| Tables II and III | removed, results that appear only there move into the text with their confidence intervals | author |
| Operating points | QPSK at 8 dB (was 15), 16-QAM at 15 dB (was 18), 64-QAM at 18 dB | ours |
| True map without shrinkage | a new curve, the true map with a plain average over the marked cells of a PRB row (`true_soft`) | author asked, the receiver is ours |
| Noise power | still known, as in v10 | unchanged |

### Training data

- Conditions are every combination of the three training jammers (block, hopping block, pilot
  comb), the three channels of v10, three modulations and two DMRS configurations, 54 in all.
- QPSK draws its SNR from 5 to 25 dB and 16-QAM (MCS 13) from 10 to 25 dB, as in v10. 64-QAM uses
  MCS 17 (code rate about 0.43) and draws its SNR from 16 to 30 dB, where link adaptation would
  choose it over 16-QAM. A no-jammer sweep of the baseline receiver on TDL-C (random allocations,
  3 840 slots per point) chose MCS 17 over MCS 20, whose BLER levels off near 0.5% up to 28 dB.

  | MCS | BLER without a jammer |
  | --- | --- |
  | 5 (QPSK) | 0.85% at 8 dB, 0.23% at 10 dB, none at 15 dB (`benign_C.csv`) |
  | 13 (16-QAM) | 1.1% at 14 dB, 0.42% at 16 dB, 0.08% at 18 dB |
  | 17 (64-QAM) | 0.78% at 18 dB, 0.31% at 20 dB, 0.10% at 22 dB |
  | 20 (64-QAM) | 0.73% at 22 dB, 0.52% at 26 dB, 0.36% at 28 dB |
- The second DMRS symbol is `dmrs_additional_position = 1` (symbol 11, as Sionna places it). A
  pilot attack hits both DMRS symbols.
- 240 000 training slots and 30 000 validation slots (new splits `train_all` and `val_all`, new
  seeds). This is about 4 400 slots per condition against 6 700 in v10. The split is twice as large
  as v10's, so it should still fit the L4's memory.
- The same U-Net (66 553 parameters), the same training as section 26 of `results.md` (up to 80
  epochs, learning rate halved on a plateau, early stopping after 10 epochs), three seeds, one map
  threshold from `val_all` for every configuration.

### Detection (Table I)

Table I gets one row pair per DMRS configuration and columns for 64-QAM. The test grids are those
of v10 plus 64-QAM block and hopping, each with one and with two DMRS symbols. The v10 U-Net is
evaluated on the one-DMRS grids too, which shows what training on more configurations costs.

### Decoding

- Operating points on TDL-C, UE on PRBs 6 to 17, as in v10. QPSK at 8 dB, 16-QAM at 15 dB and
  64-QAM (MCS 17) at 18 dB, where the baseline fails about 0.8% of the blocks without a jammer.
- The cases and their commands are in `jamloc.experiments.paper_v11`.
- Receivers drawn in Fig. 5 are the baseline, no map with the PRB average, no map with shrinkage,
  the EVM map, the U-Net map, the true map with the PRB average (new) and the true map with
  shrinkage.
- JSR from -9 to +9 dB in 3 dB steps. Each of three U-Net seeds decodes until 70 block errors or
  160 batches of 64 slots, as in v10.
- The +9 dB cases that only Table II had (pilot only and data only of the block jammer, every
  pilot jammed, the no-map cell estimate and the PRB-resolution true map) are decoded again at the
  new QPSK point for the text.
- Two DMRS symbols are decoded with QPSK for the block jammer on pilots only, the hopping jammer on
  pilots and data, and every pilot jammed.

### Fig. 5

Two rows. The top row keeps the v10 layout (QPSK, then 16-QAM). The bottom row holds 64-QAM block
and hopping on pilots and data, and the two-DMRS panels. The exact panels are chosen once the
curves are in, so that each panel shows something the others do not. Dropping Tables II and III
frees about half a page, more than the second row takes.

### Other paper changes

- Fig. 2 (attack maps) and Fig. 4 (U-Net process) are drawn again with the new U-Net.
- Setup, training paragraph, Table I and the results text are updated. The conclusion no longer
  lists one DMRS symbol as a limit.

## Code changes

- `dataset.py`: the 64-QAM modulation condition, a DMRS axis, `ALL_CONDITIONS`, the splits
  `train_all` and `val_all`, and the 64-QAM and two-DMRS test grids.
- `general_training.py`: the new training tag and test grids, and a comparison with the v10 U-Net.
- `experiments/recovery.py`: a `--dmrs-additional-position` option (files get a `_dmrs2` suffix)
  and the new U-Net under `--unet all`.
- `experiments/resolution.py`: the v11 panels with the new curve.
- Tests for each, then `docs/results.md` section 30 and the commands in `docs/reproducing.md`.

## Compute

Data generation and training on a Colab L4 (12 vCPUs), about half a day. Decoding on the L4 or the
Mac, about another half day. The runs and the new models are committed with `git add -f`, as
before.

## Not in v11

- An estimated noise power, the U-Net score as a shrinkage prior and "Beyond One Attempt" stay
  open.
- The ResNet stays out of the paper (decided 26 Nov).
