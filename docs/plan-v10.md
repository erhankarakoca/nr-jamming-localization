# Plan for paper v10

This note records where the paper stands after v9 and what the next revision should add. It was
written on 25 November 2026, after reading v9 together, so that the work can continue on another
machine.

## Where things are

- The paper is `conference-v9.tex` in the Overleaf project, six pages long. Its figures are in
  `figures/figures/` and `figures/v9/` there.
- The code is on `main`. Sections 26 and 27 of [results.md](results.md) record the general U-Net and
  the v9 runs, and the end of [reproducing.md](reproducing.md) lists their commands.
- The trained models and the run outputs behind v9 are committed even though `results/` and `data/`
  are git-ignored. They are `results/models`, `results/v9`, `results/general_training`,
  `results/gen_recovery` and `results/figures`, with the validation splits `data/val.npz` and
  `data/val_gen.npz` that the map thresholds come from. A new machine can redraw every figure from
  them without training or decoding again. Use `git add -f` for any new file there that should be
  kept. The `models` links inside the run folders are not committed, so a new run folder needs its
  own link as in `reproducing.md`.

## What v9 is missing

1. **The U-Net is hardly described.** Three sentences in Section III-D give its inputs and its
   training mix. The paper does not say what it outputs, how it is trained or how it is built. It
   also shows no detection result at all, although section 26 of the results has F1 on every test
   grid.
2. **Fig. 2 does not explain how the estimates find the jammer.** The three estimates never decide
   yes or no. Each gives every cell a jamming power, and that power enters the LLRs through
   equation (2). The decoder trusts a cell with a high estimate less and a cell near zero fully, so
   the estimate finds the jammer softly and without a threshold. The paper never says this
   plainly. Fig. 2 also lacks the true jamming power, so a reader cannot see which estimate is
   right.
3. **Fig. 3 shows one jammer only.** It shows pilot jamming, which makes the EVM failure clear, but
   nothing about hopping or the pulsed jammer the U-Net never saw.
4. **Fig. 4 has no 16-QAM curves.** 16-QAM was decoded only at +3 and +9 dB JSR. With QPSK the
   U-Net curve lies almost on the curve without a map. 16-QAM is where the U-Net gains up to 12.6
   points, so its curves would be the strongest evidence for the map.
5. **Smaller gaps.** BLER is tested only on the nominal channel, although the U-Net was trained for
   long delays and 30 m/s. The noise power is assumed known. The paper says nothing about cost.

## Changes in order of priority

1. **A subsection on the U-Net detector.** Describe the network from `src/jamloc/model.py`
   (`TinyUNet`, 66,553 parameters) and its inputs, the received power, the residual (7) and the EVM
   of each cell, with masks of the allocation, the DMRS and the data. Say that it outputs one logit
   per cell and is trained against the attack map (3) with the loss in `src/jamloc/train.py`. Give
   the training data (`train_gen`, 120,000 slots over 18 conditions, JSR from -15 to +10 dB, SNR from 5 to 25 dB for QPSK and from 10 to 25 dB for 16-QAM), the
   plateau schedule with early stopping, and the threshold that marks 1% of the clean cells of
   `val_gen`. Add a small table of F1 and false alarms on each test grid, taken from the general
   column of the section 26 table, with the pulsed jammer as the unseen case.
2. **Soft detection, made visible.** Add one or two sentences to Section III that say the estimate
   itself locates the jammer softly. Add the true jamming power of each cell to Fig. 2
   (`figures.shrinkage_example`), and write a caption that says what the circles, the dashed line
   and the squares are. Add a small block diagram of the receiver that shows where the pilot
   weights, the map and the estimate enter on the way to the decoder. A TikZ drawing in Overleaf is
   enough.
3. **A new Fig. 3.** Make it one full-width figure (`figures.attack_maps`) with a column for each
   attack, for example a block on the pilots, the hopping jammer on pilots and data, and the pulsed
   jammer. The rows are the true map, the U-Net map, the EVM map and the shrinkage estimate drawn
   as a heat map. The shrinkage row shows the soft detection directly. The true map row repeats
   Fig. 1, so Fig. 1 can go.
4. **16-QAM curves.** Decode 16-QAM at 18 dB SNR with MCS 13 from -9 to +9 dB JSR, for a block on
   pilots and data and for the hopping jammer on pilots and data. Use the v9 `run` helper from
   `reproducing.md` with the receivers `standard nomap_prb nomap_shrink evm_shrink p_r_e_shrink
   true_shrink`, then add the panels to `SWEEP_PANELS` in `experiments/resolution.py`. The QPSK
   sweeps took a few hours, and these should take about as long.

   ```bash
   SW="standard nomap_prb nomap_shrink evm_shrink p_r_e_shrink true_shrink"
   for s in 0 1 2; do
     run sweep_qam $s --snr-db 18 --mcs 13 --attacks partial --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
     run sweep_qamhop $s --snr-db 18 --mcs 13 --jammer hopping --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
   done
   ```

5. **If space allows.** Decode at 1000 ns delay and at 30 m/s, add one sentence on an estimated
   noise power from the earlier comb runs, and add one sentence on cost. Shrinkage needs a few sums
   per PRB, while the U-Net has 66,553 parameters.

## Space

v9 fills six pages. Removing Fig. 1 frees room for the wider Fig. 3. Shortening "Beyond One
Attempt" or dropping the cell estimate rows from both tables would free more. The page limit of the
target conference decides how much of item 5 fits, and it is still open.

## Other open items

- Cite the earlier work on pilot residuals (Nguyen 2007, Sheng 2014, Liu 2026). The paper already
  cites `zhong2022jamming`.
- Three small code issues were left from the final review of the v9 branch. Default recovery runs
  decode the `gen_*` receivers, which now repeat `p_r_e_*`. `figures.shrinkage_example` fails if no
  hopping row is hit on 2 to 5 symbols. A missing `val_gen` split gives a bare file-not-found error.
- Writing rules for the paper and the docs. Use plain, connected sentences and one name for each
  idea, and explain every term. Avoid colons and semicolons in prose and never use the section
  sign. Keep captions to one or two sentences.
