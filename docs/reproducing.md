# Reproducing the results

The experiments write to `data/` and `results/` by default; set `JAMLOC_DATA_DIR` or
`JAMLOC_RESULTS_DIR` to put them elsewhere. Run times depend on the hardware and the run settings. The section numbers
below point to [`results.md`](results.md), which records what each run found and the settings it
used. CPU and GPU runs can draw different samples and train different checkpoints.

## The whole pipeline

`paper_run` builds what the earlier paper drafts read. It generates any missing datasets and checkpoints, then
runs the detection, theory, margin, figure and archive stages, and packs `results/paper_run.zip`.
Data generation and training are meant for a GPU (Colab). The datasets have seven features
(including the empty-comb power) and most grid splits use 16-slot batches.

```bash
uv run python -m jamloc.experiments.paper_run --seeds 0 1 2
```

Recovery, the DMRS comparison and the latency benchmark are not part of it (see below). The current
paper, `conference-v9.tex`, reads the general training runs (section 26) and the paper v9 runs
(section 27) at the end of this file.

`full_run` is the earlier reference run (results.md §§3, 4 and 6). It trains four variants per
seed and builds `results/full_run.zip` with evaluation tables and plots, models and loss
histories, recovery curves for both channels (first detector seed) and the figures. The last
cell of [`notebooks/colab_setup.ipynb`](../notebooks/colab_setup.ipynb) runs it on Colab.

```bash
uv run python -m jamloc.experiments.full_run --seeds 0 1 2
```

## Single stages

Use the pipelines above for the reference settings, including their simulation batch sizes.

```bash
for s in train val test sched; do uv run python -m jamloc.dataset --split $s; done
uv run python -m jamloc.train --variant all --epochs 15 --seeds 0 1 2
uv run python -m jamloc.evaluate --split test                   # mean +- std over seeds
uv run python -m jamloc.evaluate --split sched                  # scheduling-aware jammer
uv run python -m jamloc.evaluate --split test --snr-db 15 25    # one SNR range only
uv run python -m jamloc.experiments.figures                     # explanatory figures + gallery
```

On Windows PowerShell, run the dataset loop as
`foreach ($s in "train","val","test","sched") { uv run python -m jamloc.dataset --split $s }`.

Figures are written as PDF (paper) and PNG (slides). `results/figures/index.html` shows each one
with a caption saying what it shows.

## Decoding with the map (§§6, 17.9, 18.3, 21)

The recovery experiment compares the standard receiver with receivers that use the jamming map.
Its LMMSE solves run on the CPU, so once `paper_run` has produced the models and `val`, run the
seeds as separate processes on a multi-core machine.

The recovery experiment takes its U-Net map from the general U-Net of section 26 by default. The
results before section 27 used the U-Net trained on the nominal mix, so the commands that repeat
them pass `--unet nominal`.

```bash
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 1 2              # nominal jammer
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --jammer pulsed --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --jammer hopping --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --snr-db 8 --seeds 0 1 2   # low-SNR case
uv run python -m jamloc.experiments.recovery --unet nominal --channel A
```

**What decoding does to the jammed bits (§19).** Wrong bits before decoding and the weight the
decoder gets on them, per receiver, for the showcase batches:

```bash
uv run python -m jamloc.experiments.decoding_bits               # results/site/decoding_bits.csv
```

**Jammer width (§22).** The runs above jam 4 of the 12 scheduled PRBs. `--jam-prb N` jams N PRBs
and writes `recovery_C_jam<N>` files. `experiments.width_sweep` then plots BLER against the width,
and the detector's Pd against the width of the test split's jammer blocks
(`results/width_sweep/`). The runs read the checkpoints through a `models` link in that folder
(`results/width_sweep/models` pointing to `results/models`).

```bash
JAMLOC_RESULTS_DIR=results/width_sweep uv run python -m jamloc.experiments.recovery --unet nominal --channel C \
    --jsr-db -3 3 9 --seeds 0 --max-mc-iter 25 --jam-prb 8 --attacks pilot data partial \
    --receivers standard lmmse p_r_e_soft p_r_e_soft_da true_soft true_soft_da
uv run python -m jamloc.experiments.width_sweep            # detection part, then the BLER figure
```

## Further studies

These reuse the nominal datasets and checkpoints where they can, and train what is missing.

```bash
# The link without a detector (§1) and the features (§2)
uv run python -m jamloc.experiments.ber_bler --channel C        # BER/BLER vs JSR (also --channel A)
uv run python -m jamloc.experiments.ber_bler --channel C --benign   # BLER vs SNR, no jammer
uv run python -m jamloc.experiments.feature_maps                # example feature maps

# Size and latency (§§5, 17.8); measure latency on the target CPU, not on Colab
uv run python -m jamloc.experiments.size_sweep                  # narrower U-Nets: train + evaluate (GPU)
uv run python -m jamloc.experiments.size_sweep --latency        # their CPU latency
uv run python -m jamloc.experiments.latency                     # CPU latency per slot, one thread

# The residual's theory (§8) and averaging it over several slots (§14)
uv run python -m jamloc.experiments.residual_theory             # analytic threshold and Pd vs simulation
uv run python -m jamloc.experiments.multislot                   # K-slot average: theory vs checks

# Unseen conditions and retraining (§§9, 10, 15)
uv run python -m jamloc.experiments.generalization              # long delays, 30 m/s, QPSK jammer, comb and hopping grids (GPU)
uv run python -m jamloc.experiments.robust_training             # retrain on mixed channels (GPU)
uv run python -m jamloc.experiments.attack_training --seeds 0 1 2   # retrain on a mix of attacks

# Baselines and calibration (§§11, 13)
uv run python -m jamloc.train --variant fused --arch pixel --epochs 15 --seeds 0 1 2   # per-cell MLP
uv run python -m jamloc.train --variant fused --epochs 15 --seeds 0 1 2 --noise-jitter-db 1
uv run python -m jamloc.evaluate --split test                   # adds the block test, MLP and jittered U-Net
uv run python -m jamloc.experiments.calibration                 # false alarms vs SNR and noise-estimate error
uv run python -m jamloc.experiments.calibration --estimator     # with the per-slot noise estimate

# Detection before damage (§§12, 17.6); needs the ber_bler runs
uv run python -m jamloc.experiments.detection_margin            # slot-level Pd vs unprotected BLER
uv run python -m jamloc.experiments.detection_margin --snr-db 15 11 8 5   # margin as the spare margin shrinks

# Intervals (§18) and a neighbor-cell interferer (§17.7)
uv run python -m jamloc.experiments.intervals                   # bootstrap intervals, paired differences, PRB-level F1
uv run python -m jamloc.experiments.neighbor                    # synchronized neighbor-cell UE
```

**One versus two DMRS symbols (§20).** This reuses the nominal test data and checkpoints from
`paper_run`, generates the missing two-DMRS splits, trains the matching P+R+E and P+E models and
writes detection and BLER tables to `results/dmrs_count/`.

```bash
uv run python -m jamloc.experiments.dmrs_count --seeds 0 1 2 --batch-size 64 --batches 25
uv run python -m jamloc.experiments.dmrs_count --small
```

**Known-symbol residuals after decoding and the refining receiver (§23).** Each command runs on
the checkpoints of `paper_run`. The recovery runs write per-seed CSVs, so seeds can run as
parallel processes; `--plot-only` then pools them and writes the paired differences.

```bash
# The null laws of both residuals, including long delays and 30 m/s (calibration panel)
uv run python -m jamloc.experiments.residual_theory --only-decoded --decoded-slots 2048

# Localization after decoding on replayed test grids, then the pooled table
for split in test sched combjam qpskjam neighbor hopjam pulsejam tonejam spoofjam spoofpart; do
  uv run python -m jamloc.experiments.decoded_map --split $split --seed 0 --per-point 8 --refine
done
uv run python -m jamloc.experiments.decoded_map --table results/decoded_map/*_refine.csv
uv run python -m jamloc.experiments.figures --only after_decoding

# Receivers with per-cell estimates and refining passes (TDL-C, 15 dB SNR, one seed per process)
RX="standard p_r_e_soft p_r_e_cell p_r_e_full true_soft true_cell true_full cell_soft cell_full p_r_e_soft_da true_soft_da"
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --attacks none pilot data partial \
    --jsr-db -15 9 --max-mc-iter 25 --receivers $RX
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --attacks pilot_whole_allocation \
    --jsr-db 3 9 --max-mc-iter 25 --receivers $RX
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --jammer hopping --jsr-db 9 \
    --max-mc-iter 25 --receivers $RX
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --noise-estimate comb \
    --attacks pilot_whole_allocation partial --jsr-db 3 9 --max-mc-iter 25 \
    --receivers standard p_r_e_soft p_r_e_soft_da p_r_e_full true_soft true_full cell_soft cell_full
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --snr-db 18 --mcs 13 --seeds 0 \
    --attacks none pilot data partial --jsr-db 3 9 --max-mc-iter 25 --target-block-errors 200 \
    --receivers standard lmmse true_soft true_cell cell_soft
```

**Shrinkage of the jamming estimate (§24).** The same commands with the shrinkage receivers. A
receiver added to a run with the same seed, cases and JSR grid decodes the same slots, so new rows
can be merged with earlier ones (`--receivers` replaces only their rows in the saved CSV).

```bash
SH="p_r_e_soft p_r_e_shrink p_r_e_shrink_full true_shrink_full cell_shrink_full"
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --attacks none pilot data partial \
    --jsr-db -15 9 --max-mc-iter 25 --receivers $SH
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --attacks pilot_whole_allocation \
    --jsr-db 3 9 --max-mc-iter 25 --receivers $SH p_r_e_shrink_da true_shrink cell_shrink
for jammer in hopping pulsed; do
  uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --jammer $jammer --jsr-db 9 \
      --max-mc-iter 25 --receivers $SH p_r_e_shrink_da
done
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --seeds 0 --noise-estimate comb \
    --attacks pilot_whole_allocation partial --jsr-db 3 9 --max-mc-iter 25 \
    --receivers p_r_e_soft_da p_r_e_shrink_full true_shrink_full cell_shrink_full
# 8 dB SNR ablation (per-PRB, per-cell and shrinkage, with channel or refining passes)
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --snr-db 8 --seeds 0 \
    --attacks partial pilot_whole_allocation --jsr-db 3 --max-mc-iter 25 \
    --receivers p_r_e_soft_da p_r_e_refine_prb p_r_e_cell_da p_r_e_full p_r_e_shrink_da \
    p_r_e_shrink_full true_soft true_shrink
# 16-QAM
uv run python -m jamloc.experiments.recovery --unet nominal --channel C --snr-db 18 --mcs 13 --seeds 0 \
    --attacks none pilot data partial --jsr-db 3 9 --max-mc-iter 25 --target-block-errors 200 \
    --receivers true_soft true_cell true_shrink cell_soft cell_shrink
```

**Resolution of the jamming estimate (§25).** Each run adds the study's receivers to the slots
of the §24 runs (same seed, cases and JSR grid), with `lmmse` and `p_r_e_shrink` as canaries. One
results folder per case, three seeds per folder, then merge and pool. Each folder reads the
checkpoints through a `models` link (`results/res_study/<case>/models` pointing to `results/models`).

```bash
NEW="nomap_prb nomap_prb2 nomap_prb4 nomap_cell nomap_shrink evm_prb evm_cell evm_shrink p_r_e_prb2 p_r_e_prb4"
run() { JAMLOC_RESULTS_DIR=results/res_study/$1 uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --seeds $2 --max-mc-iter 25 "${@:3}"; }
for s in 0 1 2; do
  run nom $s --attacks none pilot data partial --jsr-db -15 9 \
      --receivers $NEW true_prb_shrink true_shrink lmmse p_r_e_shrink
  run whole $s --attacks pilot_whole_allocation --jsr-db 3 9 --receivers $NEW lmmse p_r_e_shrink
  run hop $s --jammer hopping --jsr-db 9 --receivers $NEW true_prb_shrink lmmse p_r_e_shrink
  run pulse $s --jammer pulsed --jsr-db 9 \
      --receivers $NEW true_prb_shrink true_shrink lmmse p_r_e_shrink
done
# 16-QAM on the slots of the §24 16-QAM run (seeds 0 and 1)
for s in 0 1; do
  run qam16 $s --snr-db 18 --mcs 13 --attacks none pilot data partial --jsr-db 3 9 \
      --target-block-errors 200 --receivers nomap_prb nomap_prb2 nomap_prb4 nomap_cell \
      nomap_shrink true_prb_soft true_prb_shrink p_r_e_soft p_r_e_shrink evm_shrink lmmse true_shrink
done
# Merge with the §24 runs (every shared receiver must have the same counts), then pool and pair
M=results/res_study/merged/recovery
uv run python -m jamloc.experiments.resolution merge $M results/receiver_v7/recovery \
    results/res_study/{nom,whole,hop,pulse,qam16}/recovery
for j in nominal hopping pulsed; do
  JAMLOC_RESULTS_DIR=results/res_study/merged uv run python -m jamloc.experiments.recovery --unet nominal \
      --channel C --jammer $j --plot-only
done
JAMLOC_RESULTS_DIR=results/res_study/merged uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --snr-db 18 --mcs 13 --plot-only
uv run python -m jamloc.experiments.resolution table $M
uv run python -m jamloc.experiments.resolution table $M --qam
# JSR sweeps (new slots) and their figure
SW="standard lmmse nomap_prb nomap_shrink p_r_e_soft p_r_e_shrink evm_shrink true_shrink"
for s in 0 1 2; do
  run sweep_hop $s --jammer hopping --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
  run sweep_nom $s --attacks pilot partial --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
done
uv run python -m jamloc.experiments.resolution sweep results/res_study/sweep_nom/recovery \
    results/res_study/sweep_hop/recovery --out results/res_study/resolution_sweep
# 16-QAM under hopping and QPSK at 8 dB SNR (three seeds each)
for s in 0 1 2; do
  run qamhop $s --snr-db 18 --mcs 13 --jammer hopping --jsr-db 3 9 --target-block-errors 200 \
      --receivers standard nomap_prb nomap_cell nomap_shrink p_r_e_soft p_r_e_shrink true_soft \
      true_cell true_shrink true_prb_shrink
  run abl8 $s --snr-db 8 --attacks partial pilot_whole_allocation --jsr-db 3 --receivers \
      nomap_prb nomap_cell nomap_shrink p_r_e_soft p_r_e_cell p_r_e_shrink true_cell true_soft true_shrink
  run hop8 $s --snr-db 8 --jammer hopping --jsr-db 3 --receivers standard nomap_prb nomap_cell \
      nomap_shrink p_r_e_soft p_r_e_cell p_r_e_shrink true_soft true_cell true_shrink true_prb_shrink
done
# abl8 decodes the slots of the §24 8 dB ablation (canaries true_soft and true_shrink)
uv run python -m jamloc.experiments.resolution merge results/res_study/merged8/recovery \
    results/receiver_v7/ablation8/recovery results/res_study/abl8/recovery
JAMLOC_RESULTS_DIR=results/res_study/merged8 uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --snr-db 8 --plot-only
JAMLOC_RESULTS_DIR=results/res_study/hop8 uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --snr-db 8 --jammer hopping --plot-only
JAMLOC_RESULTS_DIR=results/res_study/qamhop uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --snr-db 18 --mcs 13 --jammer hopping --plot-only
# The estimates on one hopping slot
uv run python -m jamloc.experiments.figures --only jamming_estimates
```

**General training (section 26).** The experiment generates the missing splits, trains the
nominal, general and no-long-delay U-Nets with three seeds and evaluates them on eight test grids.
The decoding runs then put the general U-Net's map next to no map, the nominal U-Net and the true
map. These runs pass `--unet nominal`, so the `p_r_e_*` receivers use the nominal U-Net and the
`gen_*` receivers use the general one. The recovery script loads every detector, so the nominal
U-Net without the residual must be trained too.

```bash
uv run python -m jamloc.experiments.general_training --seeds 0 1 2
uv run python -m jamloc.train --variant no_residual --seeds 0 1 2 --epochs 15
RX="standard nomap_shrink p_r_e_soft p_r_e_shrink gen_soft gen_shrink true_shrink"
run() { mkdir -p results/gen_recovery/$1 && ln -sfn "$PWD/results/models" results/gen_recovery/$1/models
  JAMLOC_RESULTS_DIR=results/gen_recovery/$1 uv run python -m jamloc.experiments.recovery --unet nominal \
    --channel C --seeds $2 --max-mc-iter 25 --receivers $RX "${@:3}"; }
for s in 0 1 2; do
  run hop $s --jammer hopping --jsr-db 9
  run whole $s --attacks pilot_whole_allocation --jsr-db 3 9
  run nom $s --attacks none pilot data partial --jsr-db 9
  run qam16 $s --snr-db 18 --mcs 13 --attacks none pilot data partial --jsr-db 3 9 \
    --target-block-errors 200
  run qamhop $s --snr-db 18 --mcs 13 --jammer hopping --jsr-db 3 9 --target-block-errors 200
done
uv run python -m jamloc.experiments.general_training --decoding results/gen_recovery
```

**Paper v9 (section 27).** Every result of `conference-v9.tex` comes from these runs with the general
U-Net, which is the default of the recovery experiment. Each seed runs until 70 block errors or
160 batches of 64 slots, so a point pooled over three seeds has at least 210 block errors or 30 720 slots.

```bash
V9="--channel C --target-block-errors 70 --max-mc-iter 160"
run() { mkdir -p results/v9/$1 && ln -sfn "$PWD/results/models" results/v9/$1/models
  JAMLOC_RESULTS_DIR=results/v9/$1 uv run python -m jamloc.experiments.recovery $V9 --seeds $2 "${@:3}"; }
QPSK="standard lmmse nomap_prb nomap_cell nomap_shrink evm_prb evm_cell evm_shrink p_r_e_soft p_r_e_cell p_r_e_shrink true_prb_soft true_prb_shrink true_soft true_cell true_shrink"
QAM="standard nomap_prb nomap_cell nomap_shrink p_r_e_soft p_r_e_shrink true_soft true_cell true_shrink"
SW="standard nomap_prb evm_shrink p_r_e_shrink nomap_shrink true_shrink"
for s in 0 1 2; do
  run nom $s --attacks none pilot data partial --jsr-db 9 --receivers $QPSK
  run whole $s --attacks pilot_whole_allocation --jsr-db 3 9 --receivers $QPSK
  run pulse $s --jammer pulsed --jsr-db 9 --receivers $QPSK
  run hop $s --jammer hopping --jsr-db 9 --receivers $QPSK
  run qam16 $s --snr-db 18 --mcs 13 --attacks none pilot data partial --jsr-db 3 9 --receivers $QAM
  run qamhop $s --snr-db 18 --mcs 13 --jammer hopping --jsr-db 3 9 --receivers $QAM
  run sweep_hop $s --jammer hopping --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
  run sweep_nom $s --attacks pilot partial --jsr-db -9 -6 -3 0 3 6 9 --receivers $SW
  run passes $s --jammer hopping --jsr-db 9 --receivers p_r_e_shrink p_r_e_shrink_full
  run passes $s --attacks pilot_whole_allocation --jsr-db 9 --receivers p_r_e_shrink p_r_e_shrink_full
done
# Tables and curves
M=results/v9/merged/recovery
uv run python -m jamloc.experiments.resolution merge $M results/v9/{nom,whole,pulse,hop,qam16,qamhop}/recovery
for j in nominal hopping pulsed; do
  JAMLOC_RESULTS_DIR=results/v9/merged uv run python -m jamloc.experiments.recovery --channel C --jammer $j --plot-only
done
JAMLOC_RESULTS_DIR=results/v9/merged uv run python -m jamloc.experiments.recovery --channel C --snr-db 18 --mcs 13 --plot-only
JAMLOC_RESULTS_DIR=results/v9/merged uv run python -m jamloc.experiments.recovery --channel C --jammer hopping --snr-db 18 --mcs 13 --plot-only
uv run python -m jamloc.experiments.resolution table $M
uv run python -m jamloc.experiments.resolution table $M --qam
uv run python -m jamloc.experiments.resolution sweep results/v9/sweep_hop/recovery results/v9/sweep_nom/recovery --out results/v9/resolution_sweep
```

The plans behind each round of studies are in `plan-v2.md` to `plan-v6.md` and `review-v6.md`, and
`plan-v10.md` lists what the next revision of the paper should add.
