# Reproducing the results

The experiments write to `data/` and `results/` by default; set `JAMLOC_DATA_DIR` or
`JAMLOC_RESULTS_DIR` to put them elsewhere. Run times depend on the hardware and the run settings. The section numbers
below point to [`results.md`](results.md), which records what each run found and the settings it
used. CPU and GPU runs can draw different samples and train different checkpoints.

## The whole pipeline

`paper_run` builds what the paper reads. It generates any missing datasets and checkpoints, then
runs the detection, theory, margin, figure and archive stages, and packs `results/paper_run.zip`.
Data generation and training are meant for a GPU (Colab). The datasets have seven features
(including the empty-comb power) and most grid splits use 16-slot batches.

```bash
uv run python -m jamloc.experiments.paper_run --seeds 0 1 2
```

Recovery, the DMRS comparison and the latency benchmark are not part of it (see below).

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

```bash
uv run python -m jamloc.experiments.recovery --channel C --seeds 0 1 2              # nominal jammer
uv run python -m jamloc.experiments.recovery --channel C --jammer pulsed --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --channel C --jammer hopping --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --channel C --snr-db 8 --seeds 0 1 2   # low-SNR case
uv run python -m jamloc.experiments.recovery --channel A
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
JAMLOC_RESULTS_DIR=results/width_sweep uv run python -m jamloc.experiments.recovery --channel C \
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

The plans behind each round of studies are in `plan-v2.md` to `plan-v6.md` and `review-v6.md`.
