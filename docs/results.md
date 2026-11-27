# Results

November 2026. [`reproducing.md`](reproducing.md) lists the experiment commands. Reproducing the reported tables also
requires the recorded run settings, saved replay inputs and benchmark environment. The figure
gallery with captions is at `results/figures/index.html`.

**Setup.** 24-PRB carrier, 30 kHz SCS, PUSCH with QPSK and LDPC (MCS 5), one front-loaded DMRS
symbol, random allocation of 6–24 PRBs, TDL-A (100 ns) and TDL-C (300 ns), up to 3 m/s. Gaussian
jammer on a block of 2–12 PRBs anywhere on the carrier. JSR is per jammed resource element,
relative to the UE's data-RE energy (DMRS REs carry 3 dB more). Training: 40 000 slots; validation:
5 000; test: 31 000 (grid over JSR −15…+9 dB, SNR drawn from 5–25 dB); scheduling-aware test:
16 200. All splits use separate seed ranges.

**Reference run (sections 3, 4 and 6):** `full_run --seeds 0 1 2` on a Colab T4 GPU through the
Colab CLI. The run generated data on the GPU and trained each U-Net variant with three seeds.
Data generation, training and evaluation took about 9 minutes as reported for that run.
Detection values are **mean ± standard deviation over the three seeds**, each with its own
validation thresholds. The command's recovery used the first seed. Later recovery tables
starting in §17.9 pool three seeds. Sections 2 and 5 were run on a MacBook Pro M1 Pro CPU.

The summaries below describe the earlier runs through v5. See §§17–18 for rebuilt results
and CPU timing.

**Second round** (`docs/plan-v2.md`, also on the Colab T4, with the reference datasets and
checkpoints used for comparison; latency on the Mac):
- **Theory of the residual (section 8):** under the ideal null, its distribution is independent
  of channel and SNR. The analytic 1% threshold gave 0.91–0.92% benign false alarms across the two
  channels in this run. The measured Pd and leakage into neighbouring PRBs were close to the
  theoretical values.
- **Recovery (section 6):** the map is used softly (it says *where*; the receiver estimates *how
  strong*), which removes the BLER penalty hard erasure paid at weak jamming, and is compared
  with DMRS-only interference estimation and the classic power-test receiver.
- **Size (section 5):** the 7.7k-parameter U-Net loses 0.008 F1. Feature extraction and the
  frozen forward pass total 0.43 ms on one core.
- **Unseen conditions (sections 9, 10):** a QPSK jammer changes nothing; long delays and fast users
  raise false alarms, and training on mixed channels removes most of that.

**Third round** (`docs/plan-v3.md`, all on the Mac CPU with the same datasets and models):
- **Fair baselines (section 11):** a training-free block test that covers every cell with exact
  or benign-calibrated p-values of PRB-level statistics reaches F1 0.872, against 0.951 for the
  U-Net and at most 0.30 for the literature thresholds. The U-Net's lead is mostly on pilot
  jamming. Without spatial context (per-cell MLP) the learned detector drops to 0.694.
- **Detection before damage (section 12):** with the jammer on the user's PRBs at 15 dB SNR,
  every attack is flagged in 90% of slots at least 8.2 dB of JSR before it pushes the BLER of the
  unprotected link to 10%.
- **Calibration (section 13):** thresholds per SNR bin give the U-Net 0.6-1.4% benign false
  alarms across the tested bins. Underestimated noise power raises false alarms, and training
  with a jittered noise estimate reduces the effect. The residual's measured false-alarm rates
  are close to the theoretical rates.
- **Smarter jammers (section 15, fourth round):** a jammer on the pilot subcarriers only is
  still seen through the residual; a jammer that hops PRBs every symbol breaks every detector
  that learned or assumes whole-PRB blocks (U-Net F1 0.59). Mixed training raises hopping F1
  to 0.914 and leaves matched overall F1 close to the nominal model's. Multi-tone F1 falls
  in both the earlier and rebuilt comparisons (§§16 and 17.2).
- **Several slots (section 14):** with a fixed allocation and independent noise, the average
  follows the ideal null law. With stable interference, 8 slots lower the JNR for 90% detection
  by about 6.2 dB in this run.

## 1. What each attack does to the link

`results/ber_bler/ber_bler_{A,C}.pdf` (Colab GPU, up to 12 800 slots or 100 block errors per point;
bands are 95% confidence intervals, hollow triangles mark points without any block error at their
95% upper bound). SNR 15 dB, the jammer covers the whole allocation.

| Attack | 50% BLER, TDL-A: JSR / slot energy ratio | 50% BLER, TDL-C: JSR / slot energy ratio |
| --- | --- | --- |
| Pilot | +3.6 dB / **−7.5 dB** | +3.5 dB / **−7.7 dB** |
| Data | −2.9 dB / −2.9 dB | −2.7 dB / −2.7 dB |
| Partial-band | −3.9 dB / −3.5 dB | −4.1 dB / −3.8 dB |

Per resource element, pilot jamming needs a higher JSR, but it only occupies 1 of 14 symbols. Per
unit of jammer energy it is the most efficient attack: it reaches 50% BLER with 4–5 dB less energy
than data or partial-band jamming, in line with Clancy's pilot-jamming result. Without a jammer,
BLER is 1/12 800 (0.0078%) on each channel at this SNR.

## 2. What the features see

`results/feature_maps.pdf`, and the explanatory figures `residual_explained`, `constellations` and
`feature_distributions`. Example means over the scheduled cells, JSR 0 dB, SNR 20 dB, jammer on the
whole allocation (linear, noise-normalized):

| Condition | DMRS residual R | EVM E (data cells) | Power P |
| --- | --- | --- | --- |
| No jammer | 1.0 | 1.3 | 105 |
| Pilot | **108** | 15 (wrong symbols) | 105 (high only on the DMRS row) |
| Data | 1.0 (blind) | **56** | **193** |
| Partial-band | **108** | **66** | **200** |

This matches Table 2 of the proposal. Pilot jamming shows up in the EVM of the *data* symbols,
which is why EVM alone reports it at the wrong symbols. Two corrections to Table 2: without
jamming, normalized EVM is about 1 only at moderate SNR (about 2 at 25 dB and 4 at 30 dB on TDL-C,
consistent with channel-estimation error omitted from the variance model, issue #6), and the
residual leaks into the PRBs next to a jammed block (up to about 6 dB within two PRBs). Under the
ideal null in §8, the linear mean residual is one on each PRB. Individual residuals fluctuate,
and channel energy outside the retained subspace raises the mean.

## 3. Detection: Pd with a 1% per-cell false-alarm target

Sections 3–6 report the earlier run through v5. Later paper and site values use §§17–20 where
the corresponding experiment was rerun.

`results/eval/test_pd_vs_jsr_{scheduled,all}.pdf` (bands: ±1 std over seeds). False alarm
measured on the test set: 1.0–1.3% for every U-Net. Pd over the jammed cells inside the
allocation, TDL-A and TDL-C together:

| Attack | Detector | JSR −15 dB | −9 dB | −3 dB | +3 dB |
| --- | --- | --- | --- | --- | --- |
| Pilot | **U-Net, P+R+E** | **0.62 ± 0.02** | **0.88 ± 0.01** | **0.97 ± 0.00** | 0.98 ± 0.00 |
| | U-Net, P+E | 0.43 ± 0.02 | 0.80 ± 0.02 | 0.95 ± 0.01 | 0.98 ± 0.01 |
| | U-Net, power only | 0.25 ± 0.04 | 0.60 ± 0.03 | 0.85 ± 0.01 | 0.96 ± 0.00 |
| | U-Net, EVM only | 0.14 ± 0.02 | 0.38 ± 0.03 | 0.66 ± 0.02 | 0.79 ± 0.02 |
| | Residual threshold | 0.10 | 0.33 | 0.60 | 0.86 |
| | Residual, exact p-value | 0.10 | 0.33 | 0.61 | 0.87 |
| | Power threshold (Li et al.) | 0.01 | 0.01 | 0.04 | 0.28 |
| | EVM threshold | 0 | 0 | 0 | 0 |
| | Block test (no training) | 0.34 | 0.62 | 0.85 | 0.96 |
| | Per-cell MLP, P+R+E | 0.08 ± 0.01 | 0.37 ± 0.02 | 0.73 ± 0.01 | 0.93 ± 0.00 |
| Data | **U-Net, P+R+E** | **0.78 ± 0.01** | **0.96 ± 0.00** | 0.99 ± 0.00 | 1.00 ± 0.00 |
| | U-Net, P+E | 0.74 ± 0.01 | 0.96 ± 0.00 | 0.99 ± 0.00 | 1.00 ± 0.00 |
| | U-Net, power only | 0.55 ± 0.04 | 0.84 ± 0.02 | 0.95 ± 0.01 | 0.99 ± 0.00 |
| | U-Net, EVM only | 0.47 ± 0.02 | 0.75 ± 0.02 | 0.92 ± 0.01 | 0.99 ± 0.00 |
| | EVM threshold | 0 | 0.04 | 0.21 | 0.43 |
| | Power threshold (Li et al.) | 0.02 | 0.01 | 0.02 | 0.13 |
| | Residual threshold | 0 | 0 | 0 | 0 |
| | Block test (no training) | 0.34 | 0.71 | 0.91 | 0.98 |
| | Per-cell MLP, P+R+E | 0.01 ± 0.00 | 0.15 ± 0.01 | 0.50 ± 0.01 | 0.77 ± 0.01 |

Cell-level F1 at the same thresholds:

| Detector | Pilot | Data | Partial | All slots |
| --- | --- | --- | --- | --- |
| **U-Net, P+R+E** | **0.834 ± 0.010** | **0.957 ± 0.001** | **0.961 ± 0.002** | **0.951 ± 0.002** |
| U-Net, P+E | 0.757 ± 0.013 | 0.952 ± 0.002 | 0.960 ± 0.001 | 0.946 ± 0.001 |
| U-Net, power only | 0.817 ± 0.007 | 0.920 ± 0.003 | 0.918 ± 0.004 | 0.913 ± 0.003 |
| U-Net, EVM only | 0.459 ± 0.041 | 0.792 ± 0.003 | 0.782 ± 0.001 | 0.772 ± 0.001 |
| EVM threshold | 0 | 0.324 | 0.319 | 0.303 |
| Residual threshold | 0.584 | 0 | 0.057 | 0.058 |
| Residual, exact p-value | 0.588 | 0 | 0.058 | 0.059 |
| Power threshold (Li et al.) | 0.209 | 0.128 | 0.106 | 0.120 |
| Block test (no training, section 11) | 0.512 | 0.912 | 0.880 | 0.872 |
| Per-cell MLP, P+R+E (section 11) | 0.383 ± 0.002 | 0.711 ± 0.001 | 0.729 ± 0.000 | 0.694 ± 0.000 |
| U-Net, P+R+E, noise jitter (section 13) | 0.822 | 0.956 | 0.958 | 0.948 ± 0.001 |

By SNR (overall F1, `test_snr5-15_*` and `test_snr15-25_*`): P+R+E 0.916 / 0.987, P+E
0.914 / 0.979, power only 0.867 / 0.959, EVM only 0.707 / 0.834 for SNR 5–15 dB / 15–25 dB.

**Reading** (illustrated by `detector_outputs` and `feature_distributions`).
- The DMRS residual is what makes weak pilot jamming detectable: at −15 dB it raises Pd from
  0.43 (P+E) to 0.62, and pilot-attack F1 from 0.757 to 0.834.
- For data and partial-band jamming, P+R+E and P+E are equal, as expected: the residual cannot
  see an attack that leaves the DMRS alone.
- Each threshold baseline sees only half of the problem: the EVM threshold cannot score the DMRS
  symbol (and flags data cells of pilot-jammed PRBs instead), and the residual threshold is blind
  to data jamming.
- The classic per-cell power test (Li et al.; power relative to the slot's median scheduled cell)
  is weak at every JSR: one cell's power cannot tell jamming from the UE's own fading, which
  varies by more than 10 dB across PRBs. The power-only U-Net succeeds with the same input because
  it compares each cell with the other symbols of the same PRB, which share its channel.
- Scoring the residual by its exact tail probability instead of its raw value changes little
  here (F1 0.059 vs 0.058). Like every detector in this table it is thresholded on the
  validation set, whose non-jammed DMRS cells include neighbours of jammed PRBs: leakage raises
  that threshold (to p = 10^-10.3 instead of 10^-2). At the analytic threshold, the false-alarm
  rate is 0.95% in benign slots but 4.5% over all non-jammed DMRS cells. The value of the exact
  distribution is the training-free threshold in benign conditions (section 8); in jammed slots
  the leakage must be handled, which the U-Net does.

## 4. Robustness: scheduling-aware jammer

`results/eval/sched_pd_vs_jsr_scheduled.pdf`. The jammer hits exactly the scheduled PRBs. No
U-Net saw this attacker during training. (Numbers from the second-round regeneration of this
split, batch size 64 instead of 128, so other slots than the reference run; the differences are
within about one standard deviation.)

| Detector | F1 pilot | F1 data | F1 all | Pd pilot, −15 dB | Pd data, −15 dB |
| --- | --- | --- | --- | --- | --- |
| **U-Net, P+R+E** | **0.880 ± 0.011** | 0.930 ± 0.011 | 0.934 ± 0.011 | **0.59 ± 0.01** | 0.50 ± 0.02 |
| U-Net, P+E | 0.789 ± 0.015 | **0.937 ± 0.011** | **0.937 ± 0.011** | 0.35 ± 0.03 | **0.55 ± 0.04** |
| U-Net, power only | 0.815 ± 0.009 | 0.897 ± 0.008 | 0.888 ± 0.008 | 0.21 ± 0.03 | 0.40 ± 0.04 |
| U-Net, EVM only | 0.473 ± 0.031 | 0.712 ± 0.063 | 0.696 ± 0.057 | 0.05 ± 0.02 | 0.19 ± 0.01 |
| Power threshold (Li et al.) | | | 0.030 | | |

The slot-relative power test fails here by construction: its reference, the median scheduled
cell, is itself jammed when the jammer covers the whole allocation.

**Reading.** When the whole allocation is jammed there is no clean neighbouring PRB to compare
against. For **pilot jamming** the fused detector is hardly affected (Pd 0.59 at −15 dB, against 0.62
in the test set), because the residual is an absolute measure; without it Pd falls from 0.43 to
0.35. For **data jamming** every U-Net loses some sensitivity, the fused one slightly more than
P+E (overall F1 0.934 vs 0.937, within one standard deviation). The networks were trained only on jammers placed
at random, so part of what they learned is contrast between jammed and unjammed PRBs; adding
scheduling-aware slots to training is the obvious fix, at the cost of no longer testing an
unseen attacker.

## 5. Size and latency

`results/size_sweep/size_sweep.pdf`, `results/latency.csv`. Fused detector (P+R+E) with the width
of the first U-Net level set to 4, 6, 8 and 12 channels (3 seeds each, same data and training).
CPU time for feature extraction (0.26 ms) and detector inference after equalization, on one
thread of a MacBook Pro M1 Pro. "Frozen graph" is the same network as a TorchScript graph with
batch norm folded into the convolutions.

| Width | Parameters | F1, all | F1, pilot | Pd pilot, −15 dB | Pd data, −15 dB | Total time | Total, frozen graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 7 657 | 0.943 ± 0.002 | 0.799 ± 0.012 | 0.55 ± 0.02 | 0.72 ± 0.01 | 0.53 ms | **0.43 ms** |
| 6 | 16 933 | 0.948 ± 0.001 | 0.821 ± 0.005 | 0.59 ± 0.01 | 0.75 ± 0.01 | 0.59 ms | **0.48 ms** |
| 8 | 29 841 | 0.949 ± 0.001 | 0.812 ± 0.011 | 0.60 ± 0.02 | 0.77 ± 0.01 | 0.64 ms | 0.53 ms |
| 12 | 66 553 | 0.951 ± 0.002 | 0.834 ± 0.010 | 0.62 ± 0.02 | 0.78 ± 0.01 | 0.74 ms | 0.64 ms |

**Reading.** A 30 kHz slot lasts 0.5 ms. In this benchmark the width-12 detector takes 0.74 ms
with eager inference. With 7.7k parameters the overall F1 drops by 0.008, and the feature
extraction and frozen forward-pass medians sum to 0.43 ms.
The loss is concentrated in weak pilot jamming (Pd at −15 dB 0.55 instead of 0.62). The U-Net's
time grows much more slowly than its size (9× parameters, 1.8× time): for inputs this small,
per-layer overhead dominates, not arithmetic. With PyTorch's default eight threads the U-Net is
slower (about 1.3 ms) than on one.

## 6. What the map is good for: map-aware recovery

`results/recovery/recovery_{A,C}.pdf` (main comparison) and `recovery_{A,C}_ablation.pdf` (hard
against soft use). UE on 12 PRBs, jammer on a 4-PRB block inside the allocation, SNR 15 dB, Colab
T4. Every batch is decoded by all receivers (same slots); each receiver runs until 100 block
errors or 3 200 slots, so a point without errors means BLER below 0.12% (95% upper bound).
Receivers (`jamloc/recovery.py`):

- **No protection:** Sionna's PUSCH receiver.
- **DMRS-only (no map):** what a receiver can do with its DMRS alone. PRBs whose residual exceeds
  the analytic 1% threshold get down-weighted pilots, and the interference measured on the DMRS is
  assumed to last the whole slot.
- **Power-test map, hard (Li et al. 2012):** per-cell power test (relative to the slot's median),
  pilots of flagged PRBs dropped, flagged data erased.
- **Map, hard:** the same with the U-Net's map (this was the receiver of the first round).
- **Map, soft (proposed):** the map says *where*; the receiver estimates *how strong* from the
  flagged cells. Pilots of flagged PRBs are kept but their noise variance is multiplied by the
  PRB's residual; the jamming power on flagged data cells is estimated from their excess received
  energy and added to their noise variance before demapping.
- **True map:** the jammer's real cells, used hard or soft (its strength is still estimated).

BLER, TDL-C (TDL-A is similar, slightly easier):

| Case | JSR | No protection | DMRS-only | Power map, hard | P+R+E map, hard | **P+R+E map, soft** | True map, soft |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No jammer | – | 0 | 0 | 0.003 | 0.003 | **0** | 0 |
| Pilot | −15 dB | 0 | 0 | 0.003 | 0.053 | **0** | 0 |
| Pilot | 0 dB | 0.011 | 0 | 0.022 | 0.009 | **0** | 0 |
| Pilot | +9 dB | 0.325 | 0.008 | 0.048 | 0.003 | **0.001** | 0 |
| Data | −15 dB | 0.0003 | 0 | 0.005 | **0.115** | **0** | 0 |
| Data | 0 dB | 0.158 | 0.159 | 0.156 | 0.005 | **0** | 0 |
| Data | +9 dB | 0.844 | 0.844 | 0.177 | 0.001 | **0.0003** | 0.0003 |
| Partial | −15 dB | 0 | 0 | 0.003 | 0.125 | **0** | 0 |
| Partial | +9 dB | 0.898 | 0.011 | 0.268 | 0.003 | **0.001** | 0.001 |
| All pilots jammed | +3 dB | 0.409 | – | – | 0.409 (true map) | – | **0.040** |
| All pilots jammed | +9 dB | 0.945 | – | – | 0.945 (true map) | – | **0.395** |

**Reading.**
- **Soft use costs nothing.** With the fused map used softly the BLER stays at or below 0.001 for
  every attack and JSR, and is 0 without a jammer, exactly like the true map. Used hard, the same
  map costs up to 0.12 BLER at weak jamming, where the unprotected link has no errors: false alarms
  drop clean pilots and erase clean data. Even the *true* map costs about 0.002 when used hard,
  because erasing four of twelve PRBs throws away symbols that weak jamming had barely damaged.
  This is the weak-jamming penalty of the first round, now removed without any gating rule.
- **Without a map, the DMRS is not enough.** The DMRS-only receiver handles pilot and partial-band
  jamming (0.008 and 0.011 at +9 dB), because both hit the DMRS, but it is blind to data jamming
  (0.84, like no protection). The map's time resolution (which symbols are jammed) is what makes
  data jamming recoverable.
- **The classic power-test receiver** only helps against strong jamming (data +9 dB: 0.18) and
  adds up to about 0.02 BLER at weak jamming, where the unprotected link is (nearly) error-free:
  a per-cell power test cannot tell jamming from fading.
- **With soft use, the map without the residual is as good** (P+E soft equals P+R+E soft within
  the confidence bands): weak jamming that the P+E network misses also barely hurts decoding, and
  the receiver weights pilots by their residual anyway. The residual's advantage shows in
  localization (section 3), and inside the receiver as the pilot weight.
- **When every pilot is jammed** nothing can be interpolated. The hard receiver falls back to the
  standard estimate (identical BLER), but weighting all pilots by their residual makes the LMMSE
  filter average more strongly across the band: 0.040 instead of 0.41 at +3 dB, 0.39 instead of
  0.95 at +9 dB.

## 7. Figures for the presentation and the paper

Open `results/figures/index.html`. Explanatory figures (each with its caption): attack patterns,
system diagram, how the residual works, constellations (wrong-symbol effect), feature
distributions, detector outputs on typical slots, training curves. Result figures: BER/BLER,
feature maps, Pd vs JSR (test, scheduling-aware, two SNR ranges), recovery (main and hard vs
soft), the residual against its theory, unseen conditions, mixed training, size and latency.

## 8. The ideal null distribution of the DMRS residual

Sections 8–14 report the earlier run through v5. The rebuilt detection, margin and calibration
results are in §§17–18.

`jamloc/theory.py` and `results/theory/residual_theory.pdf`. Derivation (checked independently,
and by Monte Carlo with 250 000 draws per number): with no jammer, white Gaussian noise of known
power and the UE's channel in the retained delay subspace, the residual of PRB b is
`R_b = sum_i lambda_i E_i` with independent `Exp(1)`
variables `E_i` and weights given by the eigenvalues of the 6 x 6 block of the projector
`I - U U^H` on the PRB's pilots, normalized by their sum. The channel, the SNR and the pilot
sequence drop out under these assumptions.

- With fixed numerology, pilot geometry and projector, the 1% threshold depends on allocation
  size and PRB position. It is 2.31 in the middle of a 12-PRB allocation and 2.56 at its edges,
  where fewer degrees of freedom remain.
- under Gaussian pilot jamming the same formula with the eigenvalues of
  `P (I + JNR G) P` gives the detection probability, and for unjammed PRBs the false alarms
  caused by **leakage** of the projection.

Check against the link simulation (no training data involved):

| Quantity | Theory | Simulation |
| --- | --- | --- |
| False-alarm rate at the analytic 1% threshold, benign test slots, TDL-A | 1% | 0.91% (34 368 cells) |
| Same, TDL-C | 1% | 0.92% (28 640 cells) |
| Same, 5 to 25 dB SNR in 2.5 dB bins | flat | flat within the 95% intervals |
| Pd inside a 4-PRB pilot-jammed block, JNR 0 / 6 / 12 dB (block Rayleigh) | 0.29 / 0.69 / 0.93 | 0.30 / 0.71-0.74 / 0.93-0.94 |
| False alarm of the unjammed PRB next to the block, JNR 12 / 18 dB | 0.16 / 0.50 | 0.17 / 0.54-0.55 |
| False-alarm rate, TDL-C with 1000 ns delay spread (delays up to 8.7 us > tau_max) | assumption broken | **61%** |

**Reading.** Under the ideal null, a fixed residual threshold gives a constant false-alarm rate
across channels and SNRs. The approximation holds when the UE's channel energy outside the retained
subspace is small compared with the noise. The subspace keeps all but 1e-6 of the energy of the
*average* channel with delays up to 3 us; a
single path close to 3 us at a band-edge PRB leaks about 1.4e-4 of its energy, which matters
above about 25 dB SNR (checked independently). Our SNRs end at 25 dB and TDL-C's last tap is at
2.6 us, so no effect is visible (the 22.5-25 dB bin is at 0.6%); at higher SNR, `tau_max` or the
kept energy fraction must grow. Its two limits are also
predicted by the same formula: leakage into neighbouring PRBs under strong pilot jamming (a
reason to fuse it with the other features in a network that sees the whole map), and channels
whose delays exceed `tau_max`.

## 9. Conditions the detectors never saw

`results/generalization/generalization.pdf` (thresholds unchanged, from the matched `val` split;
200 slots per grid point, 3 seeds). Each test grid changes one thing:

| Condition | P+R+E: F1 / FA | P+E: F1 / FA | Power only: F1 / FA | EVM only: F1 | Residual threshold: FA |
| --- | --- | --- | --- | --- | --- |
| Matched test | 0.951 / 1.2% | 0.946 / 1.3% | 0.913 / 1.2% | 0.772 | 1.3% |
| TDL-C, 1000 ns delay spread | 0.818 / **6.1%** | 0.701 / **15.7%** | **0.923 / 1.3%** | 0.657 | **28%** |
| Speeds up to 30 m/s | 0.865 / **5.1%** | 0.835 / **6.7%** | **0.912 / 1.5%** | 0.707 | 1.2% |
| QPSK jammer | **0.951 / 1.1%** | 0.942 / 1.3% | 0.905 / 1.2% | **0.593** | 1.1% |

**Reading.**
- **The jammer's waveform does not matter** for the fused detector (0.951 with a QPSK jammer).
  Only the EVM-only network drops (0.77 to 0.59): QPSK interference on QPSK data moves symbols
  between constellation points, which a decision-directed EVM partly hides.
- **Long delays** break the assumption of the residual (delays up to 8.7 us > `tau_max` = 3 us):
  the UE's own channel leaks into it, exactly as section 8 predicts (61% false alarms at the
  analytic threshold). The EVM rises too, because the channel varies faster across subcarriers.
  The fused network suffers less than the one without the residual (6% vs 16% false alarms).
- **High speed** leaves the residual untouched (one DMRS symbol, 1.2% false alarms), but the
  channel changes within the slot, so the EVM of the late symbols grows without any jamming and
  the networks that use EVM flag them.
- **Power alone is robust** to both, but weaker in matched conditions.
- **The block test (section 11) breaks harder than the networks:** F1 0.591 / 21% false alarms
  at 1000 ns and 0.646 / 15% at 30 m/s (0.878 with the QPSK jammer). Its nulls are exact only
  under their assumptions, and both conditions violate them (residual: delays beyond `tau_max`;
  EVM null: measured on slow, short-delay channels). The U-Net trained with noise jitter
  (section 13) behaves like the nominal one here (0.833 / 4.8%, 0.843 / 6.1%, 0.948).
- In practice `tau_max` should cover the deployment's delay spread (it costs subspace dimensions
  and therefore sensitivity), and a gNB serving fast users would use an additional DMRS symbol,
  as NR allows. Section 10 tests whether training on such channels is enough.

## 10. Training on mixed channels

`results/robust_training/robust_training.pdf`. Every U-Net trained again (3 seeds, same size and
epochs) on `train_mix`: batches cycling through the nominal channels, long delay spreads
(TDL-A 300 ns, TDL-C 1000 ns) and speeds up to 30 m/s; thresholds from `val_mix`. The test grids
are those of section 9, so their channel *types* were part of the mix (with different seeds):
this tests whether diversity is enough, not whether the detector handles a truly unknown channel.

| Condition | P+R+E nominal | P+R+E mixed | P+E mixed | Power only mixed |
| --- | --- | --- | --- | --- |
| Matched test | 0.951 / 1.2% | **0.955 / 0.7%** | 0.948 / 0.7% | 0.914 / 1.0% |
| TDL-C, 1000 ns | 0.818 / 6.1% | **0.921 / 1.7%** | 0.911 / 2.1% | 0.928 / 1.0% |
| Speeds up to 30 m/s | 0.865 / 5.1% | **0.946 / 1.0%** | 0.939 / 1.0% | 0.917 / 1.1% |

(F1 / false-alarm rate on the cells each detector scores, mean over 3 seeds.)

**Reading.** Diverse training removes most of the loss: on fast users the fused detector is back
to 0.946 with the design false-alarm rate, and with long delays to 0.921 (1.7% false alarms),
level with the power-only network, which needed no help. It costs nothing in matched conditions
(0.955; the false-alarm rate there is lower because the thresholds come from the more varied
`val_mix`). The residual still loses information when delays exceed `tau_max`; the network
learns to rely on the other features there.

## 11. Is the deep network needed? A classical block test and a per-cell network

Third round (`docs/plan-v3.md`). The threshold baselines of section 3 come from the literature
and score only the cells where their feature is defined, so none of them can see jamming outside
the allocation. Two detectors show where the U-Net's gain really comes from (both are part of
every `evaluate` run):

- **Block test** (`baselines.block_test_scores`, no training). The strongest classical detector we
  could build from the same features. Jammers hit whole PRBs on the DMRS symbol, on the 13 other
  symbols, or on both, so every cell gets the `-log10` p-value of the statistic of its group:
  outside the allocation, the power of the DMRS cell (exact Gamma(12) null) and the power averaged
  over the PRB's 13 data symbols (exact Gamma(156) null); on scheduled DMRS cells, the residual's
  exact p-value (section 8); on scheduled data cells, the PRB-averaged EVM against benign
  validation PRBs of the same 4 dB SNR bin and edge position (its tail extrapolated as exponential
  beyond the 95th percentile, so scores do not saturate). A rule separates pilot-only from
  partial-band jamming: the DMRS cell and the data cells of a PRB share the channel, so their
  power ratio is about 1 without jamming and under partial-band jamming and above 1 when only the
  pilots are jammed; when the residual and this ratio are both significant at 1% (fixed in
  advance, not tuned), the PRB's data cells get p = 1 (they stay among the tested cells but are not
  flagged). In benign test slots every cell type
  is flagged 0.7-1.1% of the time at p < 0.01.
- **Per-cell MLP** (`model.PixelMLP`): the fused U-Net's six input channels, but only 1x1
  convolutions (4 801 parameters), so every cell is classified from its own features and masks
  without its neighbours. Same training (15 epochs, 3 seeds).

| Detector | Pilot | Data | Partial | All slots | Scheduling-aware jammer |
| --- | --- | --- | --- | --- | --- |
| **U-Net, P+R+E** | **0.834** | **0.957** | **0.961** | **0.951** | **0.934** |
| Block test (no training) | 0.512 | 0.912 | 0.880 | 0.872 | 0.848 |
| Per-cell MLP, P+R+E | 0.383 | 0.711 | 0.729 | 0.694 | 0.621 |
| EVM threshold (best literature baseline) | 0 | 0.324 | 0.319 | 0.303 | 0.399 |

(Cell-level F1 at 1% per-cell false alarm; learned detectors: mean over 3 seeds, std <= 0.011.
Pd per JSR in section 3. By SNR, block test 0.834 / 0.908 and U-Net 0.916 / 0.987 for 5-15 /
15-25 dB.)

**Reading.**
- Covering the whole carrier and averaging over the PRB closes most of the gap to the literature
  baselines (0.30 -> 0.87). The fair comparison for the U-Net is this block test: 0.951 vs 0.872.
- What remains is mostly pilot jamming (0.834 vs 0.512). From one PRB, the block test cannot tell
  whether a raised EVM comes from jammed data or from a corrupted channel estimate unless the
  pilot-to-data power contrast is significant, which fails at weak pilot jamming. It also flags
  the data cells under strong pilot jamming whenever the contrast test misses, which costs it
  false alarms (2.7% in pilot-attack slots). The U-Net resolves this from context.
- The block test's Pd for partial-band jamming levels off at 0.87 on scheduled cells (section 3
  table, +9 dB): under strong jamming the DMRS cell's power averages only 12 jammed resource
  elements, so the pilot-to-data power ratio fluctuates far more than in the benign slots its
  null comes from, and the pilot-only rule sometimes unflags jammed data cells. Requiring the
  ratio to be significant also against the equal-jamming null F(24, 312) removes this (Pd 0.99)
  but lets more pilot-induced EVM through: pilot-attack F1 0.451 instead of 0.512, overall 0.861
  instead of 0.872. The table keeps the rule with the higher F1. (The independent code review
  counted the cost at +9 dB: the rule unflags 702 of 5 024 jammed scheduled PRBs.)
- Without context the learned detector falls below the block test (0.694): comparing a cell with
  the other cells of its PRB is most of what both exploit. The U-Net's value is learning that
  structure and the cross-feature rules together, from data, and adapting them when the
  conditions change (section 10) where hand-made nulls would have to be rebuilt.

## 12. Detection before damage

`results/margin/margin.pdf` and `margin_compact.pdf` (`experiments/detection_margin`). Can an
attacker damage the link and stay unseen? The `margin` split puts the jammer on exactly the
scheduled PRBs, as in the BLER experiment of section 1, at a fixed SNR of 15 dB and JSR from -30
to 0 dB (13 200 jammed and 2 000 benign slots, simulated in batches of 8 so that every grid point
mixes 25 allocations). A slot is flagged when its largest cell score exceeds the level that 1% of
benign slots exceed; whole batches of benign slots alternate between setting that level and
checking it, so the check uses separate batches (measured slot false-alarm
rate 0.7-1.9%). The BLER of the unprotected link is what a
detector based on CRC failures or link KPIs can see.

| Channel | Attack | JSR at 10% BLER | Slot Pd 90%: P+R+E | P+E | Block test | Margin P+R+E |
| --- | --- | --- | --- | --- | --- | --- |
| TDL-A | Pilot | +0.6 dB | **-13.4 dB** | -10.0 dB | -12.3 dB | **14.0 dB** |
| | Data | -5.5 dB | -15.2 dB | -15.4 dB | -13.8 dB | 9.7 dB |
| | Partial | -6.7 dB | -17.5 dB | -16.0 dB | -15.1 dB | 10.8 dB |
| TDL-C | Pilot | +0.2 dB | **-15.0 dB** | -9.8 dB | -12.6 dB | **15.2 dB** |
| | Data | -6.5 dB | -14.7 dB | -15.3 dB | -14.8 dB | 8.2 dB |
| | Partial | -7.3 dB | -16.2 dB | -16.1 dB | -15.7 dB | 8.9 dB |

**Reading.**
- At 15 dB SNR and the 1% slot false-alarm target, the 90% detection crossing for each tested
  whole-allocation attack is at least 8.2 dB of JSR below its 10% unprotected BLER crossing.
- The residual buys the most where the attacker is most efficient: for pilot jamming the margin
  is 14.0 / 15.2 dB with it and 10.6 / 10.0 dB without (TDL-A / TDL-C).
- Per cell (90% of the jammed cells, not of the slots) the margin is much smaller: 7.0-8.9 dB for
  pilot jamming, but between -1.7 and +2.4 dB for data and partial-band jamming (on TDL-C the 90%
  cell level comes after the 10% BLER point). The slot is flagged long before every one of its
  cells is.
- The block test is up to 2.4 dB behind the fused network at the slot level (level with it for
  TDL-C data jamming).
- A first version of this split was simulated in batches of 64, so each grid point mixed only 3-4
  allocations and the calibration and check halves shared them (found by the independent code
  review). Regenerated as above, the margins moved by up to 1.8 dB; the minimum went from 7.7 to
  8.2 dB.

## 13. False alarms across SNR and under noise-estimation error

`results/calibration/calibration.pdf` (`experiments/calibration`). Here thresholds are set the
CFAR way, on the cells of benign `val` slots (1% flagged), and the false-alarm rate is measured on
the cells of the 4 000 benign `test` slots.

**SNR.** Per-cell false alarm in benign slots, by SNR (mean over 3 seeds):

| Detector | Thresholds | 6.25 dB | 11.25 dB | 16.25 dB | 21.25 dB | 23.75 dB | Overall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net, P+R+E | one | 3.9% | 1.1% | 0.17% | 0.17% | 0.10% | 1.05% | 0.948 |
| U-Net, P+R+E | per 4 dB SNR bin | 0.72% | 1.24% | 0.64% | 0.99% | 0.59% | 0.94% | 0.942 |
| U-Net, power only | one | 3.7% | 1.7% | 0.49% | 0.24% | 0.25% | 1.31% | 0.906 |
| U-Net, power only | per SNR bin | 1.3% | 1.1% | 1.2% | 1.8% | 1.7% | 1.45% | 0.895 |
| Block test | one | 0.77% | 0.91% | 0.85% | 0.98% | 0.81% | 0.86% | 0.869 |
| Residual, exact p-value | one | 0.92% | 1.13% | 1.09% | 1.02% | 0.66% | 0.95% | - |

(The F1 here uses the benign-calibrated thresholds, so it differs slightly from section 3.)

**Noise power.** Every feature is divided by the noise variance. An estimate that is off by
`delta` dB shifts P, R and E by `-delta` dB inside their masks; this is applied to the stored test
features with thresholds from the correct noise power (it leaves out the second-order effect of
the wrong noise variance on the LMMSE weights). For the residual the exact answer is
`P(R > t 10^(delta/10))`, averaged over the DMRS cells of the benign test slots, with `t` the
threshold at the same design rate as the measured curve (the cutoff calibrated on benign
validation cells, p = 1.04%).

| Detector | -2 dB | -1 dB | -0.5 dB | 0 | +0.5 dB | +1 dB | +2 dB | F1 at -1 / 0 / +1 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net, P+R+E | 54% | 37% | 15% | 1.05% | 0.11% | 0.05% | 0.04% | 0.699 / 0.948 / 0.937 |
| U-Net, P+E | 60% | 45% | 19% | 1.06% | 0.11% | 0.06% | 0.04% | 0.637 / 0.940 / 0.931 |
| U-Net, power only | 54% | 39% | 15% | 1.31% | 0.37% | 0.13% | 0.02% | 0.586 / 0.906 / 0.897 |
| Block test | 56% | 28% | 8.2% | 0.86% | 0.20% | 0.09% | 0.02% | 0.522 / 0.869 / 0.858 |
| Residual, exact p-value | 14.4% | 4.75% | 2.31% | 0.95% | 0.32% | 0.11% | 0.00% | - |
| Residual, theory | 14.4% | 4.83% | 2.38% | 1.04% | 0.40% | 0.13% | 0.01% | - |
| **U-Net, P+R+E, noise jitter** | 20% | **2.0%** | **1.2%** | 1.20% | 1.09% | 0.73% | 0.21% | **0.936 / 0.946 / 0.947** |

The last row is the fused U-Net trained with the noise estimate off by an error drawn uniformly
from +-1 dB per slot (`train --noise-jitter-db 1`, `model.noise_shift`, 3 seeds; validation and
model selection without jitter). With the thresholds from benign validation slots, its false
alarms are 1.2% at the correct noise power.

**Reading.**
- With one threshold, the U-Nets' false-alarm rate follows the SNR mix of the validation set: 4%
  at low SNR, 0.1-0.25% at high SNR. The gNB knows its SNR, and a threshold per 4 dB bin holds the
  fused detector at 0.6-1.4% everywhere for 0.006 F1. The block test and the residual, whose
  nulls are exact (or per SNR bin), need nothing: they stay at about 1% with one threshold.
- The noise power is the detectors' weak point. An underestimate of 0.5 dB raises the U-Nets'
  false alarms from 1% to 15-19%, the block test's (whose power nulls average 156 resource
  elements, so they are narrow) to 8%. The residual, with six degrees of freedom per PRB, goes
  only to 2.3%, as its theory predicts (2.4%).
- A small overestimate is harmless: at +0.5 dB the fused detector has 0.11% false alarms and F1
  0.951; at +2 dB it starts to cost (F1 0.902, Pd 0.83 instead of 0.96). So a gNB that is unsure
  of its noise floor should bias it slightly upward. In practice the thermal noise
  floor is stable and can be averaged over many slots of empty resource elements, so the error
  can be made far smaller than 0.5 dB.
- **Training with a jittered noise estimate removes the sensitivity** at almost no cost: within
  +-1 dB the false-alarm rate stays at 0.7-2.0% (without jitter: 0.05-37%), and F1 at -1 dB is
  0.936 instead of 0.699. In matched conditions F1 is 0.948 +- 0.001 against 0.951 (pilot attacks
  0.822 vs 0.834, Pd at -15 dB 0.59 vs 0.62); against the scheduling-aware jammer both reach
  0.934. Beyond the training range (-2 dB) the false alarms rise to 20%, still far below 54%.
  Within +-1 dB the jittered network is less sensitive than the residual itself, probably
  because it learns to judge each cell against the rest of the slot rather than against the
  noise floor (not tested separately).
  Its false-alarm rate still depends on the SNR with one threshold (4.4% at 6 dB, 0.3% above
  15 dB), so it needs thresholds per SNR bin like the others.

## 14. Averaging the residual over several slots

`results/theory/multislot.pdf` (`experiments/multislot`). A jammer that stays on the same PRBs
can be found at lower power by averaging each PRB's residual over K slots. Under the ideal null
in §8, with independent noise and the same allocation in every slot, the average is a weighted
sum of unit exponentials, with every weight divided by K and repeated K times
(`theory.slot_average`). The Pd calculation also assumes stable jammed PRBs, channel gains
and JNR over the K slots (coherence time about 12 ms at 3 m/s and 3.5 GHz, i.e. 24 slots).

| K (slots) | 1% threshold, mid / edge PRB | False alarm, link level | JNR for Pd 0.5 | JNR for Pd 0.9 |
| --- | --- | --- | --- | --- |
| 1 | 2.31 / 2.56 | 0.92% (727 / 78 862) | 3.1 dB | 10.7 dB |
| 2 | 1.87 / 2.03 | 0.94% (368 / 39 321) | 1.0 dB | 8.3 dB |
| 4 | 1.59 / 1.69 | 0.97% (190 / 19 539) | -0.9 dB | 6.3 dB |
| 8 | 1.40 / 1.47 | 1.15% (111 / 9 628) | -2.7 dB | 4.4 dB |

Pd: PRB 5 of a 12-PRB allocation with PRBs 4-7 jammed, block-Rayleigh jammer fading. The
false-alarm column applies the K-slot thresholds to the stored residuals of the benign `test`
and `val` slots, grouped K at a time by allocation (95% interval at K = 8: 0.96-1.39%). A
pilot-level Monte Carlo with the jammer's gain held over the K slots matches the Pd formula
by at most 0.02 at every point (largest gap 0.020 at K = 2, 0 dB, about two standard errors of
the difference: both average over random jammer gains, 2 000 draws for the formula and 4 000 for
the simulation); e.g. K = 8, 0 dB: 0.686 simulated,
0.690 theory.

**Reading.** In this run, eight slots (4 ms) lower the JNR needed for 90% detection by about
6.2 dB. The analytic thresholds target the same false-alarm rate for the single-slot test
and the average under the stated null assumptions.

## 15. Smarter jammers that use the DMRS configuration

Fourth round (`docs/plan-v4.md`). The threat model lets the jammer know the DMRS positions, but
every attack so far jammed whole PRBs on fixed symbols, a structure the block test and the U-Net
both exploit. Two unseen jammers test that (`SimConfig`, test grids `combjam` and `hopjam`, same
size as the other generalization grids; thresholds from the matched `val` split):

- **Pilot comb** (`jammer_pilot_comb`): on the DMRS symbol only the pilot subcarriers (the even
  comb of type-1 DMRS) are jammed. The receiver's channel estimate uses only those resource
  elements, so the damage per jammed RE is the same as before for half the energy on the DMRS
  symbol. RE-selective reference-signal jamming is known from LTE (Rao et al. 2017) and the NR
  PBCH (Kazemian et al. 2024).
- **Hopping** (`jammer_hopping`): the jammed PRB block (same size) moves to a new random position
  on every OFDM symbol, like a fast frequency-hopping jammer (Kleber & Laneman 2017; Schösser et
  al. 2026 include such jammers in an OFDMA anomaly benchmark).

Cell-level F1 (false-alarm rate on scored cells), nominal training:

| Detector | Matched | Pilot comb: all / pilot attacks | Hopping: all | Hopping: false alarms |
| --- | --- | --- | --- | --- |
| U-Net, P+R+E | 0.951 | 0.946 / 0.769 | 0.587 | 12.3% |
| U-Net, P+E | 0.946 | 0.928 / 0.551 | 0.564 | 11.2% |
| U-Net, power only | 0.913 | 0.904 / 0.738 | 0.545 | 20.6% |
| Block test | 0.872 | 0.876 / 0.409 | 0.470 | 24.6% |
| Per-cell MLP, P+R+E | 0.694 | 0.700 / 0.388 | **0.684** | 1.8% |

Pd at -15 dB on jammed scheduled cells, pilot comb: P+R+E 0.59 (0.62 against the full pilot
attack), P+E 0.35 (0.43), power only 0.18 (0.25), block test 0.31 (0.34).

**Retraining on mixed attacks** (`experiments/attack_training`, `results/attack_training/`): the
fused U-Net trained again on `train_atk` (batches cycling through the nominal, hopping and
pilot-comb jammers, own seeds; thresholds from `val_atk`), 3 seeds:

| Test grid | Nominal training: F1 / false alarms | Attack-mixed training: F1 / false alarms |
| --- | --- | --- |
| Matched | 0.951 / 1.2% (pilot attacks 0.834) | **0.953 +- 0.001 / 0.8%** (pilot attacks 0.884) |
| Pilot comb | 0.946 / 1.3% (pilot attacks 0.769) | **0.950 / 0.9%** (pilot attacks 0.849) |
| Hopping | 0.587 +- 0.009 / 12.3% | **0.914 +- 0.003 / 1.9%** |

**Reading.**
- **The residual matters more against the comb jammer.** It measures exactly the energy that
  corrupts the channel estimate, so the fused detector keeps its sensitivity to weak pilot
  jamming (Pd 0.59 at -15 dB), while every detector that relies on the power of the whole DMRS
  cell loses some: without the residual the pilot-attack F1 falls to 0.551, and the gap between
  the networks with and without it nearly triples (0.077 -> 0.218).
- **The hopping jammer breaks the learned structure.** Trained on whole-PRB attacks, the U-Nets
  paint whole PRB rows when part of a row is jammed (false alarms 11-21%); the block test, built
  on PRB averages, fails the same way. Only the per-cell MLP, which assumes no structure, is
  robust, and it becomes the best detector of the table.
- **Mixed training improves hopping detection.** With the three jammers in the training mix,
  the fused U-Net reaches 0.914 on the hopping jammer with 1.9% false alarms. Matched overall
  F1 stays close to the nominal model's. Multi-tone F1 falls (§16). The attack
  types of the test grids were part of the mix (with independent draws), so, as for the mixed
  channels of section 10, this shows that diversity suffices, not that every unseen attack
  structure is handled. Its higher pilot-attack F1 in matched conditions (0.884) comes partly
  from its own operating point: thresholds from `val_atk` give 0.8% false alarms instead of
  1.2%.

## 16. Pulsed, multi-tone and spoofing jammers

`results/generalization/`, `results/attack_training/`, `results/eval/{pulsejam,tonejam,spoofjam}_*`,
`results/spoofing/` (`experiments/generalization --splits spoofjam pulsejam tonejam`,
`experiments/attack_training`, `experiments/spoofing`).
Three more jammers that no detector saw in training, each changing one attribute
(`jamloc.jammer`, plan v5):

- `pulsejam` - time support: each OFDM symbol of a data or partial-band attack is jammed with
  probability 0.5 (`jammer_duty`); pilot attacks would be unchanged and are left out.
- `tonejam` - waveform and frequency support: one continuous tone per jammed PRB
  (`jammer_waveform="tone"`, `jammer_tones=1`), a twelfth of the nominal energy per cell.
- `spoofjam` - knowledge: the jammer replays the UE's DMRS on the scheduled PRBs, scaled, with one
  random phase per slot (`jammer_waveform="dmrs"`, pilot attack, scheduling-aware). Its
  contribution to the LS estimates is `a g_k`, a physically possible channel, so the delay-subspace
  projection removes it: the residual is blind by construction (`tests/test_jammer.py` checks this,
  and also that spoofing only part of the allocation leaves visible steps at the block edges).

F1 / false-alarm rate (scored cells), thresholds from `val` (mixed attacks: `val_atk`), 3 seeds:

| Detector | Pulsed | Multi-tone | Spoofing |
| --- | --- | --- | --- |
| U-Net P+R+E | 0.629 / 10.9% | **0.760** / 1.5% | 0.029 / 27.3% |
| U-Net P+E | 0.625 / 10.1% | 0.752 / 1.6% | 0.249 / 10.7% |
| U-Net P | 0.612 / 11.3% | 0.710 / 1.5% | 0.399 / 3.3% |
| U-Net E | 0.545 / 9.5% | 0.505 / 0.5% | **0.574** / 0.5% |
| Per-cell MLP | **0.711** / 0.5% | 0.321 / 0.2% | 0.000 / 7.6% |
| Block test | 0.595 / 8.6% | 0.627 / 0.8% | 0.001 / 26.1% |
| U-Net P+R+E, attack-mixed training | 0.846 / 3.0% | 0.746 / 1.0% | 0.014 / 28.0% |

**Reading.**
- **Pulsed** jamming breaks the row structure like hopping does: the U-Nets paint whole rows (10-11%
  false alarms), the per-cell MLP is best. The attack-mixed network, which never saw a pulsed
  jammer, reaches 0.85: what it learned from the hopping jammer carries over.
- **Multi-tone**: 0.76 for the fused network; pilot attacks drop to F1 0.54 because half the tones
  fall on the empty DMRS comb (no effect on the channel estimate, 1/12 of the power).
- **Spoofing** is the residual's blind spot (when the jammer's channel stays within the delay
  subspace): at the analytic 1% threshold, 0.96% of the spoofed DMRS cells and 1.00% of the
  benign ones exceed it, so the residual cannot tell them apart; and the detectors that trust the residual fail worst:
  with clean-looking pilots, the raised EVM of the data (wrong channel estimate) is read as data
  jamming, so the data symbols are flagged (27% false alarms) instead of the DMRS cells. The
  EVM-only network, which learned the pilot-jamming signature from EVM alone, localizes it best.
- **The attack is noticed, though.** Slot alarm (largest logit above the level 1% of the benign
  slots of `spoofjam` exceed; calibrated on every other benign slot, checked on the rest, slot
  false alarms 1.0-1.6%): 43-58% of spoofed slots flagged at -15 dB, 81-86% at -12 dB, 91-95% at
  -9 dB, 98-99% at -3 to 0 dB; seed 2 drops again above +3 dB (0.63 at +9 dB). Flags fall on 27-94%
  of the data cells but only 5-56% of the attacked DMRS cells.
- **Spoofing is also more harmful.** Unprotected BLER, TDL-C, 15 dB SNR, jammer on all 12 scheduled
  PRBs, 512 slots per point:

  | JSR | -15 dB | -9 dB | -3 dB | +3 dB |
  | --- | --- | --- | --- | --- |
  | Gaussian pilot jamming (whole DMRS symbol) | 0 | 0.002 | 0.014 | 0.379 |
  | Spoofing (pilot comb only, half the energy) | 0 | 0.008 | 0.170 | 0.701 |

## 17. Review v6: rebuilt detection results

`docs/review-v6.md`, `docs/plan-v6.md`. The v6 detection run used
`experiments.paper_run --seeds 0 1 2` on a Colab T4. Reported timings were 14 min for data,
31 min for 54 trainings, 86 min for evaluation and 44 min for BLER sweeps and margins
(about 3 compute units). Recovery, the two-DMRS comparison and CPU timing were run separately.
The rebuilt datasets add power on the empty DMRS comb as a seventh feature and store the
gNB's SNR estimate (`snr_db_est`) for the block test and SNR-binned thresholds. With
16-slot batches, a 500-slot test point uses 32 allocation draws, with repeats possible.
Where results are reported again below, they supersede the corresponding values in §§3–16.

### 17.1 Detection (test split, 3 seeds, thresholds from `val`)

F1 / achieved false-alarm rate per attack (all cells), and Pd of jammed scheduled cells at
-15 dB JSR:

| Detector | pilot | data | partial | all | Pd pilot -15 | Pd data -15 |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net P+R+E | 0.824 / 0.60% | 0.958 / 1.65% | 0.962 / 1.57% | **0.952** / 1.15% | 0.57 | 0.78 |
| U-Net P+R+E+C (empty comb) | 0.836 / 0.60% | 0.957 / 1.78% | 0.963 / 1.64% | 0.952 / 1.20% | **0.67** | 0.78 |
| U-Net P+E | 0.750 / 0.93% | 0.952 / 1.92% | 0.960 / 1.40% | 0.945 / 1.26% | 0.39 | 0.76 |
| U-Net P | 0.808 / 0.32% | 0.923 / 1.95% | 0.919 / 1.94% | 0.914 / 1.19% | 0.24 | 0.56 |
| U-Net E | 0.436 / 1.41% | 0.785 / 1.32% | 0.776 / 0.83% | 0.765 / 1.04% | 0.11 | 0.48 |
| U-Net P+R+E, noise jitter | 0.826 | 0.956 | 0.960 | 0.949 | 0.57 | 0.77 |
| Per-cell MLP (4 801 param.) | 0.394 | 0.722 | 0.740 | 0.706 / 1.32% | 0.07 | 0.01 |
| Per-cell MLP (66 277 param.) | 0.393 | 0.723 | 0.741 | 0.707 / 1.32% | 0.06 | 0.01 |
| Block test | 0.515 / 2.65% | 0.911 / 0.33% | 0.871 / 0.72% | 0.868 / 1.22% | 0.29 | 0.32 |
| Block test + empty comb | 0.531 | 0.911 | 0.873 | 0.869 | 0.36 | 0.32 |
| Empty-comb test | 0.721 | 0 | 0.079 | 0.079 | 0.37 | 0 |
| Residual, exact p-value | 0.575 | 0 | 0.059 | 0.059 | 0.08 | 0 |
| EVM threshold | 0 | 0.332 | 0.331 | 0.314 | 0 | 0 |
| Power threshold | 0.194 | 0.135 | 0.125 | 0.132 | 0.01 | 0.02 |

Uncertainty (`experiments.intervals`, 1000 bootstrap draws of batches within each channel x
attack x JSR point, the same draws for every detector and seed). Intervals are conditional
on the trained checkpoints and validation thresholds. The standard deviation across seeds
describes training variability on the evaluated data. Fused F1 0.952 [0.950, 0.953], pilot
0.824 [0.812, 0.835]. Paired differences, seed mean:

| Difference | F1 all | F1 pilot | Pd pilot -15 dB |
| --- | --- | --- | --- |
| P+R+E minus P+E (what the residual adds) | +0.006 [0.006, 0.007] | +0.074 [0.065, 0.085] | +0.179 [0.153, 0.203] |
| P+R+E minus block test | +0.084 [0.082, 0.086] | +0.309 [0.297, 0.320] | +0.283 [0.262, 0.303] |
| P+R+E minus per-cell MLP of the same size | +0.245 [0.240, 0.249] | +0.431 [0.417, 0.446] | +0.515 [0.490, 0.537] |
| P+R+E+C minus P+R+E (what the empty comb adds) | +0.000 [0.000, 0.001] | +0.012 [0.006, 0.019] | +0.101 [0.083, 0.119] |
| block test + comb minus block test | +0.002 | +0.016 | +0.073 |

PRB-level F1 (thresholds on the maximum cell score target 1% of scored clean validation PRBs):
U-Net P+R+E 0.943 (pilot 0.909), P+E 0.934 (0.878), block test 0.833 (0.789), EVM
threshold 0.615 (0.463), residual p-value 0.425 (0.567), empty comb 0.551 (0.721).

**Reading.**
- The residual's main gain is +0.18 Pd for weak pilot jamming at -15 dB. The per-seed estimates
  and the paired interval are positive. Overall F1 moves by 0.006.
- Spatial context, not capacity: a per-cell MLP with as many parameters as the U-Net is exactly as
  good as the small one (0.707 vs 0.706).
- The empty comb, the baseline the review asked for, beats the residual on the nominal pilot
  attack (pilot F1 0.72 vs 0.58 alone). Added to the U-Net it helps weak pilot jamming (+0.10 Pd)
  and the scheduling-aware jammer (F1 0.947 vs 0.937), but a jammer that leaves the empty comb
  clean defeats it: on `combjam` the comb network's pilot F1 is 0.23 (fused 0.78) with 4.6% false
  alarms, and with long delays 0.70 / 15% (fused 0.80 / 6.7%). It stays an ablation.
- At PRB level the EVM threshold doubles its F1 (0.31 -> 0.62): part of its cell-level failure is
  the label definition. It still trails every detector that sees the DMRS.

### 17.2 Unseen conditions (F1 / false alarms, all cells)

| Condition | P+R+E | P+R+E+C | P+E | P | Block test | Per-cell MLP |
| --- | --- | --- | --- | --- | --- | --- |
| matched test | 0.952 / 1.2% | 0.952 / 1.2% | 0.945 / 1.3% | 0.914 / 1.2% | 0.868 / 1.2% | 0.706 / 1.3% |
| scheduling-aware | 0.937 / 0.5% | 0.947 / 0.5% | 0.938 / 1.1% | 0.891 / 0.2% | 0.837 / 3.6% | 0.629 / 6.0% |
| QPSK jammer | 0.952 / 1.1% | 0.954 / 1.1% | 0.947 / 1.2% | 0.909 / 1.2% | 0.879 / 1.2% | 0.718 / 1.2% |
| channels not normalized per slot | 0.936 / 1.2% | 0.937 / 1.3% | 0.929 / 1.3% | 0.899 / 1.2% | 0.851 / 1.1% | 0.680 / 1.0% |
| TDL-C 1000 ns | 0.795 / 6.7% | 0.702 / 15.2% | 0.695 / 15.7% | 0.915 / 1.2% | 0.595 / 20.1% | 0.660 / 3.8% |
| up to 30 m/s | 0.857 / 5.5% | 0.865 / 5.2% | 0.837 / 6.4% | 0.909 / 1.4% | 0.639 / 15.0% | 0.666 / 3.8% |
| pilot comb | 0.950 / 1.2% | 0.864 / 4.6% | 0.932 / 1.6% | 0.908 / 1.2% | 0.875 / 1.6% | 0.703 / 1.2% |
| hopping | 0.595 / 12.9% | 0.583 / 14.3% | 0.565 / 11.7% | 0.558 / 19.7% | 0.474 / 24.2% | 0.688 / 1.9% |
| pulsed | 0.636 / 10.8% | 0.630 / 10.7% | 0.628 / 9.9% | 0.617 / 10.9% | 0.599 / 8.3% | 0.714 / 0.4% |
| multi-tone | 0.763 / 1.5% | 0.769 / 1.9% | 0.750 / 1.6% | 0.715 / 1.4% | 0.626 / 0.9% | 0.346 / 0.2% |
| spoofing | 0.028 / 28.5% | 0.004 / 29.6% | 0.254 / 10.5% | 0.390 / 3.5% | 0.001 / 26.3% | 0.000 / 7.7% |
| neighbor-cell UE (labels: its cells) | 0.953 / 1.1% | 0.947 / 1.3% | 0.957 / 1.0% | 0.905 / 1.4% | 0.876 / 0.7% | 0.756 / 0.1% |

Scheduling-aware jammer: pilot F1 0.870 (P+E 0.784), Pd pilot at -15 dB 0.58 (P+E 0.34).

Mixed channel training (`train_mix`, thresholds from `val_mix`): matched 0.956 / 0.7%, 1000 ns
0.918 / 1.6%, 30 m/s 0.947 / 1.0% (P+E: 0.948, 0.903, 0.941; P: 0.915, 0.919, 0.913).

Mixed attack training (F1 / false alarms):

| Condition | nominal | mixed attacks (`train_atk`) | mixed attacks + spoofing (`train_atk2`) |
| --- | --- | --- | --- |
| matched | 0.952 / 1.2% | 0.953 / 0.9% | 0.946 / 1.0% |
| pilot comb | 0.950 / 1.2% | 0.953 / 0.9% | 0.947 / 1.0% |
| hopping | 0.595 / 12.9% | 0.918 / 1.7% | 0.908 / 2.2% |
| pulsed (in neither mix) | 0.636 / 10.8% | 0.857 / 2.6% | 0.839 / 3.2% |
| multi-tone (in neither mix) | 0.763 / 1.5% | 0.750 / 1.1% | 0.738 / 1.2% |
| spoofing | 0.028 / 28.5% | 0.015 / 28.7% | **0.871 / 0.7%** |

Mixed training improves hopping F1 and keeps matched overall F1 close to the nominal model's.
Multi-tone F1 falls with either mix.

### 17.3 Pilot spoofing

Slot alarm (largest logit above the level that 1% of the benign batches of `spoofjam` exceed,
calibrated on every other benign batch; slot false alarms about 1%) and placement, range over 3
seeds:

| JSR | training | slot Pd | DMRS cells flagged | data cells flagged | F1 (slots at this JSR) | false alarms |
| --- | --- | --- | --- | --- | --- | --- |
| -12 dB | nominal | 0.77-0.84 | 0.09-0.13 | 0.55-0.60 | 0.02-0.03 | 35-38% |
| -12 dB | with spoofing | 0.83-0.89 | 0.87-0.90 | 0.04-0.05 | 0.67-0.74 | 2-3% |
| -9 dB | nominal | 0.94 | 0.06-0.11 | 0.65-0.74 | 0.01-0.02 | 46-52% |
| -9 dB | with spoofing | 0.96-0.97 | 0.96-0.98 | 0.01-0.02 | 0.84-0.90 | 1-2% |

At the analytic 1% residual threshold, 1.03% of the 53 688 spoofed DMRS cells and 0.94% of the
29 680 benign scheduled DMRS cells of `spoofjam` exceed it (`baselines.residual_cfar_scores`,
score > 2).

**Partial spoofing** (`spoofpart`: the same replay on a random block of 2-12 PRBs, so usually on
part of the allocation; 70% of the attacked cells lie inside it). The replay ends inside the
allocation and the residual sees the step: 22.1% of the 18 035 spoofed scheduled DMRS cells exceed
the analytic threshold, but so do 15.8% of the 35 381 clean scheduled DMRS cells in those slots
(0.98% in benign slots). F1 / false alarms: P+R+E 0.133 / 10.9%, P+E 0.318 / 4.3%, P 0.449 /
1.7%, E 0.511 / 0.9%, P+R+E+C 0.064 / 11.9%, block test 0.111 / 7.3%; trained on mixed attacks
0.111 / 10.0%, with whole-allocation spoofing in the mix 0.292 / 4.7%. Training on whole-allocation
replay improves its localization on that grid (§17.2), while F1 on partial replay remains low.

Unprotected BLER, TDL-C, 15 dB, 512 slots: spoofing 0.158 at -3 dB and 0.693 at +3 dB; Gaussian
jamming of the DMRS symbol 0.014 and 0.391. The analytic residual stayed close to its null
on the tested whole-allocation replay grid. Partial replay raised flags on both attacked PRBs
and clean neighbors. The model trained on whole-allocation replay could localize it using power
and EVM along with the residual.

### 17.4 Noise power

Full receiver chain with noise power off by delta. Thresholds are from the correct noise power.
False alarms are measured on scored cells in benign slots, and F1 uses all slots of the
corresponding evaluation split.

| delta | -2 dB | -1 dB | -0.5 dB | 0 | +0.5 dB | +1 dB | +2 dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net P+R+E | 55.8% | 39.9% | 16.6% | 0.95% | 0.11% | 0.04% | 0.03% |
| U-Net P+R+E, noise jitter | 19.0% | 2.1% | 1.4% | 1.2% | 1.2% | 0.8% | 0.2% |
| Block test | 49.0% | 27.7% | 8.3% | 0.93% | 0.28% | 0.12% | 0.08% |
| Residual p-value (theory) | 14.5% (14.4%) | 4.9% (4.8%) | 2.35% (2.38%) | 1.01% (1.04%) | 0.45% (0.40%) | 0.14% (0.13%) | 0.01% (0.01%) |

The feature-shift approximation used before (15% and 37% at -0.5 and -1 dB) was close.

Per-slot estimate from the empty DMRS comb (`noise_estimate="comb"`): error median -0.04 dB, 5-95%
[-1.2, +1.05] dB, std 0.70 dB without jamming; jamming on the DMRS raises it (median +1.6 dB,
95th percentile +14 dB on the test grid), and a scheduling-aware jammer on the whole DMRS symbol
of the allocation breaks it (median +10.9 dB). Detection with the estimate:

| Deployment | test_nest F1 / benign FA | sched_nest F1 |
| --- | --- | --- |
| U-Net trained and calibrated with the true N0 | 0.839 / 10.2% | 0.657 |
| same, recalibrated on `val_nest` | 0.847 / 2.35% | 0.498 |
| U-Net trained with +-1 dB jitter | 0.935 / 1.6% | 0.819 |
| U-Net trained and calibrated on the estimate (`train_nest`) | **0.948 / 0.67%** | **0.866** |
| Block test (calibrated on `val_nest`) | 0.737 / 1.7% | 0.460 |

SNR: with one threshold the fused U-Net's benign false-alarm rate falls from 4.4% at 6 dB to
0.12% at 24 dB; with one threshold per 4 dB bin of the *estimated* SNR it stays at 0.6-1.2%
(F1 0.943 vs 0.949). The residual p-value stays at 0.9-1.2% with one threshold.

### 17.5 The residual against its theory (rebuilt data)

Analytic 1% threshold on benign test slots: 1.05% (TDL-A), 0.95% (TDL-C), 0.65-1.24% per 2.5 dB
SNR bin; 62.9% on TDL-C 1000 ns. Pilots of 4 of 12 PRBs jammed (SNR 10 dB), simulation
TDL-A / TDL-C: inside the block 0.30 / 0.32 at 0 dB JNR (theory 0.29), 0.76 / 0.72 at 6 dB (0.69),
0.94 / 0.93 at 12 dB (0.93); clean neighbor 0.17 / 0.16 at 12 dB (0.16) and 0.53 / 0.54 at 18 dB
(0.50). K-slot thresholds: false alarms 0.91-1.00% for K = 1, 2, 4, 8.

### 17.6 Detection before damage, and the link's own margin

BLER without a jammer (MCS 5, random allocations): 10% at about 3.4 dB SNR (TDL-A) and 4.0 dB
(TDL-C); at 15 dB the link has about 11 dB to spare. Margin = JSR of 10% BLER minus JSR of 90%
slot Pd (jammer on the scheduled PRBs), fused U-Net, TDL-C, smallest over the three attacks:

| SNR (spare margin, TDL-C) | 15 dB (~11 dB) | 11 dB (~7 dB) | 8 dB (~4 dB) | 5 dB (~1 dB) |
| --- | --- | --- | --- | --- |
| 1% slot false-alarm target | +8.6 dB | +7.7 dB | +3.5 dB | -6.1 dB |
| 0.1% slot false-alarm target | +5.5 dB | +5.9 dB | +1.6 dB | ≤ -7.94 dB (pilot Pd below 90% over the sweep) |

TDL-A: +10.1, +8.7, +5.0, -1.8 dB (1%). The detector needs about the same JNR in every case
(data jamming, TDL-C: 90% slot Pd at -15.1, -12.9, -10.2, -6.5 dB JSR, i.e. JNR -0.1 to -2.2
dB); the damage threshold moves with the spare margin. Near the link-adaptation point a jammer
hurts the link before the slot alarm fires. Per-seed margins: `results/margin/*_per_seed.csv`.

### 17.7 Neighbor-cell interference

A synchronized UE of a neighboring cell (QPSK data and its own DMRS on a block of PRBs, its own
channel) is flagged like jamming: the fused U-Net marks 94.6% of its scheduled cells and 97.7% of
the others (Gaussian partial-band jammer at the same power: 95.8% / 97.1%), from 77% at -15 dB ISR
to 99% above -3 dB. The map localizes interference; telling legitimate interference from an
attack needs information the features do not contain (e.g. the neighbor's DMRS sequence, known to
the network, or coordination between gNBs).

### 17.8 Size and latency

F1 / pilot Pd at -15 dB: width 4 (7 657 parameters) 0.944 / 0.52, width 6 (16 933) 0.949 / 0.54,
width 8 (29 841) 0.950 / 0.56, width 12 (66 553) 0.952 / 0.57.

CPU latency was measured on an idle Apple M1 Pro with one thread and a 24-PRB allocation
(`size_sweep --latency`, 19 Nov). Median feature extraction took 0.31 ms for all seven maps.
Median U-Net eager / frozen times were 0.27 / 0.17 ms (width 4), 0.32 / 0.22 ms (6), 0.38 / 0.26 ms
(8), 0.48 / 0.37 ms (12). The feature and frozen forward-pass medians sum to
0.48, 0.53, 0.58 and 0.68 ms. Only width 4 is below 0.5 ms for these two stages. The timed
stages begin after equalization and end before decoding. Feature extraction was 0.26 ms
before the empty-comb map was added.

### 17.9 What the receiver needs from the map (review item 1)

`experiments.recovery`, TDL-C 15 dB unless stated, UE on PRBs 6-17, jammer on 4 of them, three
detector seeds each on its own 1 600 slots (4 800 per point; a receiver stops after 100 block
errors per run), run on the M1 Pro (5 processes x 2 threads, about 3.5 h in all). New receivers:
`lmmse` (the LMMSE interpolator, no map), `slot_ie` (one interference estimate per slot),
`block_soft` (block-test map), `true_prb_soft` (true map coarsened to one decision for the DMRS
cell and one for the data cells of each PRB: resolution without detection error),
`p_r_e_atk_soft` (map of the U-Net trained on `train_atk`).

The table gives the original BLER percentages at +9 dB JSR. Zero means no errors in 4 800 slots.
See §18.3 for the replay update to per-PRB true-map BLER under hopping partial-band jamming.

| Receiver | pilot | data | partial | pulsed data | pulsed partial | hopping data | hopping partial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No protection | 33 | 81 | 88 | 19 | 42 | 29 | 58 |
| LMMSE, no map | 18 | 81 | 86 | 18 | 36 | 28 | 50 |
| Slot-wide estimate | 12 | 84 | 90 | 61 | 72 | 75 | 83 |
| DMRS only | 0.60 | 81 | 1.3 | 18 | 9.9 | 28 | 57 |
| Power test, hard | 5.1 | 19 | 29 | 1.2 | 4.4 | 1.1 | 12 |
| Block test, soft | 0.06 | 0.04 | 7.7 | 0 | 3.0 | 54 | 64 |
| U-Net, hard | 0.17 | 0.08 | 0.19 | 0.42 | 0.65 | 36 | 54 |
| U-Net, soft | 0.06 | 0.04 | 0.02 | 0 | 0.50 | 21 | 31 |
| U-Net (mixed attacks), soft | 0.08 | 0.04 | 0.06 | 0 | 0.06 | 0 | 0.25 |
| True map per PRB, soft | 0.04 | 0.04 | 0.02 | 0 | 0.08 | 52 | 58 |
| True map, soft | 0.04 | 0.04 | 0.02 | 0 | 0 | 0 | 0.21 |

**Reading.**
- Without a map, the better interpolator and a slot-wide interference estimate help little; the
  slot-wide estimate hurts against pulsed and hopping jammers. DMRS-only protects against pilot
  and partial-band jamming, not data jamming.
- Resolution: the per-PRB true map is as good as the exact one for jammers that fill PRB rows or
  pulse within them; against the hopping jammer it fails (52%), because the jamming power estimated
  over all data symbols of a touched PRB is far too low on the jammed ones.
- Detection: the U-Net trained on block jammers fails against hopping (21%); trained on mixed
  attacks it matches the true map (0 / 0.25%). The block test's map falls behind the per-PRB true
  map under strong partial-band jamming (7.7% vs 0.02%) because its pilot-only rule fires on about
  a quarter of the jammed data cells (76% flagged at +9 dB, 86% at +3 dB; the rule is a trade-off:
  testing the power ratio also against the partial-band null raises partial F1 by 0.024 but lowers
  pilot F1 by 0.067 and overall F1 from 0.868 to 0.864).
- Hard use at weak jamming: U-Net map without jammer 0.23%; at -15 dB 3.9% (pilot), 9.7% (data),
  12% (partial). The soft U-Net receiver had zero block errors without a jammer and at -15 dB
  for each of these attacks.
- Soft U-Net map, nominal attacks, every JSR: at most 0.08% (4/4 800, partial +3 dB; the true map
  gives the same).
- All pilots jammed (whole allocation): true map soft 4.1% at +3 dB, unprotected 38%.
- 8 dB SNR (about 4 dB spare): unprotected 0.6% without jammer, LMMSE-based receivers 0.2%; data
  jamming +3 dB: U-Net soft 3.6%, true map 3.6%, unprotected 50%; partial +3 dB: 6.0%, 5.8%, 61%.
- TDL-A is similar (soft U-Net map at most 0.13%; block map 4.8% under partial +9 dB).

## 18. Open items of review v6

### 18.1 Detectors at matched false-alarm rates

Overall cell F1, averaged over seeds, on the nominal test and scheduling-aware splits. The
matched thresholds are an oracle for detector comparison. They use the evaluated split to
flag 1% of each detector's scored clean cells, rounded down to a whole cell. Split-wide
matching pools clean cells across the split. By-attack matching fits each attack type
separately. If a group has no scored clean cells, it uses the split-wide threshold. The
bootstrap holds the fitted decisions fixed.

| Detector | Test val | Test split-wide | Test by attack | Sched val | Sched split-wide | Sched by attack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net P+R+E | 0.952 | 0.954 | 0.951 | 0.937 | 0.949 | 0.948 |
| U-Net P+R+E+C | 0.952 | 0.955 | 0.952 | 0.947 | 0.955 | 0.953 |
| U-Net P+E | 0.945 | 0.948 | 0.946 | 0.938 | 0.936 | 0.940 |
| Block test | 0.868 | 0.865 | 0.883 | 0.837 | 0.731 | 0.870 |
| Wide per-cell MLP | 0.707 | 0.675 | 0.855 | 0.629 | 0.351 | 0.890 |
| EVM threshold | 0.314 | 0.306 | 0.623 | 0.412 | 0.185 | 0.185 |

Paired differences with by-attack matching (95% batch-bootstrap intervals). Pilot Pd uses
jammed scheduled cells at -15 dB JSR.

| Difference | Metric | Test | Scheduling-aware |
| --- | --- | --- | --- |
| P+R+E minus wide MLP | Overall F1 | +0.096 [0.093, 0.098] | +0.058 [0.055, 0.061] |
| P+R+E minus block test | Overall F1 | +0.068 [0.066, 0.070] | +0.078 [0.075, 0.081] |
| P+R+E minus P+E | Pilot Pd | +0.234 [0.207, 0.259] | +0.376 [0.361, 0.394] |
| P+R+E+C minus P+R+E | Pilot Pd | +0.096 [0.080, 0.113] | +0.109 [0.098, 0.119] |

**Reading.**

- The wider MLP's validation threshold flags 0.01% of clean cells in the test split's
  data-jammed slots. Its F1 gap to the fused U-Net widens with split-wide matching and
  narrows when each attack gets its own threshold. The fused U-Net leads every non-U-Net
  detector in overall F1 at both matched settings on both splits. The paired intervals
  for its leads over the block test, both MLPs and the EVM threshold exclude zero.
- The residual increases weak-pilot Pd at both matched settings. The empty comb adds
  another weak-pilot gain and lifts pooled F1 more on the scheduling-aware split than on
  the nominal test. With by-attack matching on the nominal test, the standalone comb
  also beats the standalone residual on pilot F1.

### 18.2 Uncertainty of the detection margins

Smallest margin over attacks, in dB, for the fused U-Net (95% intervals). Each of 1,000
stratified batch-bootstrap draws refits the slot threshold and draws new block-error counts
from binomial distributions fitted to the observed counts. The binomial approximation can
understate uncertainty from shared allocations and the simulation's error-count stopping rule.

| SNR | TDL-A, 1% slot FA | TDL-A, 0.1% slot FA | TDL-C, 1% slot FA | TDL-C, 0.1% slot FA |
| --- | --- | --- | --- | --- |
| 15 dB | 10.1 [9.6, 10.7] | 8.0 [4.8, 8.7] | 8.6 [7.3, 9.1] | 5.5 [1.3, 6.4] |
| 11 dB | 8.7 [7.5, 9.8] | 7.4 [6.1, 8.0] | 7.7 [6.6, 8.3] | 5.9 [5.1, 6.6] |
| 8 dB | 5.0 [3.9, 5.8] | 3.4 [2.8, 4.3] | 3.5 [2.3, 4.5] | 1.6 [0.5, 2.5] |
| 5 dB | -1.8 [-3.2, -1.2] | -3.7 [−∞, -2.7] | -6.1 [-11.6, -4.3] | ≤ -7.9 [−∞, -5.7] |

Selected differences on TDL-C in dB (95% intervals). The detector and false-alarm
comparisons share draws within an SNR. The between-SNR comparisons use independent draws.

| Difference | Conditions | Estimate [95% interval] |
| --- | --- | --- |
| P+R+E minus P+E | Pilot margin at 15 dB, 1% slot FA | +4.0 [1.8, 5.4] |
| P+R+E minus block test | Smallest margin at 15 dB, 1% slot FA | +2.1 [1.1, 2.7] |
| 1% minus 0.1% slot FA | P+R+E smallest margin at 15 dB | +3.1 [1.8, 7.2] |
| 15 minus 11 dB SNR | P+R+E smallest margin, 1% slot FA | +0.9 [-0.3, 2.1] |
| 15 minus 8 dB SNR | P+R+E smallest margin, 1% slot FA | +5.1 [3.5, 6.5] |
| 8 minus 5 dB SNR | P+R+E smallest margin, 1% slot FA | +9.5 [7.2, 15.3] |

**Reading.**

- The smallest margin is positive at the top three SNRs and negative at 5 dB for both
  channels and targets. On TDL-C at 5 dB and 0.1% slot false alarms, pilot Pd falls short
  of the target across the JSR grid, so the smallest point estimate is an upper bound.
- At 1% slot false alarms, the smallest margin favors the fused U-Net over the block test
  at every SNR on both channels. At 0.1%, the paired intervals establish the lead at 11
  and 8 dB. The fused U-Net's gain from the higher false-alarm target is resolved at each SNR.
- The drop from 8 to 5 dB is resolved on both channels at both targets. From 15 to 8 dB,
  it is resolved on both channels at 1% and on TDL-A at 0.1%. From 15 to 11 dB, it is
  resolved only on TDL-A at 1%.

### 18.3 Paired BLER intervals

The TDL-C replay put all receivers on the same slots. The paired 95% intervals (Newcombe's
method 10) pool outcomes across three detector seeds, with up to 4 800 slots per point and
fewer when either receiver stopped after 100 errors. Differences are in percentage points,
first receiver minus second. SNR is 15 dB unless stated.

| Receiver comparison | Conditions | Difference [95% interval] |
| --- | --- | --- |
| Soft U-Net map vs true map | Nominal pilot, +9 dB | +0.02 [-0.06, +0.12] |
| Soft U-Net map vs true map | Nominal data and partial, +9 dB, and partial, +3 dB | 0.00 [-0.08, +0.08] each |
| True map per PRB vs exact true map | Nominal and pulsed | Within 0.08, all intervals include zero |
| True map per PRB vs exact true map | Hopping data, +9 dB | +52 [48, 56] |
| True map per PRB vs exact true map | Hopping partial, +9 dB | +57 [53, 61] |
| Block-test map vs true map per PRB | Nominal partial, +9 dB | +7.7 [6.9, 8.5] |
| Soft U-Net map, nominal training, vs true map | Pulsed partial, +9 dB | +0.50 [0.32, 0.74] |
| Soft U-Net map, mixed training, vs true map | Pulsed partial, +9 dB | +0.06 [-0.03, +0.18] |
| Soft U-Net map, mixed training, vs true map | Hopping data, +9 dB | 0.00 [-0.08, +0.08] |
| Soft U-Net map, mixed training, vs true map | Hopping partial, +9 dB | +0.04 [-0.05, +0.15] |
| Hard vs soft U-Net map | No jammer | +0.23 [0.10, 0.41] |
| Hard vs soft U-Net map | Nominal pilot, -15 dB | +3.9 [3.4, 4.5] |
| Hard vs soft U-Net map | Nominal data, -15 dB | +9.7 [8.7, 10.8] |
| Hard vs soft U-Net map | Nominal partial, -15 dB | +12.2 [11.0, 13.5] |
| LMMSE, no map, vs Sionna | Nominal pilot, +9 dB | -15.7 [-18.2, -13.2] |
| Slot-wide estimate vs LMMSE, no map | Pulsed and hopping, +9 dB | +33 to +46 (pulsed data +42 [38, 46]) |
| Soft U-Net map vs true map | 8 dB SNR, nominal data, +3 dB | +0.06 [-0.04, +0.17] |
| Soft U-Net map vs true map | 8 dB SNR, nominal partial, +3 dB | +0.15 [+0.01, +0.29] |
| Sionna vs LMMSE, no map | 8 dB SNR, no jammer | +0.42 [0.25, 0.64] |

**Reading.**

- The replay reproduced the table counts except for the coarsened true map under hopping
  partial-band jamming at +9 dB. Its BLER on the common slots is 56.94% (328/576).
- Under pulsed partial-band jamming at +9 dB, the nominally trained U-Net gives 0.50%
  BLER and the true map gives zero. With mixed training, the intervals for this difference
  and for both hopping differences include zero.
- Without a map, the LMMSE interpolator lowers pilot BLER at +9 dB on the 1 024 slots both
  receivers decoded. The slot-wide estimate raises BLER under pulsed and hopping jamming.
- At 8 dB SNR, the partial-band difference is small and its interval excludes zero.
  Without a jammer, the LMMSE-based receivers give 0.21% BLER, against 0.62% for Sionna.

### 18.4 Recovery with the estimated noise power

The receiver and features use the noise power estimated in each slot from the empty DMRS
comb. The fused U-Net is trained on the estimate (`train_nest`), and the block and power
tests are calibrated on it. The TDL-C runs use 15 dB SNR, the same UE and jammer setup, and
JSRs from -15 to +9 dB in 6 dB steps. Three seeds use up to 6 400 slots each (19 200 per
point). The true-noise runs use different slots. BLER is in percent, with 95% Wilson
intervals in brackets where shown.

| Receiver | Conditions | Estimated noise power | True noise power |
| --- | --- | --- | --- |
| U-Net map, soft | No jammer | 0 | 0 |
| U-Net map, soft | Nominal pilot, +9 dB | 0.057 | — |
| U-Net map, soft | Nominal data, +9 dB | 0.089 | — |
| U-Net map, soft | Nominal partial, +9 dB | 0.23 [0.18, 0.31] | 0.02 |
| True map | Nominal partial, +9 dB | 0.20 [0.15, 0.28] | 0.02 |
| Block-test map, soft | Nominal partial, +9 dB | 9.9 | 7.7 |
| U-Net map, hard | Nominal data, -15 dB | 6.0 | 9.7 |
| True map / no protection | Whole-allocation pilot, +3 dB | 54 / 53 | 4.1 (true map) |
| True map / no protection | Whole-allocation pilot, +9 dB | 98 / 98 | — |

**Reading.**

- The soft U-Net map gives at most 0.23% BLER under nominal jamming. The true map has
  a similar increase under partial-band jamming at +9 dB, where jamming on the DMRS symbol
  of 4 of the 12 PRBs raises the noise estimate.
- Hard use of the U-Net map costs less at -15 dB under data jamming because the detector
  was trained on the estimate.
- Whole-allocation pilot jamming contaminates every element of the empty comb. The estimate
  fails, and the true map gives no recovery gain over the unprotected receiver.

## Caveats before these go into the paper

- Each Pd point averages over SNR 5–25 dB; the SNR-range files give the split by SNR.
- The 1% false-alarm threshold is set over all non-jammed cells a detector scores; inside the
  allocation the fused U-Net's false-alarm rate is somewhat higher (about 1.6% on local checks).
- 15 training epochs per run; the validation loss is lowest well before that (training curves).
- Only 190 different allocations exist, so test allocations repeat training ones; channel
  realizations and jammer draws do not.
- All slots of a simulated batch share one allocation. Since section 17 the grid splits use
  batches of 16 (the `margin` grids 8). A 500-slot test point has 31 full batches and one
  partial batch, with one allocation draw per batch. Allocations can repeat. The earlier
  test grid used batches of 64 or 128.
- The reference datasets were generated on a GPU. Record the device and software versions
  when reproducing a run. CPU and GPU runs can draw different random samples.
- Normalized EVM is biased upward at high SNR without jamming (issue #6); the detectors are
  calibrated on data that contains this effect.
- On Colab's CUDA 13 build, PyTorch's batched complex double-precision solve (MAGMA) crashes; the
  LMMSE interpolation therefore solves its small systems on the CPU.

## 19. Showcase site and new paper figures

Design `docs/superpowers/specs/2026-11-20-showcase-site-design.md`, plan
`docs/superpowers/plans/2026-11-20-showcase-site.md`.

**Paper figures.** All three come from `experiments.figures`:

- **Timing (new, system model):** detection and mitigation both work on the buffered slot.
- **Fig. 3 (new):** the per-pilot residual after projection, and `R_b` with its position-dependent
  threshold. The realization is the CPU one, so the leakage shows on clean PRB 9.
- **Fig. 6 (new):** one pilot-jammed slot as three aligned maps, plus Pd at -15 dB per detector
  from `results/intervals/test_pd_ci.csv`.

**Example slots** (`python -m jamloc.site export`, CPU, detector seed 0):

- 12 cases, schema version 2. Each case holds the feature maps, detector decisions, the
  pilot residuals, and per receiver (on the same received grid) the equalized symbols, |LLR| and
  CRC.
- `batch_crc` records receiver failures over the case's batch. The paper residual example
  `fig3` uses simulation seed 0 and one slot. The other cases use simulation seed 7 and
  64 slots per batch.
- Examples are chosen by a stated rule: the median U-Net hit rate by default, outcome-conditioned
  for `walkthrough_flip` (2/64) and `pilot_whole_comb` (true map fails in 42/64).
- The checked export manifest records commit `ebff348` and a clean working tree.
- **Jammer-power levels (23 Nov).** The seven Explore cases (pilot, data, partial-band, hopping,
  pulsed, spoofing, whole-allocation pilot jamming with estimated noise) also hold their slot at
  every jammer power: off, then -15 to +9 dB in 3 dB steps. Each level replays the case's batch
  seed with only the JSR changed and keeps the slot chosen at the case's own power, so the
  channel, noise, data and jammer waveform are the same (checked in `tests/test_site.py`: cells
  the jammer leaves alone receive identical power at every level). With the jammer off the truth
  grid is empty. Each level also records the CRC failures of the full 64-slot batch, which the
  page plots against the power. The nominal level of every case is unchanged from the earlier
  export. The files grow to 1.3–1.8 MB each (0.38–0.55 MB gzipped), loaded only by Explore.
- **Bits before and after decoding (23 Nov).** Every case also records the transmitted QPSK
  symbols and, per receiver, the symbols its bits give before decoding (LLR signs) and after
  decoding (its decoded block encoded again). Explore's "What decoding does with the bits" grids
  and the walkthrough's last step draw the wrong bits per cell before decoding and the CRC outcome
  after it. In the walkthrough slot (partial-band, +3 dB) all wrong bits before decoding
  lie in the jammed cells: 362 for the standard receiver and 352 with the U-Net map. After decoding
  the map-aware block equals the one sent (0 wrong bits), while the standard receiver's decoded
  block differs from it in 564 coded bits. Over the 64-slot batch (`partial` at +3 dB), 26% of the
  jammed cells' bits are still wrong before decoding with either the U-Net map or the true map
  (29% for the standard receiver), but they reach the decoder with a mean |LLR| of 0.2 on the wrong
  bits per jammed bit, against 17.7 for the standard receiver
  (`python -m jamloc.experiments.decoding_bits`, `results/site/decoding_bits.csv`).

**Site** (`python -m jamloc.site build | serve | deploy`, static build on the `gh-pages` branch):

- The site is configured for the project's
  [GitHub Pages address](https://erhankarakoca.github.io/nr-jamming-localization/).
- The old Animations page redirects to Walkthrough, which has the residual sweep. The old
  Constellations page redirects to Explore, which shows constellations and the recorded
  hopping and pulsed slots. Results includes Pd, margin, recovery, matched false alarms
  and the two-DMRS comparison.
- The checked build has 405 Pd rows, 16 margin rows, 60 matched metric rows, 72 matched
  difference rows, 77 recovery rows and 385 paired recovery rows. The two-DMRS tables have
  8 detection rows and 28 recovery rows. All 13 CSV downloads match their source files byte for byte.

## 20. Two DMRS symbols per slot

NR allows up to three additional single-symbol DMRS per slot. Everything above uses one
front-loaded DMRS symbol (symbol 2). `SimConfig.dmrs_additional_position = 1` puts a second one on
symbol 11, as Sionna does. The jammer's pilot attack then hits both DMRS symbols, which costs it
twice the energy at the same JSR per resource element. The residual is computed on each DMRS
symbol. The map-aware receiver estimates one slot-constant channel from the pilots of both
symbols, so a jammed DMRS symbol is down-weighted against a clean one.

Run (Colab T4; `python -m jamloc.experiments.dmrs_count --seeds 0 1 2 --batch-size
64 --batches 25`, then `--no-recovery` locally for the detection tables):

- new splits `train_dmrs2` / `val_dmrs2` / `test_dmrs2` (seeds 30-32 M);
- U-Nets P+R+E and P+E retrained on them (three seeds, 15 epochs);
- the one-DMRS rows reuse the nominal models and splits, and reproduce section 17.

**Detection** (test grid, thresholds for 1% false alarm on the matching validation split, mean of
three seeds; Pd over the jammed cells inside the allocation at -15 dB):

| DMRS | Detector | Pilot F1 | Overall F1 | Test FA | Pilot Pd | Data Pd |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | U-Net P+R+E | 0.824 | 0.952 | 1.2% | 0.574 | 0.783 |
| 1 | U-Net P+E | 0.750 | 0.945 | 1.3% | 0.394 | 0.756 |
| 1 | Block test | 0.515 | 0.868 | 1.2% | 0.290 | 0.320 |
| 2 | U-Net P+R+E | 0.881 | 0.954 | 1.2% | 0.701 | 0.801 |
| 2 | U-Net P+E | 0.845 | 0.950 | 1.2% | 0.487 | 0.795 |
| 2 | Block test | 0.671 | 0.877 | 1.2% | 0.311 | 0.398 |

Per seed, the weak-pilot Pd of P+R+E is 0.55-0.60 with one DMRS and 0.69-0.71 with two.

**Recovery** (TDL-C, 15 dB SNR, UE on 12 PRBs, jammer on 4 of them, 4 800 slots per point from
three seeds of 1 600; BLER in percent at +9 dB):

| DMRS, jammed DMRS | Attack | Standard | LMMSE, no map | U-Net soft | True map soft |
| --- | --- | --- | --- | --- | --- |
| 1 | pilot | 31.1 | 16.0 | 0.04 | 0.02 |
| 2, both | pilot | 24.2 | 4.73 | 0.06 | 0.02 |
| 2, first only | pilot | 7.33 | 1.17 | 0 | 0 |
| 1 | data | 81.9 | 81.9 | 0.04 | 0.04 |
| 2 | data | 79.3 | 78.6 | 0.02 | 0.02 |
| 1 | partial | 89.3 | 86.3 | 0.12 | 0.12 |
| 2 | partial | 85.9 | 80.7 | 0.02 | 0.02 |

These slots are new draws, so the one-DMRS row differs slightly from the table of section 17 (31%
against 33% unprotected under pilot jamming). At -15, -3 and +3 dB the soft U-Net map stays at or
below 0.04% in every case (`results/dmrs_count/recovery.csv`).

**Reading.**

- A second DMRS symbol makes pilot jamming easier to detect (pilot F1 0.824 to 0.881, weak-pilot
  Pd 0.57 to 0.70). Part of this is the jammer's doubled energy at the same JSR per resource
  element. The block test gains most on pilot attacks (0.515 to 0.671), because it now has two
  residuals per PRB.
- It also makes pilot jamming less harmful, even though the jammer spends twice the energy: the
  unprotected BLER at +9 dB falls from 31% to 24%, and the LMMSE filter alone, which averages the
  two DMRS symbols, brings it to 4.7%. A jammer that hits only the front-loaded DMRS leaves a
  clean second symbol: 7.3% unprotected, 1.2% with the LMMSE filter, no errors with the map.
- Data and partial-band jamming are about as harmful as with one DMRS, and the soft map protects
  against them in both configurations.

## 21. Data-aided passes when every pilot is jammed

With the jammer on every DMRS cell of the allocation, no map can give the receiver a clean pilot,
and even the true map left 37% BLER at +9 dB in section 17. The map-aware receiver
(`jamloc.recovery`, `iterations`) now adds data-aided passes: after a failed decoding, the LDPC
decoder's a posteriori LLRs give the mean and variance of every QPSK symbol, each data cell the
map calls clean becomes a pilot whose variance includes the remaining symbol uncertainty and the
jamming the map assigns to it, all observations of a subcarrier are combined with inverse-variance
weights, the same LMMSE filter smooths them, and the slot is decoded again. A block keeps the first
decoding whose CRC passes. Three passes (`DATA_AIDED`). Decoder feedback itself is not new: Li et
al. (MILCOM 2012, section IV) remodulate confident decisions on interference-free tones, but start
from the pilots that survive detection, and give no procedure when every pilot is corrupted.

Run (TDL-C, known noise power, three detector seeds, up to 1,600 slots each, own results folder):

```bash
JAMLOC_RESULTS_DIR=results/receiver_study  # with results/receiver_study/models -> ../models
RX="standard lmmse lmmse_da p_r_e_soft p_r_e_soft_da true_soft true_soft_da"
python -m jamloc.experiments.recovery --channel C --snr-db 15 --jsr-db 3 9 --seeds 0 \
    --max-mc-iter 25 --attacks pilot_whole_allocation --receivers $RX
python -m jamloc.experiments.recovery --channel C --snr-db 8 --jsr-db 3 9 --seeds 0 \
    --max-mc-iter 25 --attacks data partial pilot_whole_allocation --receivers $RX
# the same for seeds 1 and 2; then --plot-only per SNR pools the seeds and writes the paired CSVs
```

BLER (%), pooled over the seeds (`results/receiver_study/recovery/recovery_C{,_snr8}.csv`):

| SNR | Attack | JSR | Standard | No map | No map, DA | U-Net | U-Net, DA | True map | True map, DA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | all pilots | +3 | 39.8 | 11.1 | 2.27 | 3.62 | 1.29 | 3.65 | 1.31 |
| 15 | all pilots | +9 | 95.6 | 72.1 | 37.9 | 39.3 | 24.4 | 38.7 | 24.1 |
| 8 | all pilots | +3 | 42.6 | 14.3 | 3.58 | 11.1 | 4.08 | 10.9 | 4.08 |
| 8 | all pilots | +9 | 92.2 | 63.4 | 34.2 | 50.9 | 31.1 | 50.0 | 31.2 |
| 8 | data | +3 | 48.0 | 44.9 | 43.6 | 3.90 | 2.94 | 3.83 | 2.92 |
| 8 | data | +9 | 88.5 | 87.0 | 87.0 | 8.24 | 6.25 | 8.24 | 6.23 |
| 8 | partial | +3 | 60.8 | 50.9 | 48.4 | 5.52 | 3.96 | 5.40 | 3.88 |
| 8 | partial | +9 | 90.4 | 88.0 | 87.2 | 10.3 | 7.76 | 10.1 | 7.66 |

Paired differences on the slots both receivers decoded (percentage points, 95% interval;
`*_paired.csv`):

- U-Net map, without minus with passes: all pilots at 15 dB 2.33 (1.93 to 2.79) at +3 and 14.6
  (12.3 to 16.9) at +9; at 8 dB 6.90 (5.99 to 7.89) and 20.2 (17.3 to 23.0); data at 8 dB 0.96
  (0.69 to 1.26) and 1.94 (1.51 to 2.41); partial at 8 dB 1.56 (1.22 to 1.94) and 2.53 (1.98 to
  3.12).
- No map with passes minus U-Net map with passes, all pilots: at 15 dB 0.98 (0.64 to 1.36) at +3
  and 13.1 (10.5 to 15.6) at +9; at 8 dB **-0.50 (-0.91 to -0.10)** at +3 and 3.12 (0.90 to 5.35)
  at +9. At 8 dB and +3 dB the passes without a map end slightly lower.
- U-Net map with passes minus true map with passes: within 0.3 points everywhere, all intervals
  include zero.

**Reading.**

- With every pilot jammed, the passes cut the BLER with the map by 2 to 20 points; at +9 dB a
  quarter (15 dB) to a third (8 dB) of the blocks still fail, because their first estimate is too
  poor for the decoder to correct.
- When data cells are jammed, the map is what makes the passes work: without it they feed jammed
  data cells back as pilots and change little (at most 2.5 points on 45-88% BLER).
- The map also improves the starting estimate when every pilot is jammed (it down-weights them
  by their residual), which is why the map wins at +9 dB. At 8 dB and +3 dB that down-weighting
  makes the start slightly worse than plain LMMSE, and the passes without a map end 0.5 points
  lower.
- A blind start from the fourth power of the data symbols (Viterbi and Viterbi 1983), with the
  jammed pilots only resolving the 90-degree ambiguity, gave higher BLER than the pilot-based
  start in a 256-slot trial (all pilots jammed, 15 dB SNR: 56% against 47% at +9 dB before any
  pass) and was dropped.

**The passes on one slot (23 Nov).** The site's passes animation and the paper's figure
`data_aided_passes` (`experiments.figures`) follow the `aided_flip` slot (+9 dB JSR, every pilot of
the 12-PRB allocation jammed, TDL-C, 15 dB SNR, seed 7, slot 6). The mean squared error of the soft
U-Net receiver's channel estimate over the allocation's subcarriers is 0.565 after the first attempt
(CRC fail), 0.259 after pass 1 (CRC pass), 0.009 after pass 2 and below 0.001 after pass 3. The
receiver stops at pass 1 for this slot. Passes 2 and 3 ran because other slots of the batch had not
yet decoded. The mean certainty of the data symbols that pass 1 used as pilots (|mean|^2 of the
decoder's soft symbol) is 0.76, then 0.99 and 1.00. Each pass after the first attempt records this
per cell in the export (`passes[i].certainty`).

## 22. Partial-band walkthrough and the jammer's width

Partial-band jamming hits the pilots and the data cells of its PRBs, so one slot shows both uses of
the map. The site's walkthrough and the paper's Fig. 2 now use such a slot, and two questions the
4-PRB results leave open are answered here: how much of the allocation the test jammers cover, and
how detection and decoding change as the jammer widens.

**The walkthrough slot** (case `partial_flip`): TDL-C, 15 dB SNR, partial-band jamming at +3 dB
JSR on PRBs 10-13, all 14 symbols, the UE on PRBs 6-17 (a third of its pilots and data cells);
seed 7, the first slot of the 64-slot batch that the soft U-Net receiver recovers. In that batch the
standard receiver fails 43 of 64 slots and the soft U-Net receiver none (the 4-PRB recovery table
gives 53.8% and 0.08% at this point). In the slot:

- the U-Net flags all 56 jammed cells and no other cell;
- medians over the allocation, jammed against clean cells: power 17.1 against 15.3 dB, residual
  12.6 against -0.4 dB, EVM 14.1 against 0.2 dB, so the power alone barely shows this jammer;
- squared error of the channel estimate on the jammed PRBs (DMRS symbol, against the true
  channel): 0.140 standard (11.7 times its 0.012 on the clean PRBs), 0.045 with the map (-68%);
- mean |LLR| of the bits (mean of the two QPSK bits per point): standard 38.4 on jammed and 56.5 on
  clean cells, with the map 1.3 and 63.1.

**How much of the allocation the jammers cover.** Training, validation and test draw the jammer's
block (2-12 PRBs) anywhere on the carrier, independently of the allocation
(`results/width_sweep/allocation_cover.csv`). In the jammed test slots the block overlaps the
allocation in 85% of the slots and then covers a median of 37.5% of its PRBs; it covers more than
half of them in 21.5% (pilot), 22.0% (data) and 23.5% (partial-band) of the slots and all of them
in 2.2%, 2.0% and 2.5%. The recovery experiment's 4-PRB jammer covers a third.

**Detection against the width** (`python -m jamloc.experiments.width_sweep --part detection`,
`results/width_sweep/width_detection.csv`): the fused U-Net's Pd at 1% per-cell false alarm, on
the jammed cells inside the allocation, per attack, JSR and block width (66-117 slots per group),
means over the three training seeds. Pd does not fall as the block widens; the narrowest blocks are
the hardest. Pilot jamming at -9 dB: 0.76 at 2 PRBs (113 jammed pilot cells) and 0.88 at 12 PRBs
(694 cells); by the share of the allocation's PRBs the block covers, 0.82 for at most a quarter and
0.90 for more than three quarters. At -15 dB, Pd at 2 PRBs is 0.33 (pilot), 0.42 (data) and 0.45
(partial-band), against 0.56, 0.85 and 0.91 at 12 PRBs.

**Decoding against the width.** `recovery --jam-prb N` puts the jammer on N PRBs at a random place
inside the 12-PRB allocation. Run (TDL-C, 15 dB SNR, known noise power, three detector seeds, up to
1,600 slots each; `results/width_sweep/run.sh` runs the 18 jobs, 8 at a time):

```bash
JAMLOC_RESULTS_DIR=results/width_sweep  # with results/width_sweep/models -> ../models
python -m jamloc.experiments.recovery --channel C --snr-db 15 --jsr-db -3 3 9 --seeds 0 \
    --max-mc-iter 25 --jam-prb 8 --attacks pilot data partial \
    --receivers standard lmmse p_r_e_soft p_r_e_soft_da true_soft true_soft_da
# the same for --jam-prb 2 4 6 10 12 and seeds 1, 2; then, with the default results folder:
python -m jamloc.experiments.width_sweep --part decoding   # width_bler_C.csv and the figure
```

BLER pooled over the seeds (`results/width_sweep/width_bler_C.csv`; a receiver stops after 100
block errors, so denominators run from 384 to 4,800; "0/n": no error in n blocks):

Pilot jamming, BLER (%) by jammed PRBs of 12 (share of the allocation):

| JSR | Receiver | 2 (17%) | 4 (33%) | 6 (50%) | 8 (67%) | 10 (83%) | 12 (100%) |
|---|---|---|---|---|---|---|---|
| -3 dB | No protection | 0.02 | 0.10 | 0.31 | 0.46 | 0.90 | 1.29 |
| -3 dB | LMMSE, no map | 0/4800 | 0/4800 | 0.02 | 0.06 | 0.12 | 0.19 |
| -3 dB | U-Net map | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0.10 |
| -3 dB | U-Net map, data-aided | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 |
| -3 dB | True map | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0.10 |
| -3 dB | True map, data-aided | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 | 0/4800 |
| +3 dB | No protection | 0.73 | 4.96 | 12.26 | 19.41 | 28.77 | 37.83 |
| +3 dB | LMMSE, no map | 0.15 | 1.12 | 2.90 | 5.04 | 7.25 | 9.85 |
| +3 dB | U-Net map | 0/4800 | 0/4800 | 0.17 | 0.48 | 1.67 | 3.96 |
| +3 dB | U-Net map, data-aided | 0/4800 | 0/4800 | 0.04 | 0.17 | 0.60 | 1.48 |
| +3 dB | True map | 0/4800 | 0/4800 | 0.15 | 0.44 | 1.58 | 3.88 |
| +3 dB | True map, data-aided | 0/4800 | 0/4800 | 0.04 | 0.17 | 0.60 | 1.46 |
| +9 dB | No protection | 7.56 | 30.06 | 56.08 | 73.26 | 84.64 | 92.19 |
| +9 dB | LMMSE, no map | 4.06 | 16.28 | 30.51 | 45.18 | 55.38 | 66.49 |
| +9 dB | U-Net map | 0/4800 | 0.06 | 0.62 | 4.73 | 15.06 | 36.04 |
| +9 dB | U-Net map, data-aided | 0/4800 | 0.04 | 0.29 | 1.96 | 8.10 | 22.80 |
| +9 dB | True map | 0/4800 | 0.02 | 0.50 | 4.50 | 14.25 | 36.15 |
| +9 dB | True map, data-aided | 0/4800 | 0/4800 | 0.23 | 1.85 | 7.89 | 22.87 |

Data jamming, BLER (%) by jammed PRBs of 12 (share of the allocation):

| JSR | Receiver | 2 (17%) | 4 (33%) | 6 (50%) | 8 (67%) | 10 (83%) | 12 (100%) |
|---|---|---|---|---|---|---|---|
| -3 dB | No protection | 0.48 | 4.15 | 12.88 | 22.66 | 35.38 | 47.66 |
| -3 dB | LMMSE, no map | 0.46 | 4.54 | 13.11 | 23.51 | 36.72 | 49.48 |
| -3 dB | U-Net map | 0/4800 | 0/4800 | 0.02 | 1.15 | 5.06 | 12.26 |
| -3 dB | U-Net map, data-aided | 0/4800 | 0/4800 | 0.02 | 0.90 | 4.67 | 10.80 |
| -3 dB | True map | 0/4800 | 0/4800 | 0.02 | 1.15 | 5.02 | 12.15 |
| -3 dB | True map, data-aided | 0/4800 | 0/4800 | 0.02 | 0.90 | 4.62 | 10.76 |
| +3 dB | No protection | 6.85 | 37.39 | 66.49 | 84.90 | 96.09 | 98.18 |
| +3 dB | LMMSE, no map | 7.07 | 37.95 | 67.01 | 85.68 | 96.09 | 98.44 |
| +3 dB | U-Net map | 0/4800 | 0/4800 | 0.88 | 23.07 | 53.28 | 78.12 |
| +3 dB | U-Net map, data-aided | 0/4800 | 0/4800 | 0.54 | 21.67 | 51.99 | 77.54 |
| +3 dB | True map | 0/4800 | 0/4800 | 0.88 | 22.66 | 53.12 | 78.12 |
| +3 dB | True map, data-aided | 0/4800 | 0/4800 | 0.54 | 21.60 | 51.85 | 77.54 |
| +9 dB | No protection | 24.92 | 80.80 | 95.57 | 100.00 | 100.00 | 100.00 |
| +9 dB | LMMSE, no map | 22.44 | 80.80 | 94.79 | 100.00 | 100.00 | 100.00 |
| +9 dB | U-Net map | 0/4800 | 0.08 | 4.21 | 72.07 | 96.09 | 99.74 |
| +9 dB | U-Net map, data-aided | 0/4800 | 0.02 | 3.40 | 71.48 | 95.57 | 99.74 |
| +9 dB | True map | 0/4800 | 0.08 | 4.21 | 72.07 | 96.09 | 99.74 |
| +9 dB | True map, data-aided | 0/4800 | 0.02 | 3.40 | 71.48 | 95.57 | 99.74 |

Partial-band jamming, BLER (%) by jammed PRBs of 12 (share of the allocation):

| JSR | Receiver | 2 (17%) | 4 (33%) | 6 (50%) | 8 (67%) | 10 (83%) | 12 (100%) |
|---|---|---|---|---|---|---|---|
| -3 dB | No protection | 1.52 | 10.63 | 24.03 | 37.28 | 48.18 | 62.15 |
| -3 dB | LMMSE, no map | 0.92 | 7.28 | 20.05 | 30.06 | 43.63 | 56.08 |
| -3 dB | U-Net map | 0/4800 | 0/4800 | 0.15 | 4.08 | 12.27 | 23.72 |
| -3 dB | U-Net map, data-aided | 0/4800 | 0/4800 | 0.08 | 2.65 | 8.14 | 17.02 |
| -3 dB | True map | 0/4800 | 0/4800 | 0.12 | 4.08 | 12.15 | 23.65 |
| -3 dB | True map, data-aided | 0/4800 | 0/4800 | 0.08 | 2.67 | 8.08 | 17.02 |
| +3 dB | No protection | 13.88 | 52.97 | 78.79 | 91.67 | 97.14 | 99.74 |
| +3 dB | LMMSE, no map | 10.26 | 48.57 | 73.44 | 88.80 | 95.05 | 99.74 |
| +3 dB | U-Net map | 0/4800 | 0.06 | 2.12 | 38.34 | 66.84 | 89.58 |
| +3 dB | U-Net map, data-aided | 0/4800 | 0.06 | 1.17 | 33.96 | 59.55 | 85.68 |
| +3 dB | True map | 0/4800 | 0.06 | 2.06 | 38.34 | 66.84 | 89.58 |
| +3 dB | True map, data-aided | 0/4800 | 0.06 | 1.15 | 33.96 | 59.55 | 85.68 |
| +9 dB | No protection | 37.95 | 91.15 | 98.70 | 100.00 | 100.00 | 100.00 |
| +9 dB | LMMSE, no map | 35.42 | 88.80 | 98.44 | 100.00 | 100.00 | 100.00 |
| +9 dB | U-Net map | 0/4800 | 0.10 | 6.64 | 81.70 | 98.44 | 100.00 |
| +9 dB | U-Net map, data-aided | 0/4800 | 0.04 | 4.17 | 79.69 | 97.66 | 100.00 |
| +9 dB | True map | 0/4800 | 0.08 | 6.57 | 81.70 | 98.44 | 100.00 |
| +9 dB | True map, data-aided | 0/4800 | 0.02 | 4.15 | 79.69 | 97.66 | 100.00 |


- The U-Net map stays within 0.8 percentage points of the true map at every width, attack and
  JSR (largest gap: pilot jamming, +9 dB, 10 PRBs).
- Pilot jamming degrades gradually: the clean pilots around the jammed band carry the estimate.
  At +3 dB the U-Net map stays under 1% up to 8 PRBs (0.48%) and reaches 3.96% at 12; at +9 dB,
  36.0% at 12 PRBs (95% Wilson 33.1-39.1%), 22.8% with the data-aided passes. The 12-PRB pilot
  point is an independent replicate of section 21's whole-allocation run (39.3% and 24.4%) and
  agrees with it within the sampling error.
- Data and partial-band jamming fall off a cliff between 6 and 8 jammed PRBs. A block has 3,744
  coded bits and 1,416 information bits, plus a 16-bit CRC (code rate about 0.37). With 6 of 12 PRBs jammed the
  clean half still holds 1,872 coded bits; with 8 jammed the clean third holds 1,248, fewer than the
  information bits, so the block must draw on the jammed cells. Partial-band at +3 dB: 0.06% at 4
  PRBs, 2.12% at 6, 38.3% at 8 (the true map the same); at -3 dB, where jammed cells still carry
  information, 4.1% at 8 and 23.7% at 12 against 62.2% without protection.
- The data-aided passes help most under pilot jamming (+3 dB, 12 PRBs: 3.96% to 1.48%) and less
  under data and partial-band jamming, where the decoded symbols of jammed data cells are excluded
  and the clean cells are already used.
- The 4-PRB points agree with the recovery table of section 17 (partial-band at +3 dB: 0.06%
  against 0.08% there).

## 23. Known-symbol residuals after decoding and a receiver that refines its map

Round of 23-24 Nov 2026 (branch `decoder-aided-passes`). Three questions: can the residual theory
be stated for any known symbols, so that it also covers decoded data; can decoding fix what the
detector misses; and does the receiver need a map at all.

**One law for every known symbol** (`jamloc.recovery.MapAwareReceiver.slot_residuals`,
`decoded_cell_test`). Once a block passes its CRC, every data symbol is known, so every data
resource element gives an LS value of the channel. For each resource element, a straight line in
time is fitted by weighted least squares to the LS values of the same subcarrier on the other data
symbols, and the residual is the difference. The prediction leaves the tested value out, so the
twelve residuals of a cell are independent with equal variance (QPSK) and the cell's normalized
energy is Gamma(12, 1/12) without jamming. DMRS pilots are tested against the same prediction
(Gamma(6, 1/6)). Strong outliers are peeled off one per PRB row and round before testing, so that
jammed cells do not leak into the predictions of clean ones. This is the general proposition (any
annihilating predictor, any group of known symbols) applied in time instead of in frequency.

Calibration (`python -m jamloc.experiments.residual_theory --only-decoded`, results in
`results/theory/residual_calibration.csv`; measured fraction below the analytic level):

| Residual | Channel | 0.1% | 1% | 10% | Cells |
| --- | --- | --- | --- | --- | --- |
| DMRS, benign test slots | TDL-A | 0.08% | 1.05% | 10.1% | 29,664 |
| DMRS, benign test slots | TDL-C | 0.10% | 0.95% | 9.9% | 30,144 |
| DMRS, `longdelay` | TDL-C 1000 ns | 54.5% | 62.9% | 76.3% | 15,520 |
| Decoded data, CRC-pass benign | TDL-A | 0.10% | 1.01% | 10.0% | 430,768 |
| Decoded data, CRC-pass benign | TDL-C | 0.09% | 0.96% | 9.9% | 424,723 |
| Decoded data | TDL-C 1000 ns | 0.09% | 0.93% | | 197,600 |
| Decoded data | TDL-C 30 m/s | 0.17% | 1.19% | | 195,572 |

The decoded-data residual has no frequency model, so the long delays that break the DMRS residual
do not affect it. The straight line matters: with a constant fit, the benign false-alarm rate at
25 dB was 3.1% at 3 m/s, and at 30 m/s 20% (15 dB) and 52% (25 dB), against 1.0-1.1% and 2.8% with
the line (transmitted symbols, 192 slots per point).

**Localization after decoding** (`python -m jamloc.experiments.decoded_map --split S --seed s
--per-point 8 --refine`; the slots of each test grid are replayed exactly with the seeds of
`dataset.generate`, first 8 batches of 16 slots per grid point). Maps compared on the same cells:
`unet` (before decoding), `post` (exact tests after decoding: decoded-data residual on the data and
DMRS cells of slots whose CRC passed, Gamma(12) power test on unscheduled cells, all at 1%; the
allocation of a slot that failed keeps the U-Net map), `fused` (U-Net as prior: level 10% on cells
the U-Net flags, 1% elsewhere), `genie` (as `post` with the transmitted symbols on every slot).
Decoding by the soft receiver with shrinkage estimates and refining passes (section 24; rerun on
24 Nov in `results/da_study/loc4`, every F1 within 0.001 of the earlier per-cell run).

F1 over all cells, mean over the U-Net's training seeds 0, 1 and 2 (`results/decoded_map/table.csv`,
from `--table results/decoded_map/*_refine.csv`; the range over seeds is at most 0.03):

| Grid | U-Net | Exact tests | Fused | Genie | Benign FA, exact tests |
| --- | --- | --- | --- | --- | --- |
| test (nominal) | 0.951 | 0.887 | 0.912 | 0.886 | 1.01% |
| sched | 0.936 | 0.897 | 0.920 | 0.897 | - |
| combjam | 0.953 | 0.886 | 0.913 | 0.885 | 0.98% |
| qpskjam | 0.956 | 0.890 | 0.917 | 0.889 | 1.03% |
| neighbor | 0.961 | 0.907 | 0.933 | 0.906 | 0.99% |
| hopjam | 0.597 | 0.877 | 0.865 | 0.884 | 1.00% |
| pulsejam | 0.638 | 0.878 | 0.860 | 0.879 | 0.96% |
| tonejam | 0.768 | 0.585 | 0.657 | 0.583 | 0.96% |
| spoofjam | 0.027 | 0.255 | 0.228 | 0.773 | 1.00% |
| spoofpart | 0.138 | 0.510 | 0.433 | 0.683 | 1.02% |

- The exact tests stay at their 1% design rate on every grid's benign slots (0.96-1.03% over all
  cells); the U-Net varies between 0.59% and 1.40%, the fusion 1.04-1.20%.
- The network wins on the attacks it was trained on (it pools weak evidence across a PRB row);
  the exact tests win on every support it never saw, without training.
- Spoofing: the scheduled-cell F1 of the exact tests is 0.46-0.79 between -15 and -6 dB JSR
  (U-Net 0.02-0.03). From 0 dB on most spoofed slots no longer decode (547/768 decoded at 0 dB, 319 at +3 dB and
  182 at +9 dB, over the three seeds), and the network's map is kept there.
- Multi-tone: the sum over a cell dilutes a single tone. With `--peaks` (sum and largest error
  each at half the level, both laws exact) F1 rose from 0.594 to 0.640 on 2 batches per point and
  fell by 0.004 on the nominal grid; the sum stays the default.
- Figure `after_decoding` (`python -m jamloc.experiments.figures --only after_decoding`): one
  hopping slot, 89 false alarms and 20 misses before decoding, 3 and 0 after.

**Receiver.** New options of `MapAwareReceiver` (receivers in `experiments.recovery`):
`cellwise` (jamming power per flagged cell from its own 12 resource elements instead of the
average over the flagged data cells of its PRB), `refine` (after a failed decoding, a data cell
whose remodulated error energy exceeds the Gamma(12) 1% quantile joins the map), `renoise` (after
the data-aided channel update, the noise power is the value for which the median of
`|y - h x_bar|^2 / (|h|^2 v + N0)` over confident symbols of clean cells is ln 2; then pilots,
residuals, channel and LLRs are recomputed). `p_r_e_full` / `true_full` use all three with three
passes; `cell_soft` / `cell_full` use no map (DMRS CFAR pilot weights, per-cell estimate on every
data cell).

Bug found and fixed during the round: the first `renoise` used the filter's error variance under
the wrong noise power and came out several dB low (commit 8aa773e). Numbers from before the fix are
not used.

All receivers of a jammer decode the same slots (three detector seeds, up to 1,600 slots each, a
receiver stops after 100 block errors). Runs: `results/da_study/{table_nom,table_nom_old,
table_whole,table,table_old,table_full}`, merged per seed into `results/receiver_v7/recovery`
(pooled `recovery_C{,_hopping,_pulsed}.csv`, paired differences in `*_paired.csv`). The older replay
files in `results/recovery` are on different slots (the draws of a point depend on the JSR grid run
before it) and are not mixed in. BLER (%), TDL-C, 15 dB SNR, known noise power, +9 dB JSR on 4 of
12 PRBs, or every DMRS cell of the allocation (last two columns):

| Receiver | nom. pilot | nom. data | nom. partial | pulsed data | pulsed partial | hop data | hop partial | all pilots +3 | all pilots +9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No protection | 30 | 79 | 91 | 19 | 45 | 26 | 61 | 40 | 96 |
| LMMSE, no map | 16 | 80 | 86 | 18 | 35 | 25 | 50 | - | - |
| Slot-wide estimate | 12 | 81 | 91 | 60 | 69 | 75 | 84 | - | - |
| DMRS only | 0.77 | 80 | 0.77 | 18 | 10 | 25 | 58 | - | - |
| Power test, hard | 4.6 | 20 | 24 | 1.4 | 4.9 | 1.4 | 12 | - | - |
| Block test, soft | 0.04 | 0 | 7.8 | 0.06 | 2.7 | 51 | 64 | - | - |
| U-Net, hard | 0.29 | 0.08 | 0.15 | 0.54 | 0.62 | 35 | 51 | - | - |
| U-Net, soft per PRB | 0.06 | 0 | 0.04 | 0.06 | 0.38 | 21 | 31 | 3.6 | 39 |
| ... with channel passes | 0.04 | 0 | 0.02 | 0 | 0.29 | 20 | 24 | 1.3 | 24 |
| U-Net, soft per cell | 0.06 | 0 | 0.06 | 0 | 0.33 | 1.6 | 10 | 3.6 | 39 |
| ... with refining passes (`p_r_e_full`) | 0 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.98 | 22 |
| U-Net, mixed attacks, soft | 0.04 | 0 | 0.06 | 0 | 0.04 | 0 | 0.23 | - | - |
| No map, per cell | 0.04 | 0 | 0.08 | 0 | 0.02 | 0 | 0.35 | 4.6 | 44 |
| ... with refining passes | 0.02 | 0 | 0.04 | 0 | 0 | 0 | 0.15 | 1.3 | 25 |
| True map per PRB, soft | 0.02 | 0 | 0.02 | 0.06 | 0.08 | 50 | 58 | - | - |
| True map, soft per PRB | 0.02 | 0 | 0.02 | 0 | 0 | 0 | 0.12 | 3.6 | 39 |
| True map, per cell, refining | 0 | 0 | 0 | 0 | 0 | 0 | 0.08 | 1.0 | 21 |

Zeros are 0 of 4,800. Without a jammer and at -15 dB JSR every receiver had no block error in
4,800 slots. Paired differences (percentage points, 95% interval, common slots):

- Hopping data: per-PRB minus full 21.1 (19.1 to 23.1); per cell minus full 1.56 (1.24 to 1.95);
  full equals the true map and the mixed-attack U-Net (0.00, -0.08 to 0.08).
- Hopping partial: per-PRB minus full 30.8 (28.0 to 33.7); per cell minus full 10.1 (9.1 to 11.3);
  full minus true full 0.17 (0.06 to 0.33); mixed-attack minus full -0.02 (-0.19 to 0.14).
- Pulsed partial: per-PRB minus full 0.38 (0.22 to 0.59).
- Every pilot jammed: per-PRB minus full 2.65 (2.22 to 3.14) at +3 dB and 16.7 (14.3 to 19.2) at
  +9 dB; channel passes minus full 0.31 (0.11 to 0.54) and 2.46 (1.25 to 3.68); full minus true
  full -0.02 (-0.13 to 0.08) and 0.61 (0.12 to 1.12); no map minus map, one attempt 0.94 (0.60 to
  1.29) and 4.43 (2.61 to 6.23), with the refining passes 0.29 (0.10 to 0.51) and 2.96 (1.60 to
  4.33). The map matters here because the excess energy of clean cells is measured against a
  poor channel estimate.
- Nominal attacks: every difference among the soft receivers is within noise (all intervals
  include zero); the power test with hard use is 4.6-24 points worse than the full receiver.

Estimated noise power (`--noise-estimate comb`, Colab, `recovery_C_nest*`), every pilot jammed,
three seeds: per-PRB soft 52% (+3 dB) and 98% (+9 dB), channel passes 15.3% and 84%, full 4.4% and
68%, true map full 4.3% and 68%, map-free full 4.2% and 68%. Channel passes minus full 10.6 (9.2 to
12.0) and 17.2 (13.4 to 21.1); full minus true full 0.10 (-0.06 to 0.27) and 0.87 (0.08 to 1.67).
Partial-band on 4 PRBs: every map receiver at or below 0.38%.

16-QAM (MCS 13, 18 dB SNR, seeds 0 and 1, `recovery_C_snr18_mcs13*`), BLER with the true map per
PRB / true map per cell / no map per cell: pilot +3 dB 0.53 / 0.53 / 1.75%, +9 dB 1.0 / 1.0 / 5.4%;
data 4.0 / 7.4 / 19.7% and 7.8 / 8.9 / 19.7%; partial-band 5.6 / 7.5 / 21.1% and 7.8 / 8.8 /
28.1%. Without jamming no receiver erred in 3,200 slots. On a clean cell the per-cell excess energy
has the variance (2|h|^2 N0 + N0^2 + |h|^4 kappa) per resource element, kappa = Var|x|^2 = 0 for
QPSK and 0.32 for 16-QAM, so without a map 16-QAM cells get false jamming power.

The 8 dB SNR run of the same receivers (Colab, `recovery_C_snr8*`, all nominal cases) is in
`results/receiver_v7/recovery`. Its findings are superseded by the ablation of section 24.


## 24. Shrinkage of the per-cell jamming estimate

Round of 24 Nov 2026 (branch `decoder-aided-passes`). At 8 dB SNR the per-cell receiver of
section 23 lost 0.4-1.3 points against the per-PRB receiver with channel passes. The question was
how to keep the per-cell power of a jammer that hops between symbols without paying the noise of a
12-element estimate when the jammer fills the PRB row.

**Ablation at 8 dB** (`results/da_study/abl8`, U-Net map, TDL-C, +3 dB JSR, three seeds of 1,600
slots). Partial-band jamming: per-PRB with channel passes 3.90%, per-PRB with refining passes
3.85%, per-cell with channel passes 4.48%, per-cell with refining passes and noise re-estimation
4.52%. Every pilot jammed: 4.77%, 4.69%, 4.77% and 5.83%. The per-cell estimate costs about half a
point under block jamming, and re-estimating the noise power when it is known costs another point
with every pilot jammed. The map refinement itself costs nothing.

**Pooling rules tried.**

- z-threshold (commit 24d0b5e, since removed): a cell keeps the PRB mean when its own estimate is
  within z standard errors of it. With z = 2.5 the 8 dB cost was 0.27 points (4.17% against
  3.90%). A sweep with the true map (one attempt, 1,280 slots, seed 4242, partial +3 dB at 8 dB)
  gave per-PRB 5.47%, per-cell 6.17%, z = 2.5 5.78%, z = 4 5.55%, z = 8 and 16 5.47%. z = 8
  recovered the per-PRB result (8 dB, U-Net map: 3.90% with channel passes, 3.85% refining) and kept
  hopping close (one attempt 1.58% / 10.7%, refining 0 / 0.35%), but it is a tuned constant.
- Random-effects shrinkage (`cellwise="shrink"`, commit a34c4c6): each flagged cell's raw estimate
  is pulled towards the row mean with the weight tau2 / (tau2 + s2), where s2 is the sampling
  variance of the cell's estimate (spread of its 12 terms over 12) and tau2 the spread of the raw
  estimates over the row minus their mean sampling variance, floored at zero (Morris 1983,
  DerSimonian and Laird 1986). No parameter. Unit test: a row the jammer fills stays within 13% of
  the row mean's deviation that per-cell estimates show, and in a row the map flags whole but the
  jammer hits every other symbol, the jammed cells get 3.3 (true 4) and the clean ones 0.03, where
  the row mean gives both 2.2.

**Final runs** (`results/da_study/final/{nom,whole,whole1,whole2,hop,pulse,nest2}`, merged with the
earlier runs by `merge_runs.py` into `results/receiver_v7/recovery`; every run carried one receiver
of the earlier runs, and its counts matched them exactly, so all receivers share their slots; the
Colab and local runs of seeds 0 and 1 of the nominal grid agree in all 28 counts). BLER (%), TDL-C,
15 dB SNR, known noise power, +9 dB JSR on 4 of 12 PRBs or every DMRS cell of the allocation:

| Receiver | nom. pilot | nom. data | nom. partial | pulsed data | pulsed partial | hop data | hop partial | all pilots +3 | all pilots +9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net, soft per PRB | 0.06 | 0 | 0.04 | 0.06 | 0.38 | 21 | 31 | 3.6 | 39 |
| U-Net, shrinkage (`p_r_e_shrink`) | 0.06 | 0 | 0.04 | 0 | 0.33 | 1.5 | 10 | 3.6 | 39 |
| ... with channel passes | - | - | - | - | - | 1.2 | 6.0 | 1.3 | 24 |
| ... with refining passes (`p_r_e_shrink_full`) | 0 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.90 | 20 |
| No map, shrinkage, passes (`cell_shrink_full`) | 0.02 | 0 | 0.04 | 0 | 0 | 0 | 0.12 | 1.0 | 24 |
| True map, shrinkage, refining (`true_shrink_full`) | 0 | 0 | 0 | 0 | 0 | 0 | 0.08 | 0.85 | 19 |

The channel-pass hopping entries are from `results/da_study/shrink_hop` (same slots, the hopping
run has no other cases). Paired differences (points, 95% Newcombe interval, common slots):

- Hopping: per-PRB minus shrinkage-refining 21.1 (19.1 to 23.1) data and 30.7 (27.9 to 33.6)
  partial; shrinkage one attempt minus refining 1.52 (1.20 to 1.91) and 9.84 (8.82 to 11.0); per
  cell minus shrinkage, one attempt, 0.04 (-0.08 to 0.16) and 0.17 (-0.08 to 0.43); refining minus
  true map refining 0.00 and 0.17 (0.03 to 0.34); minus the map-free receiver 0.00 and 0.12
  (-0.03 to 0.30); the old per-cell refining receiver and the mixed-attack U-Net within 0.02.
- Every pilot jammed: channel passes minus refining 0.40 (0.18 to 0.64) at +3 dB and 4.76 (3.52
  to 6.04) at +9 dB; refining minus true map 0.04 (-0.06 to 0.15) and 0.44 (-0.04 to 0.92); map
  minus no map -0.15 (-0.33 to 0.03) and -3.67 (-4.94 to -2.43); the old per-cell refining receiver
  (ungated noise re-estimation) minus refining 0.08 (-0.06 to 0.23) and 2.17 (1.27 to 3.10). In one
  attempt the map-free shrinkage receiver loses 0.90 (0.56 to 1.25) and 4.43 (2.59 to 6.25) points
  more than the U-Net map.
- Nominal and pulsed: all differences among the soft receivers within 0.4 points; pulsed partial
  per-PRB minus refining 0.38 (0.22 to 0.59).

**8 dB with shrinkage** (`results/receiver_v7/ablation8`, slots of `abl8`): partial-band 4.02%
with channel passes and 4.00% refining, every pilot 4.77% and 4.67%. Refining minus per-PRB channel
passes +0.10 (-0.04 to 0.26) and -0.10 (-0.29 to 0.07): the low-SNR cost is gone. Per-cell minus
shrinkage, channel passes, partial: 0.46 (0.24 to 0.70). True map one attempt: per-PRB 5.35%,
shrinkage 5.54% (difference 0.19, 0.02 to 0.36).

**16-QAM** (MCS 13, 18 dB SNR, seeds 0 and 1, Colab, `colab_qam16b` on the slots of
`colab_qam16`). BLER (%) at +3 / +9 dB JSR:

| Receiver | pilot | data | partial |
| --- | --- | --- | --- |
| True map, per PRB | 0.53 / 1.0 | 4.0 / 7.8 | 5.6 / 7.8 |
| True map, shrinkage | 0.53 / 1.0 | 4.3 / 7.9 | 5.7 / 7.9 |
| True map, per cell | 0.53 / 1.0 | 7.4 / 8.9 | 7.5 / 8.8 |
| No map, shrinkage | 1.1 / 4.0 | 6.2 / 11 | 9.6 / 17 |
| No map, per cell | 1.8 / 5.4 | 20 / 20 | 21 / 28 |

Shrinkage minus per-PRB with the true map: 0.25 (0.05 to 0.47) data +3 dB, the other points
within -0.4 to 0.3; per cell minus shrinkage 0.9 to 3.1 points (all intervals above zero). Without
a map, shrinkage removes 9-13 points of the per-cell loss, and the map still saves 0.5 to 9.2
points (partial +9 dB: 7.9% against 17%).

**Noise re-estimation gate** (commit 45d6c0f). The automatic gate first ran the re-estimation only
when the map flagged a DMRS cell. The map-free receiver flags DMRS cells with the residual test,
whose normalization is the very noise estimate a jammed comb inflates, so its gate stayed shut
(estimated noise, every pilot +3 dB: 8.2% against 4.2% with the U-Net map). The gate now opens
whenever the noise power was estimated in the slot and stays shut when it is known.
Estimated noise power (`--noise-estimate comb`, every pilot jammed, `final/nest2` on the slots of
`colab_nest`): refining passes 4.21% (+3 dB) and 67.9% (+9 dB), true map 4.08% and 67.9%, no map
4.04% and 69.3%, channel passes 15.3% and 84.4%, one attempt 52% and 98%. Channel passes minus
refining 11.2 (9.9 to 12.6) and 18.0 (14.1 to 21.9); refining minus true map 0.13 (-0.01 to 0.27)
and 0.00 (-0.63 to 0.63); map minus no map 0.17 (-0.13 to 0.46) and -1.39 (-2.64 to -0.15). With
the old gate the map-free receiver had 8.19% at +3 dB; the map receivers are unchanged by the new
gate. Partial-band on 4 PRBs: every soft receiver at or below 0.23%.

Conclusions for the paper: shrinkage is the proposed estimator (no tuning, matches per-PRB under
block jamming and per-cell under hopping, best of all rules in 16-QAM). The map-free receiver with
shrinkage is a strong baseline for QPSK. The map matters with every pilot jammed at high JSR and
with 16-QAM.

## 25. Resolution of the jamming estimate and the value of a map

Round of 24-25 Nov 2026 (branch `resolution-study`). After the review of the v7 drafts, the
question became which part of the receiver needs cell resolution. Two kill rules were fixed before
the runs: drop the resolution claim if the best per-PRB estimate (any of the scales 1, 2, 4) came
within 1 BLER point of shrinkage everywhere, and drop the map claim if no map beat the receiver
without one.

**Design.** The soft receiver is crossed over maps and estimates of the data cells' jamming power.
Maps: none (`nomap_*`, every data cell estimated, every pilot weighted by its residual with no
threshold, `pilot_weights="all"`), the EVM threshold (`evm_*`), the U-Net (`p_r_e_*`), the true
map coarsened to PRB rows (`true_prb_*`) and the true map (`true_*`). Estimates: per PRB (average
over the row's flagged data cells), per PRB times 2 or 4 (`*_prb2`, `*_prb4`, `scale`), per cell,
and shrinkage. Every run decodes the slots of `results/receiver_v7` (same seeds, cases, JSR grid),
so the new receivers pair with the earlier ones. Runs in `results/res_study/{nom,whole,hop,pulse}`,
merged into `results/res_study/merged/recovery` (`python -m jamloc.experiments.resolution merge`).
Canaries: `lmmse` reproduced every count and every per-slot outcome of the earlier runs in every
case and seed. `p_r_e_shrink` reproduced nominal, every-pilot and pulsed exactly and differed on 3
of about 8,000 hopping slots (one per seed and attack at most), which comes from the summation
order of the GPU-deterministic sums (42450a5). The merged files keep the earlier rows.

**Table** (BLER %, QPSK, TDL-C, 15 dB SNR, known noise power, one attempt, +9 dB JSR on 4 PRBs,
three seeds, `python -m jamloc.experiments.resolution table results/res_study/merged/recovery`):

| Map | Estimate | pilot | data | partial | pulsed data | pulsed partial | hop data | hop partial | all pilots +3 | all pilots +9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no protection | - | 30.1 | 79.5 | 90.9 | 19.1 | 45.2 | 26.2 | 60.6 | 39.8 | 95.6 |
| none | per PRB | 0.04 | 0.02 | 0.06 | 0.08 | 0.08 | 50.0 | 58.5 | 4.50 | 44.0 |
| none | per cell | 0.04 | 0.02 | 0.08 | 0 | 0.02 | 0 | 0.35 | 4.56 | 43.9 |
| none | shrinkage | 0.04 | 0.02 | 0.06 | 0 | 0.02 | 0 | 0.42 | 4.52 | 43.9 |
| EVM | per PRB | 16.0 | 2.19 | 9.71 | 0.10 | 6.06 | 0.10 | 16.3 | 11.3 | 71.7 |
| EVM | shrinkage | 16.0 | 2.21 | 9.59 | 0.10 | 6.47 | 0.12 | 16.2 | 11.3 | 71.7 |
| U-Net | per PRB | 0.06 | 0 | 0.04 | 0.06 | 0.38 | 21.1 | 30.9 | 3.62 | 39.3 |
| U-Net | per cell | 0.06 | 0 | 0.06 | 0 | 0.33 | 1.56 | 10.4 | 3.62 | 39.3 |
| U-Net | shrinkage | 0.06 | 0 | 0.04 | 0 | 0.33 | 1.52 | 10.1 | 3.62 | 39.3 |
| true per PRB | per PRB | 0.02 | 0 | 0.02 | 0.06 | 0.08 | 50.0 | 58.3 | - | - |
| true per PRB | shrinkage | 0.02 | 0 | 0.02 | 0 | 0 | 0 | 0.15 | - | - |
| true | per PRB | 0.02 | 0 | 0.02 | 0 | 0 | 0 | 0.12 | 3.65 | 38.7 |
| true | shrinkage | 0.02 | 0 | 0.02 | 0 | 0 | 0 | 0.12 | 3.65 | 38.7 |

Every arm decoded all 4,800 benign slots. Scaling the per-PRB estimate by 2 or 4 never lowered
its BLER (the best-scale row equals the per-PRB row for both maps). The DMRS-only per-PRB receiver
(`dmrs_ie`) fails on 79.9% under data jamming, so the data-cell average is the strongest per-PRB
choice. A receiver stops after 100 block errors, so high BLERs rest on fewer slots (for example
576 slots for no-map per PRB under hopping partial).

**Paired differences** (points, 95% Newcombe interval, common slots, `*_paired.csv`):

- Resolution without a map, hopping: per PRB minus shrinkage 50.0 (46.4 to 53.5) data and 58.3
  (54.2 to 62.3) partial. With the U-Net map 19.8 (17.9 to 21.8) and 19.8 (17.4 to 22.3). With the
  true map coarsened to PRBs 50.0 and 58.2. Pulsed 0.04 to 0.08, intervals including zero. Block
  jamming 0.00.
- Per cell minus shrinkage at 15 dB: within -0.06 to +0.17 everywhere, intervals including zero.
- No map against the true map, hopping partial: shrinkage without a map is 0.29 (0.14 to 0.49)
  points worse. Every pilot jammed: 0.88 (0.55 to 1.23) and 5.08 (3.35 to 6.80) worse.
- U-Net map minus no map (shrinkage): hopping 1.52 (1.20 to 1.91) and 9.77 (8.74 to 10.90), pulsed
  partial 0.31 (0.16 to 0.52), every pilot jammed -0.90 (-1.25 to -0.56) and -4.43 (-6.25 to
  -2.59), nominal blocks within 0.04.
- EVM map minus no map (shrinkage): pilot 15.9 (14.4 to 17.6), data 2.19 (1.80 to 2.64), partial
  9.53 (8.56 to 10.58), pulsed partial 6.45, hopping partial 15.8, every pilot 6.58 and 27.4 (23.6
  to 31.1). Pulsed and hopping data 0.10 and 0.12. The EVM threshold (1% of clean validation data
  cells over 5-25 dB SNR) detects 84.7% of the jammed data cells under data jamming at +9 dB and
  15 dB SNR (768 slots), with no false alarm, and the missed cells keep zero jamming power.
- Pilot weighting: the no-map receiver with the analytic 1% residual test on the pilots
  (`cell_shrink`) and with every pilot weighted by its residual (`nomap_shrink`) have no discordant
  slot with every pilot jammed.

Kill rules: resolution holds (per PRB 20 to 58 points worse under hopping for every map short of
the exact one). The map claim holds in a narrower form: with QPSK and clean pilots no map beats
none, and the U-Net map helps only with every pilot jammed (QPSK at 15 dB, see 16-QAM and 8 dB below).

Conclusion: the resolution has to come from the map or from the estimate. A per-PRB average over
the flagged cells of the exact map is as good as shrinkage, but no detector is exact for a jammer
it has not seen, so the estimate should carry the resolution.

**16-QAM** (MCS 13, 18 dB SNR, seeds 0 and 1, Colab `results/res_study/qam16` on the slots of
`colab_qam16`, merged into `results/res_study/merged/recovery`; canaries `lmmse` and
`true_shrink` reproduced their counts). BLER (%) at +3 / +9 dB:

| Map | Estimate | pilot | data | partial |
| --- | --- | --- | --- | --- |
| none | per PRB | 1.06 / 4.03 | 5.78 / 10.4 | 9.03 / 16.5 |
| none | per cell | 1.62 / 5.41 | 19.7 / 19.9 | 20.5 / 28.2 |
| none | shrinkage | 1.25 / 4.03 | 6.47 / 10.8 | 9.78 / 16.8 |
| U-Net | per PRB | 0.75 / 1.53 | 5.25 / 9.34 | 8.09 / 12.4 |
| U-Net | shrinkage | 0.75 / 1.53 | 5.97 / 9.59 | 8.59 / 12.7 |
| true | per PRB | 0.53 / 1.00 | 4.00 / 7.78 | 5.59 / 7.84 |
| true | per cell | 0.53 / 1.00 | 7.38 / 8.91 | 7.47 / 8.84 |
| true | shrinkage | 0.53 / 1.00 | 4.25 / 7.94 | 5.66 / 7.94 |

Scaling never helped (best scale = per PRB). The EVM threshold, calibrated on QPSK, fails on 19 to
75% and is not a meaningful 16-QAM baseline (the EVM feature also slices to QPSK points). Paired
differences (points, common slots):

- Without a map, per cell minus shrinkage 0.38 (0.15 to 0.65) to 12.8 (11.4 to 14.3), largest
  under data jamming at +3 dB. Per PRB minus shrinkage -0.19 to -0.75 (partial +3 dB: -0.75, -1.10
  to -0.42), so on a jammer that fills its block shrinkage gives up at most 0.75 points to the row
  average.
- With the true map, per cell minus shrinkage 0.9 to 3.1, per PRB minus shrinkage -0.25 to 0.
- U-Net map minus no map (shrinkage): -0.50 (-0.82 to -0.24) to -4.32 (-5.18 to -3.49), every
  point significant. True map minus no map -0.72 to -9.29. U-Net minus true map 0.22 to 4.78.
- Analytic pilot test minus weighting every pilot (no map, shrinkage): -0.19 to 0.06, significant
  only at pilot and data +3 dB (-0.19 and -0.25).

In 16-QAM the learned map pays off at every point, and shrinkage removes nearly all of the noise
cost of the cell's own estimate.

**JSR sweeps** (new slots, QPSK, TDL-C, 15 dB, three seeds, -9 to +9 dB in 3 dB steps,
`results/res_study/sweep_{hop,nom}`, figure `results/res_study/resolution_sweep.pdf` from
`python -m jamloc.experiments.resolution sweep`). Receivers: no protection, LMMSE, no map per PRB,
U-Net per PRB, EVM + shrinkage, U-Net + shrinkage, no map + shrinkage, true map + shrinkage.

- Hopping data: shrinkage without a map and the true map decode all 4,800 slots at every JSR.
  Per PRB without a map 0.54% at 0 dB, 4.7% at +3, 19.6% at +6 and 53.0% at +9, worse than no
  protection (15.4% and 27.7%) from +6 dB. U-Net + shrinkage up to 1.44%, EVM + shrinkage up to
  0.75%.
- Hopping partial: shrinkage without a map 0.02, 0.06, 0.25, 0.52% from 0 to +9 dB, the true map
  0.02, 0.04, 0.08, 0.21% (largest gap 0.31 points, unpaired). U-Net + shrinkage 0.42 to 9.66%,
  per PRB without a map up to 62%.
- Pilot: every receiver that estimates the jamming power at or below 0.06%. EVM + shrinkage 16.8%
  at +9 dB, the same as LMMSE without weights (17.2%), since the EVM map never flags a pilot.
- Partial-band block: every estimating receiver at or below 0.19%, EVM + shrinkage up to 13.5%.

**16-QAM under hopping** (MCS 13, 18 dB SNR, new slots, three seeds, target 200 block errors,
`results/res_study/qamhop`). BLER (%) at +3 / +9 dB.

| Map | Estimate | hop data | hop partial |
| --- | --- | --- | --- |
| no protection | - | 75.7 / 92.8 | 85.4 / 97.7 |
| none | per PRB | 82.9 / 97.8 | 86.1 / 99.0 |
| none | per cell | 1.33 / 1.52 | 9.50 / 19.3 |
| none | shrinkage | 1.77 / 1.65 | 11.4 / 21.3 |
| U-Net | per PRB | 81.6 / 95.6 | 83.6 / 97.9 |
| U-Net | shrinkage | 8.32 / 9.77 | 24.3 / 40.3 |
| true per PRB | shrinkage | 1.62 / 1.60 | 9.71 / 15.0 |
| true | per PRB | 0.21 / 0.31 | 2.98 / 7.23 |
| true | per cell | 0.23 / 0.40 | 3.48 / 7.60 |
| true | shrinkage | 0.21 / 0.35 | 3.08 / 7.52 |

Paired differences in points on common slots, at +3 / +9 dB.

- Resolution. Per PRB minus shrinkage without a map is 80.7 / 95.6 (data) and 74.5 / 75.9 (partial),
  on 768 slots each. Per PRB fails more often than no protection.
- Per cell minus shrinkage without a map is -0.44 (-0.75 to -0.15) / -0.12 (-0.33 to 0.07) data and
  -1.85 (-2.43 to -1.29) / -1.84 (-2.41 to -1.28) partial. Where the jammer moves, the cell's own
  estimate is up to 1.9 points better than shrinkage (on the 16-QAM blocks shrinkage is up to 12.8
  points better).
- U-Net map minus no map, both with shrinkage, is +6.57 (5.86 to 7.32) / +8.02 (7.22 to 8.88) data and +13.1
  (11.8 to 14.5) / +19.3 (17.3 to 21.4) partial. The U-Net was not trained on hopping jammers.
- True map minus no map, both with shrinkage, is -1.56 / -1.29 data and -8.27 (-9.08 to -7.51) / -13.5 (-14.8
  to -12.3) partial. The true map coarsened to PRB rows minus no map is -0.15 / -0.04 data and -1.65 /
  -6.49 partial.

With 16-QAM the "no map is as good as the true map" result of QPSK does not hold under hopping
partial-band jamming. An exact cell map is worth up to 13.5 points there, a PRB-resolution map at
most 6.5, and the U-Net map costs up to 19.3. The resolution claim holds in every case.

**8 dB SNR** (QPSK, +3 dB JSR, three seeds). Partial-band block and every pilot jammed on the
slots of `results/receiver_v7/ablation8` (canaries `true_soft` and `true_shrink` reproduced their
counts, merged into `results/res_study/merged8`), hopping on new slots (`results/res_study/hop8`).
BLER (%) below.

| Map | Estimate | partial | all pilots | hop data | hop partial |
| --- | --- | --- | --- | --- | --- |
| no protection | - | - | - | 20.3 | 32.0 |
| none | per PRB | 6.08 | 13.9 | 12.3 | 17.7 |
| none | per cell | 7.40 | 14.0 | 1.46 | 2.94 |
| none | shrinkage | 6.27 | 13.8 | 1.56 | 3.29 |
| U-Net | per PRB | 5.54 | 11.5 | 12.2 | 18.0 |
| U-Net | per cell | 6.46 | 11.5 | 3.92 | 8.94 |
| U-Net | shrinkage | 5.73 | 11.5 | 3.98 | 8.90 |
| true per PRB | shrinkage | - | - | 1.58 | 3.21 |
| true | per PRB | 5.35 | 11.3 | 1.04 | 2.23 |
| true | per cell | 6.23 | 11.3 | 1.12 | 2.38 |
| true | shrinkage | 5.54 | 11.3 | 1.08 | 2.27 |

Paired differences in points on common slots.

- Per PRB minus shrinkage without a map is +11.0 (9.81 to 12.2) under hopping data and +14.3 (12.7 to
  16.0) under hopping partial, -0.17 (-0.34 to -0.01) on a partial-band block and +0.04 with every pilot jammed.
- Per cell minus shrinkage without a map is +1.05 (0.74 to 1.39) on a partial-band block, +0.18
  (-0.11 to 0.49) with every pilot jammed, and under hopping -0.10 (-0.26 to 0.04) data and -0.35 (-0.55 to -0.18) partial.
- True map minus no map, both with shrinkage, is -0.72 (-0.99 to -0.46) on a partial-band block, -2.77
  (-3.51 to -2.07) with every pilot jammed, and under hopping -0.48 (-0.71 to -0.28) and -1.02 (-1.34 to -0.74).
- U-Net minus no map, both with shrinkage, is -0.53 (-0.82 to -0.25) on a partial-band block, -2.37 (-3.12
  to -1.65) with every pilot jammed, and under hopping +2.42 (2.00 to 2.89) and +5.69 (4.94 to 6.51).
- The data-aided U-Net receiver of §21 (`p_r_e_soft_da`) reaches 3.90% (partial) and 4.77% (every
  pilot) on the same slots.

At 8 dB the cell's own estimate is noisier and shrinkage beats it on a jammed block, while under
hopping it gives up at most 0.35 points to it. Maps help a little more than at 15 dB (up to 0.7
points on a partial-band block and 2.8 with every pilot jammed).

**Overall.** One estimate per PRB fails against a hopping jammer unless the map is
exact, in every setting tested (QPSK at 8 and 15 dB, 16-QAM at 18 dB). Shrinkage without a map is
the safe estimate. It gives up at most 1.9 points to the cell's own estimate where the jammer
moves and gains up to 12.8 where it fills its rows. With QPSK the true map is worth at most
1 point over shrinkage without a map except with every pilot jammed (0.9 to 5.1). With 16-QAM it
is worth up to 9.3 points on a block and 13.5 under hopping. The U-Net delivers part of this on
the attacks it was trained on and costs up to 19 points on the hopping jammer it never saw.

## 26. General training for every jammer, channel and modulation

Round of 25 Nov 2026 (branch `mixed-training`), run on an RTX 4070 laptop GPU. The U-Net of
section 25 was trained only on block jammers, nominal channels and QPSK, so it was tested on hopping and
16-QAM without having seen either. This round trains it on all of them and asks whether the map
then pays off.

**A bug in the EVM feature.** Until this round the EVM sliced every equalized symbol to the
nearest QPSK point. On clean 16-QAM slots at 25 dB SNR its mean was 52.8 times the noise level
instead of about 1. It now slices to the transmitted constellation (`features.nearest_point`), and
QPSK values are unchanged. The U-Net rows of the 16-QAM tables of section 25 came from a network that saw
the broken input. The tables below supersede them.

**Training.** `train_gen` / `val_gen` (120 000 / 15 000 slots) cycle through 18 conditions, every
combination of three jammers (block, hopping, pilot comb), three channel conditions (nominal,
1000 ns delay spread, 30 m/s) and two modulations (QPSK, and 16-QAM at MCS 13 with SNR 10-25 dB).
The pulsed jammer stays out as an unseen test. The fused U-Net trains for up to 80 epochs with a
learning rate that halves when the validation loss has not improved for 3 epochs, and stops after
10 epochs without improvement (`train --schedule plateau --patience 10`). The best epoch came at
10 to 16, and training stopped after 20 to 26 epochs. After about 15 epochs the training loss kept
falling while the validation loss rose. The nominal U-Net was trained again on this machine as in
`paper_run` (15 epochs, cosine). An ablation, `train_gen_short`, leaves out the long delay spreads
(12 conditions).

**Detection.** The table gives F1 / false alarms on scored cells over three seeds each
(`experiments.general_training`). The thresholds come from each training's validation split, and
the 16-QAM grids use SNR 10-25 dB.

| Test grid | Nominal | General | General, no long delays |
| --- | --- | --- | --- |
| matched | 0.950 / 1.2% | 0.954 / 0.7% | 0.955 / 0.8% |
| hopping | 0.602 / 13.8% | 0.923 / 1.5% | 0.922 / 1.8% |
| pilot comb | 0.950 / 1.2% | 0.954 / 0.7% | 0.954 / 0.8% |
| 1000 ns delay | 0.801 / 6.8% | 0.924 / 1.5% | 0.811 / 6.7% |
| 30 m/s | 0.848 / 6.0% | 0.949 / 0.8% | 0.950 / 0.9% |
| 16-QAM | 0.934 / 0.4% | 0.967 / 0.4% | 0.967 / 0.5% |
| 16-QAM hopping | 0.576 / 10.0% | 0.940 / 1.2% | 0.939 / 1.4% |
| pulsed (in no mix) | 0.633 / 10.9% | 0.821 / 3.8% | 0.799 / 4.6% |

Standard deviations over seeds are at most 0.013. With the threshold set to 1% false alarms on
each grid's own clean cells, the Pd of jammed scheduled pilot cells at -15 dB JSR is 0.58 / 0.49 /
0.57 on the matched grid and 0.77 / 0.71 / 0.76 on the 16-QAM grid, for the nominal, general and
no-long-delay U-Nets in that order.

**Reading.**
- General training improves F1 on every grid and keeps false alarms at 0.4 to 1.5% except on the
  pulsed jammer (3.8%). On the pulsed jammer, which no mix contained, F1 rises from 0.633 to 0.821.
- It costs weak-pilot sensitivity. At a matched 1% false-alarm rate the Pd at -15 dB falls from
  0.58 to 0.49. Without the long delay spreads it stays at 0.57, but F1 on those channels falls
  back to 0.811. Channels whose delays exceed `tau_max` raise the residual without a jammer, so a
  network trained on them learns to trust it less. A residual whose `tau_max` adapts to the UE
  would avoid the trade-off.

**Decoding.** These runs use `experiments.recovery` with the receivers `gen_soft` and
`gen_shrink` on TDL-C, with one decoding attempt and three detector seeds of up to 1 600 slots
each. The slots differ from those of section 25, and a receiver stops after 100 block errors with
QPSK and 200 with 16-QAM. The tables give the pooled BLER in percent
(`python -m jamloc.experiments.general_training --decoding results/gen_recovery`). The runs came
before the recovery experiment switched to the general U-Net, so today they need `--unet nominal`.

QPSK at 15 dB SNR, with the jammer on 4 of 12 PRBs at +9 dB or on every pilot of the allocation.

| Receiver | hopping data | hopping partial | every pilot +3 | every pilot +9 | block pilot | block data | block partial | no jammer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No protection | 25.5 | 59.4 | 38.2 | 93.5 | 29.0 | 84.4 | 85.7 | 0/4800 |
| No map, shrinkage | 0/4800 | 0.62 | 5.08 | 41.7 | 0.02 | 0.02 | 0.19 | 0/4800 |
| Nominal U-Net, per PRB | 22.5 | 29.5 | 4.25 | 40.4 | 0.08 | 0/4800 | 0.12 | 0/4800 |
| Nominal U-Net, shrinkage | 0.62 | 6.99 | 4.25 | 40.4 | 0.08 | 0/4800 | 0.12 | 0/4800 |
| General U-Net, per PRB | 0/4800 | 0.27 | 4.52 | 39.7 | 0/4800 | 0/4800 | 0.10 | 0/4800 |
| General U-Net, shrinkage | 0/4800 | 0.29 | 4.52 | 39.7 | 0/4800 | 0/4800 | 0.10 | 0/4800 |
| True map, shrinkage | 0/4800 | 0.25 | 4.10 | 38.4 | 0/4800 | 0/4800 | 0.10 | 0/4800 |

16-QAM (MCS 13) at 18 dB SNR, with +3 / +9 dB JSR.

| Receiver | block pilot | block data | block partial | hopping data | hopping partial |
| --- | --- | --- | --- | --- | --- |
| No protection | 55.8 / 90.1 | 94.8 / 99.5 | 96.9 / 99.7 | 76.5 / 94.0 | 87.6 / 97.3 |
| No map, shrinkage | 0.92 / 4.56 | 7.40 / 10.2 | 9.56 / 17.5 | 2.00 / 1.54 | 12.0 / 21.0 |
| Nominal U-Net, per PRB | 1.19 / 2.77 | 4.79 / 7.25 | 10.0 / 10.4 | 63.8 / 82.3 | 72.8 / 89.2 |
| Nominal U-Net, shrinkage | 1.19 / 2.77 | 5.19 / 7.25 | 10.2 / 10.5 | 34.8 / 33.0 | 62.8 / 69.1 |
| General U-Net, per PRB | 0.42 / 1.19 | 4.81 / 7.19 | 5.27 / 8.23 | 0.38 / 0.42 | 3.62 / 7.81 |
| General U-Net, shrinkage | 0.42 / 1.19 | 5.23 / 7.21 | 5.54 / 8.33 | 0.44 / 0.42 | 3.98 / 7.88 |
| True map, shrinkage | 0.33 / 1.04 | 5.19 / 7.21 | 5.35 / 8.21 | 0.38 / 0.35 | 3.48 / 7.19 |

Without a jammer, every receiver except no protection (0.04%) and no map (0.02%) decoded all 4 800
16-QAM slots.

**Paired differences** of the general U-Net with shrinkage against three receivers, in points
with 95% Newcombe intervals on common slots.
- Minus the true map with shrinkage, at most +0.19 on 16-QAM blocks, +0.06 / +0.06 and +0.50 (0.29 to
  0.73) / +0.69 (0.44 to 0.95) under 16-QAM hopping, +0.04 (-0.06 to 0.16) under QPSK hopping
  partial, +0.42 (0.21 to 0.64) and +1.08 (0.27 to 1.90, 832 slots) with every pilot jammed.
- Minus no map with shrinkage, -0.33 (-0.56 to -0.14) under QPSK hopping partial, -0.56 (-0.94 to
  -0.19) and -2.47 (-4.37 to -0.57) with every pilot jammed, -0.50 to -9.80 (-10.81 to -8.83) on
  16-QAM blocks, -1.56 / -1.12 and -8.07 (-8.89 to -7.29) / -12.95 (-14.23 to -11.72) under
  16-QAM hopping. QPSK blocks within 0.08, intervals including zero.
- Minus the nominal U-Net with shrinkage, -0.62 and -6.72 (-7.54 to -5.98) under QPSK hopping, -0.77 to
  -4.71 on 16-QAM blocks, -32.5 to -61.7 under 16-QAM hopping.

**Reading.**
- Trained on the conditions it meets, the learned map comes within 1.1 points of the true map in
  every tested case, and within 0.2 on every block attack. It beats the receiver without a map in
  every case where a map can help, by up to 13 points with 16-QAM, and costs nothing on clean
  slots.
- The nominal U-Net, which never saw hopping or 16-QAM, now fails much more under 16-QAM hopping
  (33 to 69%) than the one of section 25 (8.3 to 40%). The difference presumably comes from the
  EVM input, which was broken there and is correct here.
- With an accurate map, the per-PRB average over the flagged cells and shrinkage give the same
  BLER within 0.5 points. Shrinkage is what protects a receiver whose map is wrong.
- The general U-Net's lower weak-pilot Pd does not show in decoding, because jamming that weak
  barely hurts the link.

## 27. Paper v9 with one U-Net and more Monte Carlo runs

Round of 25 Nov 2026 (branch `mixed-training`), run on an RTX 4070 laptop GPU. Every result of
`conference-v9.tex` comes from these runs. The U-Net behind the `p_r_e_*` receivers is the general
one of section 26, which is now the default of the recovery experiment (`--unet general`, threshold
from `val_gen`). Each seed decodes until 70 block errors or 160 batches of 64 slots, so a point
pooled over three seeds rests on at least 210 block errors or 30 720 slots. All 431 receiver points
meet this rule. The commands are under "Paper v9" in `reproducing.md`. The decoding runs of the
earlier sections used the U-Net trained on the nominal mix, and `--unet nominal` repeats them.

**QPSK** on TDL-C at 15 dB SNR and +9 dB JSR unless stated. The table gives the BLER in percent
pooled over three seeds, and a zero means no block error in 30 720 slots.

| Map | Estimate | no jammer | pilot only | data only | pilot + data | pulsed data | pulsed pilot + data | hopping data | hopping pilot + data | every pilot +3 | every pilot +9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline receiver | - | 0 | 31.3 | 82.8 | 89.8 | 19.4 | 43.1 | 25.6 | 59.1 | 37.4 | 94.8 |
| none | PRB average | 0 | 0.08 | 0.10 | 0.17 | 0.04 | 0.07 | 51.0 | 57.8 | 4.87 | 43.4 |
| none | cell estimate | 0 | 0.09 | 0.12 | 0.18 | 0 | 0.04 | 0 | 0.51 | 4.89 | 43.6 |
| none | shrinkage | 0 | 0.08 | 0.10 | 0.17 | 0 | 0.04 | 0 | 0.53 | 4.87 | 43.4 |
| EVM | shrinkage | 0 | 17.0 | 1.88 | 7.85 | 0.09 | 6.31 | 0.08 | 14.9 | 10.7 | 71.1 |
| U-Net | PRB average | 0 | 0.02 | 0.08 | 0.11 | 0 | 0.02 | 0 | 0.31 | 4.48 | 39.6 |
| U-Net | shrinkage | 0 | 0.02 | 0.08 | 0.12 | 0 | 0.01 | 0 | 0.32 | 4.48 | 39.6 |
| true, PRB resolution | PRB average | 0 | 0.02 | 0.08 | 0.11 | 0.03 | 0.05 | 51.0 | 57.4 | - | - |
| true | PRB average | 0 | 0.02 | 0.08 | 0.11 | 0 | 0.01 | 0 | 0.24 | 3.92 | 38.7 |
| true | shrinkage | 0 | 0.02 | 0.08 | 0.12 | 0 | 0.01 | 0 | 0.24 | 3.92 | 38.7 |

**16-QAM** at MCS 13 and 18 dB SNR, with the BLER in percent at +3 / +9 dB JSR.

| Map | Estimate | pilot only | data only | pilot + data | hopping data | hopping pilot + data |
| --- | --- | --- | --- | --- | --- | --- |
| baseline receiver | - | 51.6 / 89.8 | 93.2 / 99.5 | 97.4 / 100 | 75.5 / 90.9 | 87.5 / 98.7 |
| none | PRB average | 1.01 / 4.19 | 6.08 / 9.64 | 8.65 / 16.0 | 84.9 / 97.9 | 88.8 / 99.2 |
| none | cell estimate | 1.56 / 5.61 | 20.0 / 19.9 | 19.9 / 27.2 | 1.86 / 1.53 | 10.7 / 16.7 |
| none | shrinkage | 1.05 / 4.18 | 6.75 / 9.97 | 9.04 / 16.4 | 2.08 / 1.76 | 12.4 / 18.4 |
| U-Net | PRB average | 0.50 / 1.36 | 4.23 / 7.03 | 5.09 / 7.78 | 0.37 / 0.46 | 3.76 / 7.06 |
| U-Net | shrinkage | 0.50 / 1.36 | 4.46 / 7.25 | 5.25 / 7.78 | 0.43 / 0.49 | 3.79 / 7.18 |
| true | PRB average | 0.44 / 1.11 | 4.22 / 7.00 | 4.94 / 7.74 | 0.32 / 0.42 | 2.98 / 6.28 |
| true | cell estimate | 0.44 / 1.11 | 7.23 / 7.81 | 7.19 / 8.49 | 0.53 / 0.49 | 3.79 / 6.49 |
| true | shrinkage | 0.44 / 1.11 | 4.44 / 7.21 | 5.07 / 7.67 | 0.36 / 0.44 | 3.40 / 6.28 |

**Paired differences** in percentage points, with 95% Newcombe intervals on common slots.
- No map, PRB average minus shrinkage under hopping, 51.0 (46.6 to 55.3) data only and 57.6 (52.9
  to 62.1) pilot + data. The PRB average is 24.4 (19.8 to 28.8) worse than the baseline receiver
  under data-only hopping.
- PRB-resolution true map minus true map, both with the PRB average, under hopping, 51.0 (46.6 to
  55.3) and 57.1 (52.4 to 61.6).
- No map minus true map, both with shrinkage, under hopping pilot + data, 0.29 (0.23 to 0.36).
- U-Net map minus no map, both with shrinkage. With QPSK and some clean pilots between -0.22 (-0.29
  to -0.15) and 0.00. With every pilot jammed -0.54 (-0.93 to -0.15) at +3 dB and -3.82 (-5.80 to
  -1.83) at +9 dB. With 16-QAM -0.51 to -8.41 (-9.98 to -6.94) on a block and -1.31 to -12.58
  (-14.54 to -10.75) under hopping.
- U-Net map minus true map, both with shrinkage. At most +0.07 with QPSK and some clean pilots,
  +0.44 (0.23 to 0.67) and +0.87 (-0.10 to 1.84) with every pilot jammed, at most +0.73 (0.42 to
  1.08) with 16-QAM.
- EVM map minus no map, both with shrinkage, +16.8 (14.9 to 18.9) under pilot-only jamming and
  +28.7 (23.8 to 33.2) with every pilot jammed at +9 dB.
- With 16-QAM, the cell estimate minus shrinkage without a map is +9.81 (8.12 to 11.60) under
  data-only jamming of a block at +9 dB and -1.48 (-2.38 to -0.61) under hopping pilot + data.

**BLER against JSR** (-9 to +9 dB, `results/v9/resolution_sweep.pdf`). Under data-only hopping,
shrinkage with or without a map failed on at most 2 of 30 720 slots at every JSR, while the PRB
average without a map was worse than the baseline receiver from +6 dB on (48.4% against 29.2% at
+9 dB). Under hopping pilot + data jamming, shrinkage stayed at or below 0.51% without a map and
0.29% with the U-Net map. With a block jammer, every receiver that estimates the jamming power stayed
at or below 0.17% at every JSR, except the one with the EVM map (up to 17.8% under pilot-only
jamming).

**Data-aided passes** (`p_r_e_shrink_full`, three passes after a failed CRC). With every pilot
jammed at +9 dB, the BLER of the U-Net receiver falls from 39.8% to 17.8%, by 21.7 percentage points
(18.3 to 25.1). Under hopping pilot + data jamming it falls from 0.32% to 0.09%.

**Reading.**
- The resolution of the jamming estimate must come from the map or from the estimate. With an exact
  map the PRB average works, with a PRB-resolution map or without a map it fails under hopping, and
  shrinkage without a map comes within 0.3 points of the true map.
- Trained on the conditions it meets, the U-Net map is close to the true map everywhere tested and
  pays off where the cell's own energy is ambiguous, with 16-QAM and when every pilot is jammed.
- The EVM map raises the BLER because it marks the damaged data symbols and never the jammed pilots.

## 28. Paper v10 with 16-QAM curves and clearer method figures

Round of 25 Nov 2026 (branch `paper-v10`), run on an Apple M1 Pro (10 cores, no GPU) after the
Colab L4 and A100 runtimes were unavailable. The commands are under "Paper v10" in
`reproducing.md`. `docs/plan-v10.md` lists what this round answers.

**Method figures.** The link output now keeps the jamming signal at the gNB (`SlotBatch.y_jam`),
so the figures can show the jamming power that reached each cell, in units of the jammer's power
per resource element. The shrinkage example (`figures.shrinkage_example`) marks this power next to
the cell estimate, the PRB average and the shrinkage estimate. The attack-map figure
(`figures.attack_maps`) now shows three attacks (a block on the pilots, the hopping jammer and the
pulsed jammer on pilots and data, +6 dB JSR, 15 dB SNR). Next to the U-Net and EVM maps it shows
the jamming power that the receiver assigns without a map, from shrinkage on the data cells and
from the pilot residual on the DMRS cells. A pilot noise variance sigma^2 R_b corresponds to the
jamming power (R_b - 1) N0 on that pilot, since sigma^2 = N0 / E_p and jamming of power S adds
S / E_p to the variance of an LS value. On the example slot the U-Net misses no jammed cell in the
three attacks, while the EVM threshold marks 28 clean data cells under pilot jamming.

**16-QAM against JSR** (MCS 13, 18 dB SNR, TDL-C, pilots and data jammed, general U-Net, three
seeds, at least 210 block errors or 30 720 slots per point). BLER in percent.

Block of 4 PRBs on pilots and data.

| Receiver | -9 dB | -6 dB | -3 dB | +0 dB | +3 dB | +6 dB | +9 dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline receiver | 18.3 | 45.8 | 70.1 | 80.7 | 97.4 | 98.7 | 99.7 |
| no map, PRB average | 1.51 | 2.86 | 4.29 | 5.83 | 8.84 | 11.4 | 17.9 |
| no map, shrinkage | 1.69 | 3.35 | 4.89 | 6.46 | 9.64 | 11.7 | 18.6 |
| EVM map, shrinkage | 13.7 | 42.5 | 63.6 | 75.3 | 84.9 | 74.7 | 72.1 |
| U-Net map, shrinkage | 0.91 | 2.04 | 2.91 | 4.09 | 5.95 | 6.89 | 8.36 |
| true map, shrinkage | 0.85 | 1.83 | 2.87 | 4.04 | 5.77 | 6.59 | 8.26 |

Hopping jammer on pilots and data.

| Receiver | -9 dB | -6 dB | -3 dB | +0 dB | +3 dB | +6 dB | +9 dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline receiver | 3.4 | 15.2 | 40.5 | 66.4 | 87.2 | 96.1 | 97.9 |
| no map, PRB average | 2.12 | 10.4 | 30.2 | 60.7 | 86.7 | 98.2 | 99.7 |
| no map, shrinkage | 0.86 | 2.64 | 5.19 | 8.36 | 11.7 | 15.8 | 19.7 |
| EVM map, shrinkage | 2.14 | 11.2 | 28.4 | 37.5 | 46.5 | 49.4 | 56.5 |
| U-Net map, shrinkage | 0.24 | 0.52 | 1.09 | 2.1 | 3.45 | 5.49 | 7.37 |
| true map, shrinkage | 0.19 | 0.42 | 0.87 | 1.75 | 2.92 | 4.67 | 6.62 |

Paired differences in percentage points on common slots, with 95% Newcombe intervals.

- Block, U-Net map minus no map, both with shrinkage. -9 dB -0.78 (-0.95 to -0.63), -6 dB -1.36 (-1.69 to -1.05), -3 dB -1.81 (-2.26 to -1.40), +0 dB -2.24 (-2.82 to -1.70), +3 dB -3.26 (-4.09 to -2.49), +6 dB -4.64 (-5.68 to -3.66), +9 dB -10.86 (-12.71 to -9.11).
- Block, U-Net map minus true map. -9 dB +0.04 (+0.01 to +0.08), -6 dB +0.18 (+0.09 to +0.28), -3 dB +0.07 (-0.02 to +0.16), +0 dB +0.08 (-0.03 to +0.19), +3 dB +0.22 (+0.04 to +0.41), +6 dB +0.29 (+0.07 to +0.52), +9 dB +0.20 (-0.02 to +0.43).
- Block, PRB average minus shrinkage, no map. -9 dB -0.19 (-0.29 to -0.09), -6 dB -0.47 (-0.68 to -0.28), -3 dB -0.57 (-0.86 to -0.29), +0 dB -0.80 (-1.16 to -0.46), +3 dB -0.67 (-1.08 to -0.29), +6 dB -0.36 (-0.70 to -0.05), +9 dB -0.66 (-1.28 to -0.05).
- Hopping, U-Net map minus no map, both with shrinkage. -9 dB -0.64 (-0.75 to -0.55), -6 dB -2.12 (-2.46 to -1.83), -3 dB -4.21 (-4.86 to -3.62), +0 dB -6.45 (-7.48 to -5.50), +3 dB -7.97 (-9.28 to -6.76), +6 dB -10.33 (-12.01 to -8.75), +9 dB -12.59 (-14.59 to -10.70).
- Hopping, U-Net map minus true map. -9 dB +0.05 (+0.02 to +0.08), -6 dB +0.10 (+0.06 to +0.15), -3 dB +0.19 (+0.12 to +0.26), +0 dB +0.34 (+0.22 to +0.47), +3 dB +0.54 (+0.34 to +0.75), +6 dB +0.76 (+0.48 to +1.06), +9 dB +0.68 (+0.37 to +1.02).
- Hopping, PRB average minus shrinkage, no map. -9 dB +1.34 (+1.12 to +1.59), -6 dB +7.43 (+6.35 to +8.63), -3 dB +23.44 (+20.45 to +26.52), +0 dB +49.74 (+44.50 to +54.54), +3 dB +75.52 (+70.69 to +79.37), +6 dB +83.85 (+79.61 to +87.12), +9 dB +81.77 (+77.45 to +85.29).

**Reading.**

- Every point meets the Monte Carlo rule of section 27. Pooled over the seeds, each rests on at
  least 210 block errors or 30 720 slots.
- With 16-QAM the U-Net map pays off at every JSR, by 0.6 to 12.6 percentage points, and the gain
  grows with the JSR. It stays within 0.8 points of the true map.
- On a block, the PRB average without a map is 0.2 to 0.8 points better than shrinkage. Under the
  hopping jammer it fails on up to 99.7% of the blocks, and at +9 dB more often than the baseline
  receiver (97.9%).
- The EVM map, whose threshold comes from the QPSK split `val`, fails on 14% to 85% of the blocks
  under the block jammer and on 2% to 57% under the hopping jammer. As with QPSK, it never marks
  the jammed pilots.

**Paper v10** (`conference-v10.tex` on Overleaf, seven pages). It adds a TikZ diagram of the link
and the receiver (Fig. 1), presents the method in the receiver's order (pilot weights, excess power,
shrinkage), says why the estimate is pooled along the PRB and not along the symbol, and
explains that the estimate locates the jammer softly. A new subsection describes the U-Net and
gives its detection F1 on each test grid (section 26), with a figure of the network at work on one
hopping slot (`figures.unet_process`, its features, blocks drawn to the size of their grid and
channel count, and the score and map). The other method figures are the ones above.
The subsection on further decoding attempts was left out while the paper had six pages. It stays in v9 and in
the long version. An independent review checked all 91 table entries and the text against the
result files and found no mismatch. Its points on wording and precision were applied.
On 26 Nov the term "unexplained energy" became "excess power" throughout the paper.
The same day Fig. 1 got step numbers in place of equation numbers, the attack-map figure became
Fig. 2 because the text cites it first, and every float was moved next to the text that cites it.

## 29. Which inputs and which network

Round of 26 Nov 2026 (branch `input-ablation`), run on a Colab L4 (12 vCPUs). The commands are
under "Inputs and architecture" in `reproducing.md`.

**Questions.** Does the EVM input still help once the pilot residual is there? The EVM rises on
the data cells of a PRB whose pilots are jammed, and on fast or long-delay channels without any
jammer. And is the U-Net's pooling path needed, or would any CNN of the same size do?

**Detectors.** Two ablations of the general U-Net of section 26, trained on `train_gen` in the
same way (up to 80 epochs, learning rate halved after a plateau, early stopping after 10 epochs,
thresholds from `val_gen`, three seeds). *U-Net P+R* is the same U-Net without the EVM channel
(66 445 parameters). *ResNet P+R+E* (`model.SlotResNet`) keeps the full 24 x 14 grid. One 3 x 3
convolution lifts the inputs to 30 channels, four residual blocks of two 3 x 3 convolutions follow,
and a 1 x 1 convolution gives the logits (66 991 parameters, against 66 553 for the U-Net). Each
cell sees the 19 x 19 cells around it, whereas the U-Net's pooling lets every cell see the whole
slot. The best validation loss was 0.117 to 0.120 for the U-Net, 0.123 to 0.124 for the ResNet and
0.210 to 0.213 without the EVM. The training split and the test grids were generated again on the
L4, and the general U-Net's F1 on these grids matches section 26 to within 0.001.

**Detection.** F1 / false alarms on scored cells, mean over three seeds (standard deviations at most
0.003).

| Test grid | U-Net P+R+E | U-Net P+R | ResNet P+R+E |
| --- | --- | --- | --- |
| matched | 0.954 / 0.7% | 0.922 / 0.5% | 0.957 / 0.5% |
| hopping | 0.923 / 1.5% | 0.867 / 1.3% | 0.925 / 1.2% |
| pilot comb | 0.954 / 0.7% | 0.923 / 0.6% | 0.956 / 0.5% |
| 1000 ns delay | 0.924 / 1.5% | 0.912 / 0.9% | 0.928 / 1.4% |
| 30 m/s | 0.949 / 0.8% | 0.920 / 0.6% | 0.952 / 0.7% |
| 16-QAM | 0.966 / 0.4% | 0.903 / 0.8% | 0.963 / 0.6% |
| 16-QAM hopping | 0.940 / 1.2% | 0.859 / 1.8% | 0.937 / 1.3% |
| pulsed (in no mix) | 0.822 / 3.8% | 0.794 / 2.8% | **0.907 / 1.3%** |

F1 per attack type, and the Pd of jammed scheduled pilot cells at -15 dB JSR with the threshold
that gives 1% false alarms on each grid's own clean cells.

| Detector | Grid | F1 pilot | F1 data | F1 partial | Pilot Pd at -15 dB, 1% FA |
| --- | --- | --- | --- | --- | --- |
| U-Net P+R+E | matched | 0.871 | 0.957 | 0.960 | 0.50 |
| U-Net P+R | matched | 0.904 | 0.915 | 0.930 | 0.54 |
| ResNet P+R+E | matched | 0.879 | 0.959 | 0.962 | 0.57 |
| U-Net P+R+E | 16-QAM | 0.911 | 0.966 | 0.972 | 0.71 |
| U-Net P+R | 16-QAM | 0.851 | 0.885 | 0.925 | 0.64 |
| ResNet P+R+E | 16-QAM | 0.795 | 0.969 | 0.973 | 0.72 |

**Decoding.** Each map is given to the shrinkage receiver on the same fresh slots (TDL-C, QPSK at
15 dB SNR or 16-QAM MCS 13 at 18 dB, the cases of Tables II and III of paper v10). Each seed runs
until 70 block errors or 160 batches of 64 slots, and every point has at least 210 block errors or
30 720 slots. BLER in percent, and paired differences in percentage points with 95% intervals.

| Case | Point | No map | U-Net P+R+E | U-Net P+R | ResNet P+R+E | True map | U-Net P+R - U-Net P+R+E | ResNet P+R+E - U-Net P+R+E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QPSK block | data +9 dB | 0.08 | 0.07 | 0.11 | 0.07 | 0.07 | +0.04 (+0.02 to +0.07) | -0.00 (-0.01 to +0.01) |
| QPSK block | no jammer | 0/30720 | 0/30720 | 0/30720 | 0/30720 | 0/30720 | -0.00 (-0.01 to +0.01) | -0.00 (-0.01 to +0.01) |
| QPSK block | partial +9 dB | 0.13 | 0.08 | 0.15 | 0.09 | 0.08 | +0.07 (+0.04 to +0.10) | +0.00 (-0.01 to +0.02) |
| QPSK block | pilot +9 dB | 0.06 | 0.03 | 0.02 | 0.02 | 0.01 | -0.01 (-0.03 to +0.00) | -0.01 (-0.03 to +0.00) |
| QPSK hopping | data +9 dB | 0/30720 | 0/30720 | 0/30720 | 0/30720 | 0/30720 | -0.00 (-0.01 to +0.01) | -0.00 (-0.01 to +0.01) |
| QPSK hopping | partial +9 dB | 0.42 | 0.25 | 0.25 | 0.20 | 0.18 | -0.00 (-0.04 to +0.04) | -0.05 (-0.08 to -0.02) |
| QPSK every pilot | every pilot +3 dB | 5.06 | 4.23 | 4.33 | 3.90 | 3.79 | +0.08 (-0.14 to +0.31) | -0.34 (-0.55 to -0.13) |
| QPSK every pilot | every pilot +9 dB | 43.23 | 38.72 | 37.85 | 37.67 | 37.50 | -0.87 (-1.84 to +0.10) | -1.04 (-1.95 to -0.14) |
| 16-QAM block | data +9 dB | 9.51 | 6.70 | 8.44 | 6.61 | 6.58 | +1.95 (+1.41 to +2.54) | -0.09 (-0.25 to +0.06) |
| 16-QAM block | no jammer | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | -0.00 (-0.01 to +0.01) | -0.00 (-0.01 to +0.01) |
| 16-QAM block | partial +9 dB | 16.52 | 8.40 | 11.04 | 8.40 | 8.52 | +2.77 (+2.02 to +3.58) | -0.00 (-0.24 to +0.24) |
| 16-QAM block | pilot +9 dB | 4.09 | 1.22 | 1.17 | 1.09 | 1.02 | -0.05 (-0.13 to +0.03) | -0.13 (-0.21 to -0.06) |
| 16-QAM hopping | data +9 dB | 1.99 | 0.46 | 1.03 | 0.43 | 0.41 | +0.53 (+0.43 to +0.64) | -0.03 (-0.06 to -0.01) |
| 16-QAM hopping | partial +9 dB | 22.06 | 7.57 | 9.20 | 7.11 | 6.53 | +1.48 (+0.76 to +2.22) | -0.45 (-0.86 to -0.06) |

**Reading.**
- **The EVM helps.** Without it F1 falls on every grid, most with 16-QAM (0.966 to 0.903) and
  under hopping (0.923 to 0.867, and 0.940 to 0.859 with 16-QAM). On the matched grid, data-jamming
  F1 falls from 0.957 to 0.915. The EVM costs a little on pilot attacks with QPSK. Without it the
  network reaches pilot F1 0.904 instead of 0.871, because the raised EVM of the data cells of a
  PRB whose pilots are jammed looks like data jamming (section 7). With 16-QAM the EVM helps pilot
  attacks too (0.911 against 0.851).
- **In decoding the map without the EVM is worse wherever data are jammed.** The gap is 1.95
  percentage points (1.41 to 2.54) under data jamming of a 16-QAM block, 2.77 (2.02 to 3.58) under
  pilot + data jamming, and 1.48 (0.76 to 2.22) under 16-QAM hopping on pilots and data. It is
  never better by a significant margin. Its gain with every pilot jammed at +9 dB, 0.87 (-0.10 to
  1.84), lies within the interval.
- **The U-Net's pooling is not needed.** The ResNet matches it on every grid of the training mix
  (F1 within 0.004) and does much better on the pulsed jammer, which no mix contained (0.907 with
  1.3% false alarms against 0.822 with 3.8%). Its seeds also agree more closely. Its one weakness
  is pilot jamming with 16-QAM (pilot F1 0.795 against 0.911). The DMRS is symbol 2, so the data
  cells of the last symbols lie beyond its 19 cells and cannot tell a jammed pilot from jammed data.
  These marks do not cost BLER. The excess power of those cells is real, since it comes from
  the wrong channel estimate, and shrinkage probably accounts for it.
- **In decoding the ResNet's map is as good as the U-Net's or better.** It lowers the BLER by 1.04
  percentage points (0.14 to 1.95) with every pilot jammed at +9 dB, by 0.34 (0.13 to 0.55) at +3
  dB, and by 0.45 (0.06 to 0.86) under 16-QAM hopping on pilots and data. It is never worse by a
  significant margin.
- **For the paper.** The evidence supports a convolutional network that sees its neighbours, not
  the U-Net in particular. The per-cell network of section 11 shows that context is what matters
  (F1 0.694 against 0.951 with nominal training). The U-Net can stay as the detector with this
  ablation as its justification, or the ResNet can replace it, which means running the decoding
  behind every table and figure again.

## 30. One U-Net for every configuration (paper v11)

Round of 26-27 Nov 2026 (branch `paper-v11`). The author asked for two DMRS symbols, 64-QAM,
fewer tables, smoother BLER curves and the true map without shrinkage (`plan-v11.md`). The
commands are under "Paper v11" in `reproducing.md`. Training and most decoding ran on a Colab L4
(12 vCPUs), the rest on the Mac.

**Training.** One U-Net (66 553 parameters, as before) on `train_all`, 240 000 slots cycling
through 54 conditions, the 18 of the general training times three modulations (QPSK, 16-QAM MCS 13,
64-QAM MCS 17 at 16-30 dB SNR) times one or two DMRS symbols. Same schedule as section 26 (plateau
halving, early stopping after 10 epochs). The three seeds stopped after 27 or 28 epochs at a
validation loss of 0.097 to 0.098 (`results/one_unet/logs`). One threshold from `val_all`.

**Detection.** F1 / false alarms on scored cells, mean of three seeds, v10 U-Net (`train_gen`,
threshold from `val_gen`) against the v11 one, on test grids with one and two DMRS symbols
(`results/one_unet/one_unet.csv`, `python -m jamloc.experiments.one_unet --table-only`).

| Grid | v10 U-Net, 1 DMRS | v11 U-Net, 1 DMRS | v10 U-Net, 2 DMRS | v11 U-Net, 2 DMRS |
| --- | --- | --- | --- | --- |
| QPSK block | 0.954 / 0.7% | 0.951 / 0.9% | 0.927 / 1.8% | 0.954 / 0.9% |
| QPSK hopping | 0.923 / 1.5% | 0.921 / 1.8% | 0.908 / 1.9% | 0.919 / 1.8% |
| QPSK pilot comb | 0.954 / 0.7% | 0.951 / 0.9% | 0.920 / 1.8% | 0.949 / 1.0% |
| QPSK 1000 ns | 0.924 / 1.5% | 0.919 / 1.7% | 0.873 / 3.9% | 0.914 / 2.0% |
| QPSK 30 m/s | 0.949 / 0.8% | 0.946 / 1.0% | 0.924 / 1.8% | 0.951 / 0.9% |
| 16-QAM block | 0.966 / 0.4% | 0.964 / 0.6% | 0.946 / 1.1% | 0.969 / 0.5% |
| 16-QAM hopping | 0.940 / 1.2% | 0.935 / 1.6% | 0.933 / 1.3% | 0.942 / 1.4% |
| 64-QAM block | 0.952 / 0.1% | 0.979 / 0.3% | 0.931 / 0.8% | 0.981 / 0.3% |
| 64-QAM hopping | 0.947 / 0.4% | 0.959 / 0.9% | 0.934 / 0.7% | 0.962 / 0.8% |
| QPSK pulsed (unseen) | 0.822 / 3.8% | 0.774 / 5.6% | 0.814 / 4.0% | 0.785 / 5.1% |

| Grid | Detector | DMRS | F1 pilot | Pilot Pd at -15 dB | Same at 1% FA |
| --- | --- | --- | --- | --- | --- |
| QPSK block | v10 U-Net | 1 | 0.871 | 0.451 | 0.495 |
| QPSK block | v11 U-Net | 1 | 0.861 | 0.501 | 0.519 |
| QPSK block | v10 U-Net | 2 | 0.758 | 0.295 | 0.226 |
| QPSK block | v11 U-Net | 2 | 0.905 | 0.575 | 0.586 |
| 16-QAM block | v10 U-Net | 1 | 0.911 | 0.624 | 0.712 |
| 16-QAM block | v11 U-Net | 1 | 0.877 | 0.662 | 0.717 |
| 16-QAM block | v10 U-Net | 2 | 0.823 | 0.402 | 0.383 |
| 16-QAM block | v11 U-Net | 2 | 0.933 | 0.725 | 0.788 |
| 64-QAM block | v10 U-Net | 1 | 0.968 | 0.870 | 0.958 |
| 64-QAM block | v11 U-Net | 1 | 0.927 | 0.877 | 0.931 |
| 64-QAM block | v10 U-Net | 2 | 0.875 | 0.641 | 0.683 |
| 64-QAM block | v11 U-Net | 2 | 0.972 | 0.900 | 0.962 |

Reading of the detection results.
- **Two DMRS symbols need training.** The v10 U-Net, which never saw a second DMRS symbol, falls to
  pilot F1 0.758 on the QPSK block grid with two DMRS symbols (0.905 for the v11 one) and doubles
  its false alarms (1.8% against 0.9%). Its F1 over all attacks drops from 0.954 to 0.927.
- **Training on every configuration costs little.** With one DMRS symbol and QPSK or 16-QAM the
  v11 U-Net is within 0.005 F1 of the v10 one and marks slightly more clean cells (0.9% against
  0.7% on the QPSK block grid). With 64-QAM it gains (0.979 against 0.952 on the block grid),
  although its pilot F1 there is lower (0.927 against 0.968).
- **The unseen pulsed jammer suffers.** F1 0.774 against 0.822 with 5.6% false alarms against
  3.8%. The ResNet of section 29 did much better on this grid (0.907), so this is the main cost of
  keeping the U-Net.

**Operating points.** QPSK at 15 dB made the receivers that estimate the jamming power fail on 0 to
3 of 30 720 blocks, so the v10 curves followed chance and zero-error points sat at the 95% upper
bound of 1.2e-4. A first run at 10 dB still gave 0 to 5 errors per seed and was stopped. Each
modulation now runs where the baseline receiver fails about 0.8% of the blocks without a jammer on
random allocations. That is QPSK at 8 dB, 16-QAM at 15 dB and 64-QAM (MCS 17) at 18 dB
(`plan-v11.md` has the no-jammer sweep). With the fixed 12-PRB allocation, the receivers with the
U-Net map or the true map fail on 0.26% (QPSK), 0.26% (16-QAM) and 0.16% (64-QAM) of the blocks
without a jammer, so every point of the new figure rests on tens to hundreds of errors.

**Decoding at +9 dB JSR.** BLER in percent, three seeds pooled (`python -m
jamloc.experiments.paper_v11 numbers results/v11`). A block covers 4 PRBs inside the allocation,
hop is the hopping block, and every pilot is jammed at +3 and +9 dB.

| Case | Baseline | No map, PRB avg. | No map, cell | No map, shrinkage | EVM map | U-Net map | True map, PRB avg. | True map | PRB-res. true map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QPSK, no jammer | 0.57 | 0.31 | 0.32 | 0.30 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| QPSK block, pilot | 47.1 | 3.59 | 3.81 | 3.55 | 20.0 | 2.65 | 2.62 | 2.62 | 2.62 |
| QPSK block, data | 86.5 | 9.12 | 10.2 | 9.20 | 66.1 | 8.40 | 8.16 | 8.19 | 8.16 |
| QPSK block, pilot + data | 93.8 | 11.4 | 13.0 | 11.4 | 83.9 | 10.0 | 9.82 | 9.96 | 9.82 |
| QPSK hop, data | 62.5 | 60.9 | 2.25 | 2.43 | 26.2 | 1.87 | 1.82 | 1.84 | 60.9 |
| QPSK hop, pilot + data | 78.6 | 66.4 | 7.63 | 7.96 | 48.4 | 6.65 | 6.40 | 6.46 | 65.6 |
| QPSK pulsed, data | 50.4 | 5.95 | 1.95 | 1.83 | 19.7 | 1.76 | 1.52 | 1.59 | 5.55 |
| QPSK pulsed, pilot + data | 63.5 | 8.85 | 4.34 | 4.02 | 35.9 | 3.83 | 3.08 | 3.11 | 7.57 |
| QPSK every pilot +3 | 41.1 | 14.4 | 14.4 | 14.3 | 15.8 | 12.1 | 11.9 | 11.9 | - |
| QPSK every pilot +9 | 93.2 | 56.5 | 56.5 | 56.8 | 66.7 | 50.7 | 50.9 | 50.9 | - |
| 16-QAM, no jammer | 0.59 | 0.44 | 0.72 | 0.45 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| 16-QAM block, pilot | 94.8 | 11.0 | 13.5 | 11.1 | 76.8 | 5.22 | 5.17 | 5.17 | 5.17 |
| 16-QAM block, data | 99.5 | 28.4 | 41.0 | 29.2 | 64.8 | 23.8 | 23.0 | 23.7 | 23.0 |
| 16-QAM block, pilot + data | 99.7 | 39.8 | 52.1 | 40.3 | 93.0 | 26.9 | 26.3 | 26.5 | 26.3 |
| 16-QAM hop, data | 96.9 | 99.5 | 9.87 | 9.97 | 30.3 | 4.85 | 4.64 | 4.71 | 99.2 |
| 16-QAM hop, pilot + data | 99.5 | 99.5 | 33.2 | 34.2 | 73.4 | 17.7 | 16.6 | 16.8 | 99.2 |
| 64-QAM, no jammer | 0.56 | 0.56 | 1.19 | 0.56 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 |
| 64-QAM block, pilot | 96.9 | 20.2 | 28.1 | 20.1 | 87.2 | 4.71 | 4.56 | 4.56 | 4.56 |
| 64-QAM block, data | 99.7 | 35.8 | 60.4 | 36.9 | 55.6 | 20.9 | 20.3 | 20.5 | 20.3 |
| 64-QAM block, pilot + data | 100 | 54.0 | 78.4 | 55.8 | 95.3 | 22.4 | 21.6 | 22.0 | 21.6 |
| 64-QAM hop, data | 98.4 | 99.7 | 14.7 | 17.2 | 21.0 | 3.48 | 3.23 | 3.31 | 99.7 |
| 64-QAM hop, pilot + data | 99.5 | 99.5 | 50.7 | 52.7 | 73.4 | 17.7 | 15.4 | 15.8 | 99.0 |
| QPSK two DMRS, no jammer | 0.44 | 0.14 | 0.15 | 0.14 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| QPSK two DMRS block, pilot | 35.1 | 1.41 | 1.53 | 1.40 | 7.96 | 1.25 | 1.23 | 1.23 | 1.23 |
| QPSK two DMRS block, data | 82.0 | 5.92 | 6.76 | 6.13 | 63.5 | 5.41 | 5.42 | 5.41 | 5.42 |
| QPSK two DMRS block, pilot + data | 92.4 | 7.43 | 8.31 | 7.56 | 79.7 | 6.91 | 6.84 | 6.88 | 6.84 |

Paired differences quoted in the paper, in percentage points with their 95% intervals, all at
+9 dB unless stated.

- U-Net map minus shrinkage without a map. QPSK block pilot -0.81 (-1.08 to -0.55), block pilot +
  data -1.61 (-2.28 to -0.99), hop pilot + data -1.34 (-1.86 to -0.87), every pilot +3 dB -2.15
  (-3.06 to -1.27), +9 dB -7.81 (-10.69 to -4.89), pulsed data -0.14 (-0.25 to -0.03). 16-QAM
  block pilot + data -11.88 (-14.50 to -9.24), hop -16.90 (-19.76 to -14.07). 64-QAM block -34.38
  (-38.66 to -29.84), hop -36.52 (-40.61 to -32.24). Two DMRS block pilot -0.21 (-0.30 to -0.11),
  block pilot + data -0.57 (-0.90 to -0.26), every pilot (sweep run) -2.99 (-4.65 to -1.35).
- U-Net map minus true map. At most +0.15 with QPSK, +0.56 (0.36 to 0.78) under pulsed pilot +
  data, +0.89 (0.31 to 1.50) under 16-QAM hop, +1.72 (0.99 to 2.49) under 64-QAM hop. In the sweeps
  up to +2.9 (64-QAM hop at 0 dB).
- Shrinkage minus cell estimate without a map. Block data -0.99 (-1.47 to -0.56) with QPSK, -12.15
  (-14.83 to -9.46) with 16-QAM, -24.22 (-28.41 to -19.82) with 64-QAM. Hop pilot + data +0.25
  (0.03 to 0.49) with QPSK, +1.56 (0.38 to 2.75) with 16-QAM, +1.95 (0.13 to 3.77) with 64-QAM.
- True map with the PRB average minus true map with shrinkage. QPSK hop pilot + data -0.06 (-0.21
  to 0.08), 16-QAM block data -0.68 (-1.27 to -0.12), 64-QAM hop pilot + data -0.43 (-0.85 to
  -0.02). In the sweeps up to 1.2 points lower with 16-QAM and 64-QAM.
- EVM map minus shrinkage without a map. QPSK block pilot +16.84 (14.74 to 19.09), block pilot +
  data +71.88 (66.89 to 75.94).
- Shrinkage without a map minus the U-Net map without a jammer. +0.19 (0.14 to 0.24) with 16-QAM,
  +0.40 (0.33 to 0.47) with 64-QAM.

**BLER against JSR** (`results/v11/resolution_sweep.pdf`, Fig. 5 of v11). Eleven panels in two
rows. QPSK at 8 dB has (a) a block on the pilots, (b) hopping data, (c) hopping pilots + data and
(d) every pilot. 16-QAM at 15 dB has (e) a block and (f) hopping, both on pilots + data. 64-QAM at
18 dB has (g) a block and (h) hopping. QPSK with two DMRS symbols has (i) a block on the pilots,
(j) hopping pilots + data and (k) every pilot. The receivers are those of v10 plus the true map
with a plain PRB average over its marked cells (`true_soft`). Every point has at least tens of
block errors, so no zero-error triangles remain. The two-DMRS block sweep ran on the L4 after the
Mac proved too slow (about 28 minutes per point with seven processes). The two-DMRS hopping sweep
ran on both. The Mac's copy is used, and the L4 copy, on other slots, agrees within its intervals
(2.21% against 2.17% for shrinkage without a map at +9 dB, 1.72% against 1.73% for the U-Net map).

Reading of the decoding results.
- **The new operating points fix Fig. 5.** The curves are smooth, and the bands are narrow enough
  to separate the receivers that the v10 figure could not.
- **A map now helps with QPSK too.** At 15 dB the U-Net map was within 0.22 points of shrinkage
  without a map. At 8 dB it is 0.8 to 1.6 points better at +9 dB and 2.2 and 7.8 points better
  with every pilot jammed, within 0.15 points of the true map (0.56 under the unseen pulsed
  jammer).
- **64-QAM is where the map matters most.** Under pilot + data jamming at +9 dB, shrinkage without
  a map fails on 52.7% to 55.8% of the blocks, the U-Net map on 17.7% to 22.4% and the true map on
  15.8% to 22.0%. Without a
  jammer, shrinkage without a map costs 0.40 points, because the varying symbol energy of 64-QAM
  looks like jamming, and the U-Net map costs nothing.
- **The true map does not need shrinkage.** A plain average over the marked cells is as good with
  QPSK and up to 1.2 points better with 16-QAM and 64-QAM. With a perfect map the marked cells of a
  row share the same jamming power, so the cell-level part of shrinkage only adds noise.
- **A second DMRS symbol makes pilot jamming less harmful** at the same JSR per resource element
  (baseline 35.1% instead of 47.1%, shrinkage 1.40% instead of 3.55% at +9 dB) and makes every
  estimate better with every pilot jammed (30.7% instead of 51.8% for shrinkage at +9 dB). The
  U-Net map still lowers the BLER, by 0.2 to 3.0 points, within 0.6 points of the true map.

Two-DMRS sweeps (BLER in percent, one row per JSR from -9 to +9 dB).

| Case | JSR | Baseline | No map, PRB avg. | No map, shrinkage | EVM map | U-Net map | True map, PRB avg. | True map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block, pilot | -9 | 0.62 | 0.16 | 0.16 | 0.15 | 0.15 | 0.15 | 0.15 |
| block, pilot | 0 | 2.54 | 0.30 | 0.30 | 0.39 | 0.27 | 0.27 | 0.27 |
| block, pilot | +9 | 36.1 | 1.66 | 1.65 | 8.87 | 1.33 | 1.33 | 1.33 |
| hop, pilot + data | -9 | 0.83 | 0.29 | 0.28 | 0.27 | 0.27 | 0.23 | 0.23 |
| hop, pilot + data | 0 | 11.4 | 3.71 | 0.83 | 6.72 | 0.63 | 0.57 | 0.57 |
| hop, pilot + data | +9 | 74.5 | 58.6 | 2.21 | 36.4 | 1.72 | 1.64 | 1.66 |
| every pilot | -9 | 1.12 | 0.23 | 0.23 | 0.19 | 0.20 | 0.21 | 0.21 |
| every pilot | 0 | 10.9 | 1.54 | 1.55 | 1.30 | 1.29 | 1.28 | 1.28 |
| every pilot | +9 | 92.4 | 31.0 | 30.7 | 34.1 | 27.6 | 27.0 | 27.0 |

**For the paper.** `conference-v11.tex` on Overleaf drops Tables II and III, moves their results into
the text with paired intervals, and shows Fig. 5 in two rows. Table I gives F1 and false alarms with
one and two DMRS symbols for QPSK, 16-QAM and 64-QAM. The Times build is six pages.

## 31. Pictures of the method and the cost of every step (paper v12)

`plan-v12.md`. The author could follow the equations of the pilot weights and of shrinkage but
could not picture them, and asked for the cost and the processing time of the method and how it
fits current 3GPP work. Everything here runs on the Mac. Results are in `results/v12`.

**Pilot weights on one slot** (`figures.pilot_weights`, `results/v12/figures/pilot_weights.pdf`).
QPSK, TDL-C, 15 dB SNR, pilot-only jamming of PRBs 10 to 13 at +6 dB. The jammed PRBs have
residuals of 20.3 to 25.6 dB, the others -2.3 to 6.6 dB. The normalized squared error of the
LMMSE estimate on the jammed PRBs is -1.1 dB with every pilot trusted alike and -8.7 dB with the
pilot weights. On the other PRBs it is -16.3 and -25.5 dB, because without weights the jammed
pilots leak into the estimate of their neighbours. At 0 dB JSR the same numbers are -7.1 and
-13.3 dB on the jammed PRBs and -20.7 and -25.8 dB on the others.

**Shrinkage with its weights** (`figures.shrinkage_example`). The figure now draws the standard
error of every cell estimate, an arrow from each estimate to its shrunk value and the weight
`alpha` of every cell, taken from the receiver's own code (`MapAwareReceiver._shrinkage`). In the
row of the block jammer every weight is 0. In the row of the hopping jammer the weights lie between
0.60 and 1.00, and the 0.60 belongs to the strongest jammed cell, whose estimate has the largest
standard error (3.0 is pulled to 2.0 against a true 3.2).

**Cost per slot** (`experiments.complexity`, `results/v12/complexity/complexity.csv`). 12 PRBs,
QPSK at 8 dB, hopping pilot + data at +9 dB, batch of one, median of 200 runs on one core of the
M1 Pro. Counts are real multiply-accumulates (a complex one counts four), LDPC decoding is counted in
messages along the edges (15 152 edges, 20 iterations, both directions).

| Steps | What | MACs, 1 DMRS | ms, 1 DMRS | MACs, 2 DMRS | ms, 2 DMRS |
| --- | --- | --- | --- | --- | --- |
| 1-3 | LS values, residuals and pilot weights | 10 368 | 0.124 | 20 736 | 0.176 |
| 4 | LMMSE channel estimate with pilot weights | 3 587 328 | 1.388 | 16 174 080 | 3.204 |
| 4 | The same estimate in the delay subspace | 425 376 | 0.234 | 558 432 | 0.247 |
| 4 | LMMSE with a filter fixed per allocation | 41 472 | 0.013 | 82 944 | 0.026 |
| 5-6 | Excess power and shrinkage | 18 720 | 0.360 | 17 280 | 0.351 |
| 7 | Equalization and LLRs | | 1.037 | | 1.027 |
| 7 | LDPC decoding | 606 080 msg | 18.133 | 574 400 msg | 17.131 |
| det. | Features of the U-Net | | 0.389 | | 0.454 |
| det. | U-Net forward pass | 5 451 264 | 0.581 | 5 451 264 | 0.581 |
| det. | U-Net forward pass, frozen graph | 5 451 264 | 0.452 | 5 451 264 | 0.451 |
| 1-7 | Whole receiver without a map | | 21.311 | | 22.116 |
| | Baseline receiver | | 19.685 | | 18.574 |

Readings.

- **The channel estimate is the costly step, and it need not be.** Pilot weights change every slot,
  so the full LMMSE solve (72 x 72 with one DMRS symbol, 144 x 144 with two, plus 144 right-hand
  sides for the error variance) runs every slot. The prior correlation has only 21 eigenvalues
  above 1e-9 of its trace for 12 PRBs. `recovery.lmmse_lowrank` computes the same estimate in that
  subspace (a 21 x 21 solve) and matches the full solve to a relative squared error of about 1e-16,
  also with dropped pilots (tested). It needs 8 times fewer MACs with one DMRS symbol and 29 times
  fewer with two. The BLER runs of the paper use the full solve.
- **What the receiver adds.** Steps 1 to 6 with the low-rank estimate take 0.45 million MACs per
  slot (0.60 million with two DMRS symbols). LDPC decoding of the same slot passes 0.61 million
  edge messages. In this implementation the whole receiver without a map is 8% slower than the
  baseline receiver (21.3 against 19.7 ms), and LDPC decoding takes 18 ms of either.
- **The U-Net** takes 5.45 million MACs, 66 553 parameters, and 0.84 ms for its features and frozen
  forward pass on one core, more than a 0.5 ms slot. A gNB would pipeline it or use an accelerator.

**Detection rate against the JSR** (`experiments.detection_curves`, three seeds of the v11 U-Net,
threshold from `val_all`. The EVM threshold from `val`, as in the decoding runs). Pd over the jammed
cells inside the allocation.

| Grid | Cells | -15 dB | -9 dB | -3 dB | +3 dB | +9 dB |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net, QPSK block | DMRS (pilot only) | 0.497 | 0.821 | 0.950 | 0.990 | 0.996 |
| U-Net, QPSK block, 2 DMRS | DMRS | 0.569 | 0.862 | 0.965 | 0.988 | 0.995 |
| U-Net, QPSK hopping | DMRS | 0.484 | 0.817 | 0.956 | 0.987 | 0.997 |
| EVM threshold, QPSK block | DMRS | 0 | 0 | 0 | 0 | 0 |
| U-Net, QPSK block | data (data only) | 0.719 | 0.928 | 0.991 | 0.996 | 0.999 |
| U-Net, QPSK block, 2 DMRS | data | 0.704 | 0.944 | 0.993 | 0.998 | 1.000 |
| U-Net, QPSK hopping | data | 0.682 | 0.929 | 0.976 | 0.994 | 0.998 |
| U-Net, pulsed (unseen) | data | 0.612 | 0.902 | 0.978 | 0.993 | 0.997 |
| EVM threshold, QPSK block | data | 0.001 | 0.046 | 0.219 | 0.480 | 0.760 |

The EVM numbers agree with section 2 (0, 0.04, 0.21, 0.43 at -15 to +3 dB), and the U-Net's pilot
Pd at -15 dB agrees with section 30 (0.50 and 0.58).

**3GPP and O-RAN context**, checked against the documents themselves (27 Nov).

- TR 38.913 V19.0.0 says the RAN design "shall ensure resilience against jamming". O-RAN WG11
  (ETSI TR 104 106 V3.0.0) lists T-RADIO-01, disruption through radio jamming, sniffing and
  spoofing.
- TR 38.843 V18.0.0 clause 4.3 calls collaboration level x "implementation-based only AI/ML
  algorithms without information exchange". Clause 6.1 asks for complexity in real-valued
  operations (TOPs, FLOPs, MACs) and real-valued parameters. Its model monitoring includes drift
  and out-of-distribution detection of the input and output data.
- Draft TR 38.760-1 V0.8.0 (6G radio, RAN1) studies AI-based PUSCH receivers for DMRS overhead
  reduction and notes that, for a given low-overhead DMRS pattern, the type of receiver (AI or
  not) at the network is transparent to the UE.
- O-RAN fronthaul (ETSI TS 103 859 V17.1.0, clause 9.2.2.4) lets the O-RU report interference plus
  noise for allocated PRBs as one value per PRB, measured on the PUSCH DMRS after subtracting the
  useful signal, for one symbol or averaged over several. No 3GPP or O-RAN measurement covers the
  data symbols cell by cell.
- InterfO-RAN (MobiHoc 2025, DOI 10.1145/3704413.3764454) detects uplink interference from I/Q
  samples with a CNN on a GPU next to NVIDIA Aerial in under 650 us.

**For the paper.** `conference-v12.tex` adds the pilot-weights figure (Fig. 3), the redrawn
shrinkage figure (Fig. 4), the cost subsection with Table II and the low-rank estimate (8), the
detection curves (Fig. 7), the 3GPP and O-RAN sentences and a third contribution. The Times build
is eight pages.
