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
