# Plan v5: explain the propositions, redraw Figs. 1-2, widen the jammer models

18 November 2026, after the readability revision. Three points of feedback on the paper:
Figs. 1 and 2 should be more professional and compact; "do we only have a Gaussian jammer?";
the propositions are hard to interpret ("why do we need them, how are they derived, what do they
show?").

## 1. Propositions: question, statement, meaning, derivation

Every result gets the same four parts: the receiver question it answers, the statement, what it
means (in words and one number), and a short derivation.

| Result | Question it answers | Where the paper uses it |
| --- | --- | --- |
| Proposition 1 (residual without jamming) | How large can the residual get from noise alone? | analytic threshold, block test, DMRS-only receiver, Fig. theory (a) |
| Corollary: threshold and tail | Which cutoff gives 1% false alarms without any training data? | same |
| Proposition 2 (residual under jamming) | How often does that cutoff fire on a jammed PRB, and on a clean neighbor? | Fig. theory (b), pilot weights of the soft receiver |
| Corollary: K slots | How much does waiting for a persistent jammer help? | multi-slot result (figure exists, not yet in the paper) |

- **One picture behind all of them.** Each PRB has six noise samples on its pilots. The projection
  uses up part of them (18 of 72 dimensions for 12 PRBs, more at the allocation edges). The
  residual is the noise energy left in the remaining dimensions, normalized to mean one, so it does
  not depend on SNR or channel. With d = 1 / sum(lambda_i^2) effective dimensions it behaves like a
  Gamma(d, 1/d) variable, the average of d unit exponentials.
- **Small table after Proposition 1** (computed from `theory.py`):

  | 12 PRBs | weights | d | exact 1% threshold | Gamma approximation |
  | --- | --- | --- | --- | --- |
  | middle PRB | .025 .141 .204 .210 .210 .210 | 5.1 | 2.31 | 2.30 |
  | edge PRB | .000 .006 .155 .277 .281 .281 | 3.9 | 2.56 | 2.54 |

  This explains why the edge thresholds are higher: fewer dimensions, larger fluctuations.
- Threshold and K-slot results become corollaries; the bidiagonal generator moves to a footnote
  or appendix. Proposition 2 gets the interpretation "inside the block: detection; next to the
  block: leakage; its mean: the pilot weight of the receiver".
- Add the multi-slot curves as a third panel of the theory figure.

## 2. Figures 1 and 2: one column each, matplotlib

- **Fig. 1 (attack patterns), 3.5 x 2.45 in.** 2 x 2 grids (no jammer, pilot, data, partial-band)
  with shared axes, one patch per region instead of per-cell hatching, labels in orange, and a
  dashed outline on the data cells of the pilot panel: "EVM rises here, label stays 0". Fallback if
  float height is tight: one carrier grid plus three one-row symbol strips.
- **Fig. 2 (receiver), 3.5 x 2.15 in.** Upper lane: received grid -> channel estimate -> LMMSE
  equalizer -> demap -> LDPC. Three taps below it: P from the grid, R from the raw LS pilots,
  E from (x_hat, nu). Taps -> U-Net + threshold -> map Y_hat. Dashed feedback labelled "pilot
  variance" into channel estimation and "data variance" into demapping. UE, jammer, channel and
  noise boxes are dropped; the signal model equation covers them.
- Residual illustration: one column, LS dots and projection on top, R_b per PRB below, shared PRB
  axis.
- Same style for all: 7-8 pt fonts, 0.6-0.8 pt lines, Okabe-Ito colors, readable in grayscale.
  Matplotlib, because the output can be rendered and checked here; TikZ only compiles in Overleaf.

## 3. Jammer models

**What the paper covers today.** Two waveforms: Gaussian (the threat model and Proposition 2)
and random QPSK (unseen test: fused F1 0.951 for both). Six resource patterns: pilot, data,
partial-band, scheduling-aware, pilot comb, symbol hopping. Not covered: tone or narrowband,
pulsed (partial-time), swept, reactive, correlated or spoofing, asynchronous, multi-antenna.

**Paper text.** A short "jammer models" paragraph that separates waveform, time-frequency
support, alignment and attacker knowledge (Lichtman et al. taxonomy), states what is tested and
what is not, and corrects the conclusion ("Gaussian jammer" -> "Gaussian and QPSK, grid-aligned").

**Candidate experiments** (test splits only, existing models, no retraining; about 1-2 h CPU each):

| # | Jammer | Why it matters | Expected result |
| --- | --- | --- | --- |
| E1 | **DMRS replay (pilot spoofing)**: the jammer sends the UE's DMRS sequence, scaled, through its own channel | Its contribution to the LS estimates looks like a physical channel, so it lies in the delay subspace: **the residual is blind by construction**. The strongest attack on our method. | Residual P_d at the false-alarm level; power ratio and EVM (wrong channel estimate) may still reveal it; the fused U-Net partly. Either result is worth reporting. |
| E2 | **Pulsed (partial-time)**: a block of PRBs jammed on a random subset of OFDM symbols | Breaks the "whole PRB row" structure that the U-Net and the block test learned, in a different way than hopping. | Map precision drops for the block test and the nominal U-Net; attack-mixed training should help. |
| E3 | Tone / sparse jammer: a few subcarriers per PRB | Energy diluted over 12 REs per cell; tests power/EVM averaging. | Detected later in JSR; residual sees it only on pilots. |

Asynchronous (off-grid) jammers and multiple receive antennas remain follow-up work (several
days each, need a time-domain link and a new label convention).

**New references (DOIs resolved via Crossref):** Lichtman et al., "A communications jamming
taxonomy", IEEE S&P 2016, 10.1109/MSP.2016.13; Tugnait, "Pilot spoofing attack detection and
countermeasure", IEEE TCOM 2018, 10.1109/TCOMM.2018.2797989; Shahriar et al., "Performance impact
of asynchronous off-tone jamming attacks against OFDM", IEEE ICC 2013, 10.1109/ICC.2013.6654850.

## 4. Order and checks

1. Propositions rewrite + table + multi-slot panel (text, about half a day).
2. Figs. 1, 2 and the residual illustration (render, inspect at column width and in grayscale).
3. Jammer paragraph and limitation; experiments E1 (and E2/E3 if wanted): code + tests, test
   splits, evaluation of all detectors, results in `docs/results.md`.
4. Independent review of the new text and code, numbers checked against the CSVs, push to
   Overleaf and GitHub.

## Status

- [ ] 1. Propositions
- [ ] 2. Figures
- [ ] 3. Jammer models (E1, E2, E3 to be decided)
- [ ] 4. Review and push
