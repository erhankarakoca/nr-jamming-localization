# Plan for paper v12

Written on 27 November 2026, after the author read v11 and asked for three things. The paper may
grow to about 10 pages while we decide what earns its place, without padding.

1. **Cost and practicality.** A short account of the computational cost and the processing time,
   and how the method fits current and coming 3GPP work, so that readers see it is practical.
2. **Pictures of the method.** The author could follow the equations of the pilot weights and of
   shrinkage but could not picture them. Each should come with a figure that shows the idea.
3. **Figures and statements that matter.** Room for the important ones, no verbiage.

The ResNet stays out. `conference-v12.tex` starts as a copy of v11, so v11 stays as it is.

## What v12 adds, most valuable first

| Item | Where | Status |
| --- | --- | --- |
| Pilot weights on one slot (new figure) | `figures.pilot_weights` | done, Fig. 3 |
| Shrinkage with error bars, arrows and the weight of every cell (Fig. 3 redrawn) | `figures.shrinkage_example` | done, Fig. 4 |
| The receiver's steps with their order and time on one core (new table) | `experiments.complexity` | done, Table II, with the low-rank estimate |
| 3GPP and deployment paragraph | paper text, sources from a search | done, sources checked |
| Detection rate against JSR for pilot and data cells (new figure) | `experiments.detection_curves` | done, Fig. 7 |
| Related work set side by side (new table) | paper | left out, four works readable only as abstracts |
| One plain sentence before each equation block, subsection renamed "Shrinkage Along the PRB" | paper | done |

An algorithm box was first left out because Fig. 1 and the cost table already number the steps.
After reading v12 the author asked for one. Algorithm 1 lists the steps for one slot with the
equation of each and the two lines a map changes, and it replaces the overview paragraph at the
start of Section III.

## Decisions

- **Pilot weights figure.** One QPSK slot on TDL-C at 15 dB SNR, pilot-only jamming of PRBs 10
  to 13 at +6 dB JSR (as Fig. 2). Three stacked panels over the PRBs of the allocation. (a) The LS
  value of every pilot, the true channel and the smooth fit (projection onto the delay subspace).
  (b) The residual of each PRB in dB, with 0 dB where noise alone puts it. (c) The channel
  estimate with every pilot trusted alike and with the pilot weights, next to the true channel.
- **Shrinkage figure.** The rows of v11 Fig. 3 (block and hopping jammer, +9 dB, 15 dB SNR). The
  cell estimates get error bars of one standard error, an arrow runs from each cell estimate to its
  shrunk value, and a thin panel below shows the weight of every cell. The weight and the standard
  error come from the receiver's own code (`MapAwareReceiver._shrinkage`), not from a copy.
- **Cost table.** Rows follow the steps of Fig. 1(b). 1 and 2 (LS values, residual), 3 and 4
  (weighted LMMSE), 5 and 6 (excess power, shrinkage), the detector (features, U-Net), and LDPC
  decoding for scale. Columns give the order of the operations, the count for 12 PRBs and the
  median time for one slot on one core of the M1 Pro, with one and with two DMRS symbols. The U-Net
  takes 5.45 million multiply-accumulates per slot.
- **Detection curves.** The v11 U-Net (three seeds, threshold from `val_all`) and the EVM
  threshold on the local grids `test`, `test_dmrs2`, `hopjam` and `pulsejam`. Pd of jammed DMRS
  cells under pilot-only jamming and of jammed data cells under data-only jamming, from -15 to
  +9 dB JSR.
- **3GPP.** Claims enter the paper only with a source that was checked. The search (Codex) covers
  the AI/ML study and work items of Rel-18 to Rel-20, whether a model inside the gNB needs any
  specification, the gNB's interference measurements per PRB, jamming in 3GPP and O-RAN, gNB
  processing budgets and the cost of LDPC decoding.
- **Related work table.** Each entry is checked against the paper it describes before it goes in.

## Checks

- Every new function has a test in `tests/test_v12.py`, and the whole suite passes.
- Figures are drawn into `results/v12` and copied to `figures/v12` of the paper.
- The Times build of `conference-v12.tex` compiles without overfull boxes, and its page count is
  reported.

## Outcome

- The search for the related-work table read 11 of the 15 works in full. Both EVM papers, Mehrabian
  and Kaddoum, and Han et al. were readable only as abstracts, so several cells of the table would
  have been guesses. The table stays out. The reading corrected three sentences of the
  introduction (Yoshii estimates interference for each modulation symbol, not each OFDM symbol, the
  2022 EVM paper works per resource block, and Kakizaki refines interference estimates, not the
  channel, with decoded symbols).
- The cost table showed that the full LMMSE solve with pilot weights is the costliest step of the
  receiver. `recovery.lmmse_lowrank` gives the same estimate in the delay subspace with 8 to 29
  times fewer operations. The BLER runs still use the full solve.
- The Times build of `conference-v12.tex` has eight pages, also with Algorithm 1.
- The excess power is defined as the received power minus the expected power of the user's
  signal and the noise. The old phrase about what the signal and the noise do not explain is
  gone from v12 and the README.
